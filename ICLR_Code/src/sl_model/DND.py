import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as AMP
from scipy import stats as st
from sl_model.embedding_model import Embedder

import numpy as np

# constants
ALL_KERNELS = ['cosine', 'l1', 'l2']
ALL_POLICIES = ['1NN']

class DND():
    """The differentiable neural dictionary (DND) class. This enables episodic
    recall in a neural network.

    notes:
    - a memory is a row vector

    Parameters
    ----------
    dict_len : int
        the maximial len of the dictionary
    memory_dim : int
        the dim or len of memory i, we assume memory_i is a row vector
    kernel : str
        the metric for memory search

    Attributes
    ----------
    encoding_off : bool
        if True, stop forming memories
    retrieval_off : type
        if True, stop retrieving memories
    reset_memory : func;
        if called, clear the dictionary
    check_config : func
        check the class config

    """

    def __init__(self, dict_len, hidden_lstm_dim,
                exp_settings, device):
        # params
        self.dict_len = dict_len
        self.kernel = exp_settings['kernel']
        self.hidden_lstm_dim = hidden_lstm_dim
        self.mapping = {}
        self.device = device

        # dynamic state
        self.encoding_off = False
        self.retrieval_off = False

        # Non Embedder Memory Sizes
        self.mem_store = exp_settings['mem_store']
        if self.mem_store == 'obs/context':
            self.mem_input_dim = exp_settings['num_arms'] + exp_settings['barcode_size']
        elif self.mem_store == 'context':
            self.mem_input_dim = exp_settings['barcode_size']

        # These two won't work for barcode tasks without figuring out a way to get the barcode
        # Cheat and get it from the input?
        elif self.mem_store == 'obs':
            self.mem_input_dim = exp_settings['num_arms']
        elif self.mem_store == 'hidden':
            self.mem_input_dim = exp_settings['dim_hidden_lstm']

        # Experimental changes
        self.exp_settings = exp_settings
        self.epoch_counter = 0
        self.embedder_loss = np.zeros((exp_settings['epochs'] + exp_settings['noise_eval_epochs']*len(exp_settings['noise_percent'])))

        # allocate space for per trial hidden state buffer
        self.trial_buffer = [()]

        if self.mem_store == 'embedding':
            # Embedding model
            self.embedder = Embedder(self.exp_settings, device = self.device)
            learning_rate = exp_settings['embedder_learning_rate']
            self.embed_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.embedder.parameters()), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        # allocate space for memories
        self.reset_memory()
        # check everything
        self.check_config()

    def reset_memory(self):
        self.keys = [[]]
        self.vals = []
        self.trial_hidden_states = []
        self.key_context_map = {}
        self.context_counter = 0 
        self.sorted_key_list = sorted(list(self.mapping.keys()))
        
    def check_config(self):
        assert self.dict_len > 0
        assert self.kernel in ALL_KERNELS

    def inject_memories(self, input_keys, input_vals):
        """Inject pre-defined keys and values

        Parameters
        ----------
        input_keys : list
            a list of memory keys
        input_vals : list
            a list of memory content
        """
        assert len(input_keys) == len(input_vals)
        for k, v in zip(input_keys, input_vals):
            self.save_memory(k, v)

    def save_memory(self, memory_key, memory_val):
        try:
            test = self.keys[0][0]
        except IndexError:
            self.keys.pop(0)
        # Save every embedding of the trial
        self.trial_buffer.pop(0)
        keys = self.trial_buffer
        self.trial_hidden_states = [keys[i] for i in range(len(keys)) if keys[i] != () and i > len(keys)//4]

        # Append new memories at head of list to allow sim search to find these first
        for embedding, context_location, _, real_bc in self.trial_hidden_states:
            self.keys = [[torch.squeeze(embedding.data), context_location, real_bc]] + self.keys
            self.vals = [torch.squeeze(memory_val.data)] + self.vals

        while len(self.keys) > self.dict_len:
            self.keys.pop()
            self.vals.pop()
        return

    def save_memory_non_embedder(self, memory_key, barcode_tensor, memory_val):

        # Noisy barcodes are the memory_key, barcode_tensor is holding the ground truth barcode
        barcode_string = np.array2string(barcode_tensor.cpu().numpy())[2:-2].replace(" ", "").replace(".", "")

        try:
            test = self.keys[0][0]
        except IndexError:
            self.keys.pop(0)

        # Append new memories at head of list to allow sim search to find these first
        self.keys = [[(torch.squeeze(memory_key.data)), barcode_string]] + self.keys
        self.vals = [torch.squeeze(memory_val.data)] + self.vals

        # remove the oldest memory, if overflow
        if len(self.keys) > self.dict_len:
            self.keys.pop()
            self.vals.pop()
        return
        
    def get_memory(self, query_key, real_label_as_string, real_label_id):
        """
        Embedder memory version:

        Takes an input hidden state (query_key) and a ground truth barcode (real_label_as_string)
        Passes the query key into the embedder model to get the predicted barcode
        Uses self.key_context_map and the predicted barcode to retrieve the LSTM state stored for that barcode

        Also handles the embedder model updates, and buffering information for the save_memory function at the end of the episode
        """

        # Embedding Model Testing Ground
        agent = self.embedder

        # Unfreeze Embedder to train
        for name, param in agent.named_parameters():
            param.requires_grad = True

        # Model outputs class probabilities
        embedding, model_output = agent.forward(query_key)

        # Treat model as predicting a single id for a class label, based on the order in self.sorted_key_list
        # Calc Loss for single pull for updates at end of episode
        emb_loss = self.criterion(model_output, real_label_id)

        # Freeze Embedder model until next memory retrieval
        for name, param in agent.named_parameters():
            param.requires_grad = False

        # Get class ID number for predicted barcode
        soft = torch.softmax(model_output, dim=1)
        pred_memory_id = torch.argmax(soft)
        predicted_context = self.sorted_key_list[pred_memory_id]

        key_list = [self.keys[x][0] for x in range(
            len(self.keys)) if self.keys[x] != []]

        # Something is stored in memory
        if key_list:
            similarities = compute_similarities(embedding, key_list, self.kernel)
                
            # get the best-match memory
            best_memory_val, _ = self._get_memory(similarities)
        
        # If nothing is stored in memory yet, return 0's
        else:
            self.trial_buffer.append((embedding, predicted_context, emb_loss, real_label_as_string))
            return _empty_memory(self.hidden_lstm_dim, device=self.device), _empty_barcode(self.exp_settings['barcode_size'])
        
        # Store embedding and predicted class label memory index in trial_buffer
        self.trial_buffer.append((embedding, predicted_context, emb_loss, real_label_as_string))
        return best_memory_val, predicted_context

    def get_memory_non_embedder(self, query_key):
        """Perform a 1-NN search over dnd

        Parameters
        ----------
        query_key : a row vector
            a DND key, used to for memory search

        Returns
        -------
        a row vector
            a DND value, representing the memory content

        """
        try:
            test = self.keys[0][0]
            n_memories = len(self.keys)
        except IndexError:
            n_memories = 0

        # if no memory, return the zero vector
        if n_memories == 0 or self.retrieval_off:
            return _empty_memory(self.hidden_lstm_dim, self.device), _empty_barcode(self.exp_settings['barcode_size'])
        else:
            # compute similarity(query, memory_i ), for all i
            key_list = [self.keys[x][0] for x in range(len(self.keys))]
            similarities = compute_similarities(query_key, key_list, self.kernel)

            # get the best-match memory
            best_memory_val, best_memory_id = self._get_memory(similarities)

            # get the barcode for that memory
            barcode = self.keys[best_memory_id][1]

            return best_memory_val, barcode

    def _get_memory(self, similarities, policy='1NN'):
        """get the episodic memory according to some policy
        e.g. if the policy is 1nn, return the best matching memory
        e.g. the policy can be based on the rational model

        Parameters
        ----------
        similarities : a vector of len #memories
            the similarity between query vs. key_i, for all i
        policy : str
            the retrieval policy

        Returns
        -------
        a row vector
            a DND value, representing the memory content
        """
        best_memory_val = None
        if policy == '1NN':
            best_memory_id = torch.argmax(similarities)
            best_memory_val = self.vals[best_memory_id]
        else:
            raise ValueError(f'unrecog recall policy: {policy}')
        return best_memory_val, best_memory_id

"""helpers"""
def compute_similarities(query_key, key_list, metric):
    """Compute the similarity between query vs. key_i for all i
        i.e. compute q M, w/ q: 1 x key_dim, M: key_dim x #keys

    Parameters
    ----------
    query_key : a vector
        Description of parameter `query_key`.
    key_list : list
        Description of parameter `key_list`.
    metric : str
        Description of parameter `metric`.

    Returns
    -------
    a row vector w/ len #memories
        the similarity between query vs. key_i, for all i
    """
    # reshape query to 1 x key_dim
    q = query_key.data.view(1, -1)
    # reshape memory keys to #keys x key_dim
    M = torch.stack(key_list)
    # compute similarities
    if metric == 'cosine':
        similarities = F.cosine_similarity(q, M)
    elif metric == 'l1':
        similarities = - F.pairwise_distance(q, M, p=1)
    elif metric == 'l2':
        similarities = - F.pairwise_distance(q, M, p=2)
    else:
        raise ValueError(f'unrecog metric: {metric}')
    return similarities

def _empty_memory(memory_dim, device):
    """Get a empty memory, assuming the memory is a row vector
    """
    return torch.squeeze(torch.zeros(memory_dim, device=device))

def _empty_barcode(barcode_size):
    """Get a empty barcode, and pass it back as a string for comparison downstream
    """
    empty_bc = "0"*barcode_size
    return empty_bc