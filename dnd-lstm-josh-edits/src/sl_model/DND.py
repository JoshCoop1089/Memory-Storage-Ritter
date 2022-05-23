# from hashlib import new
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                exp_settings):
        # params
        self.dict_len = dict_len
        self.kernel = exp_settings['kernel']
        self.hidden_lstm_dim = hidden_lstm_dim

        # dynamic state
        self.encoding_off = False
        self.retrieval_off = False

        # allocate space for per trial hidden state buffer
        self.trial_buffer = [()]

        # Experimental changes
        self.exp_settings = exp_settings

        # allocate space for memories
        self.reset_memory()
        # check everything
        self.check_config()

    def reset_memory(self):
        self.keys = [[]]
        self.vals = []
        # self.keys = [[] for _ in range(self.exp_settings['num_barcodes'])]
        # self.vals = [0 for _ in range(self.exp_settings['num_barcodes'])]
        self.key_context_map = {}
        self.context_counter = 0
        self.recall_sims = []

        # Embedding model (reset every epoch due to changing contexts)
        self.embedder = Embedder(self.exp_settings)
        learning_rate = 5e-4
        self.embed_optimizer = torch.optim.Adam(self.embedder.parameters(), lr=learning_rate)
        
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
        """Save an episodic memory to the dictionary

        Parameters
        ----------
        memory_key : a row vector
            a DND key, used to for memory search
        memory_val : a row vector
            a DND value, representing the memory content
        """
        # ----During Trial----
        # If not at end of episode return to training
        if self.encoding_off:
            return

        # ----End of Trial----
        # Embedding Model should get frozen after saving before going back to main LSTM agent
        for name, param in self.embedder.named_parameters():
            if param.requires_grad:
                print (name, param.data)

        # # Reached end of episode, different ways to update memory
        # mem_update_boolean = self.exp_settings['kaiser_key_update']

        keys = self.trial_buffer
        trial_hidden_states = [keys[i] for i in range(len(keys)//4, len(keys), 2)]

        """
        # # Trying different versions of Kaiser Update averaging
        # if mem_update_boolean:
        #     # Update way of storing queries in self.trial_buffer
        #         # get_memory will automatically store a (hidden_state, best_memory_match_id) tuple
        #         # Use -1 for best_memory_match_id to indicate no valid pulls for a hidden state query
        #     # Go through all queries in trial_hidden_state
        #         # Find any instances which might have had a valid pull on the same memory?
        #     # Two options based on sim thresh
        #         # Query was a good pull
        #             # Find memory which matched to query
        #             # Create updated key with Kaiser avg
        #             # Append new updated key to front of list, where list is made from all updates on some original memory
        #         # Query wasn't a good pull
        #             # Make new list, and append at end of memory buffer

        #     new_keys = []
        #     ids = []

        #     # Find any multi pulls on same memory (also handles single pulls)
        #     unique_id = set([int(x[1]) for x in trial_hidden_states if x[1] != -1])
        #     unique_id_dict = {key: [] for key in unique_id}
        #     for state, mem_id in trial_hidden_states:
        #         if mem_id != -1:
        #             unique_id_dict[mem_id].append(state)


        #     # Modified for exp_settings dict version (not using multiple possible update types anymore, will probably be phased out with embedding model)
        #     # Create single updated memory from all pulls on same id using most recent memory update version
        #     # if 'update_avg' in self.change:
        #     for id, vals in unique_id_dict.items():
        #         new_key = sum(vals) + self.keys[id][0]
        #         new_key2 = new_key/F.normalize(new_key, p=1, dim=1)
        #         new_keys.append(torch.squeeze(new_key2.data))
        #         ids.append(id)

        #     # # Avg each instance with original, then avg together all repeats
        #     # elif 'update_ind_avg' in self.change:
        #     #     for id, vals in unique_id_dict.items():
        #     #         avs = []
        #     #         for val in vals:
        #     #             new_key = self.keys[id][0] + val
        #     #             new_key2 = new_key/F.normalize(new_key, p=1, dim=1)
        #     #             avs.append(new_key2)
        #     #         avg_key = sum(avs)

        #     #         # If there is a single pull, don't renormalize over sum of avgs
        #     #         if len(avg_key == 1):
        #     #             new_key2 = avg_key[0]
        #     #         else:
        #     #             new_key2 = avg_key/F.normalize(avg_key, p=1, dim=1)
                        
        #     #         new_keys.append(torch.squeeze(new_key2.data))
        #     #         ids.append(id)

        #     # Take new updated memory and add it to  list of updates to original memory
        #     for id, updated_key in zip(ids, new_keys):
        #         mem_history = self.keys[id]
        #         new_mem_history = [updated_key] + mem_history

        #         # Put the new updated memory list back into main memory holder
        #         self.keys.append(new_mem_history)
        #         self.vals.append(torch.squeeze(memory_val.data))
            
        #     # Find all pulls which didn't match in memory, and put them in memory
        #     no_matches = [x[0] for x in trial_hidden_states if x[1] == -1]
        #     for x in no_matches:
        #         self.keys.append([torch.squeeze(x.data)])
        #         self.vals.append(torch.squeeze(memory_val.data))

        #     # Clean up memory of all replaced keys
        #     temp_keys = [self.keys[id] for id in range(len(self.keys)) if id not in ids]
        #     self.keys = temp_keys
        #     temp_vals = [self.vals[id] for id in range(len(self.vals)) if id not in ids]
        #     self.vals = temp_vals

        #     # Trim memory if it is over the expected length
        #     # Older non-updated memories will be at the front
        #     while len(self.keys) > self.dict_len:
        #         self.keys.pop(0)
        #         self.vals.pop(0)

        # # Embedding Model Updates
        # else:
        """

        # Trial buffer contained embedding,location from get_memory
        for embedding, context_location in trial_hidden_states:
            old_emb = self.keys[context_location]
            self.keys[context_location] = [torch.squeeze(embedding.data)] + old_emb
            self.vals[context_location] = torch.squeeze(memory_val.data)

    def get_memory(self, query_key, context_label):
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
        # Embedding Model Testing Ground
        agent = self.embedder

        # Unfreeze Embedder to train
        for param in agent.parameters():
            param.requires_grad = True

        embedding, predicted_context = agent(query_key)
        # print("Raw:", predicted_context)
        # print("Embedding:", embedding)
        
        # Update pass over the embedding model
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predicted_context, context_label)
        # print("Loss: ", loss)
        self.embed_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.embed_optimizer.step()
       
        # Predicted context will be tensor of floats, need to transform into binarystring for key mapping
        predicted_context = torch.where(predicted_context > 0.5, 1, 0)  
        # print('P-CTX:', predicted_context)
        # barcode -> string starts out at '[[1 1 0]]', thus the reductions on the end
        predicted_context = np.array2string(predicted_context.numpy())[2:-2].replace(" ", "")

        # Task not yet seen, no stored LSTM yet
        if predicted_context not in self.key_context_map:
            self.key_context_map[predicted_context] = self.context_counter
            context_location = self.context_counter
            self.keys.append([])
            self.vals.append(0)
            self.context_counter += 1
            best_memory_val = _empty_memory(self.hidden_lstm_dim)

        # Task seen, get LSTM attached to task
        else:
            context_location = self.key_context_map[predicted_context]
            best_memory_val = self.vals[context_location]
            # First reference of a task, but before LSTM is saved to memory
            if type(best_memory_val) == int:
                best_memory_val = _empty_memory(self.hidden_lstm_dim)

        # Store embedding and predicted class label in trial_buffer
        self.trial_buffer.append((embedding, context_location))

        # Freeze Embedder model until next memory retrieval
        for param in agent.parameters():
            param.requires_grad = False

        return best_memory_val, predicted_context

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
            best_memory_id = int(torch.argmax(similarities))
            # self.recall_sims.append(similarities[best_memory_id].detach().numpy())
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

def _empty_memory(memory_dim):
    """Get a empty memory, assuming the memory is a row vector
    """
    return torch.zeros(1, memory_dim)