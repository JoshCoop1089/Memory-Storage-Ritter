# # from hashlib import new
# from operator import index
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
        self.embedder_loss = np.zeros((exp_settings['epochs']))

        # allocate space for per trial hidden state buffer
        self.trial_buffer = [()]

        if self.mem_store == 'embedding':
            # Embedding model
            self.embedder = Embedder(self.exp_settings, device = self.device)
            learning_rate = exp_settings['embedder_learning_rate']
            self.embed_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.embedder.parameters()), lr=learning_rate)

        # allocate space for memories
        self.reset_memory()
        # check everything
        self.check_config()

    def reset_memory(self):
        self.keys = [[]]
        self.vals = []
        self.recall_sims = []     
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
        # Embedding Model should get frozen after get_memory so this should never print anything
        for name, param in self.embedder.named_parameters():
            if param.requires_grad:
                print (name, param.data)

        # # Reached end of episode, different ways to update memory
        # mem_update_boolean = self.exp_settings['kaiser_key_update']

        # Save full buffer per trial
        keys = self.trial_buffer
        trial_hidden_states = [keys[i] for i in range(len(keys)) if keys[i] != ()]
        # print(trial_hidden_states)

        # trial_hidden_states = [keys[i] for i in range(len(keys)//4, len(keys), 2)]

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

        try:
            # Trial buffer contained embedding,location from get_memory
            context_net = np.zeros(self.exp_settings['pulls_per_episode'])
            for idx, (embedding, context_location, _) in enumerate(trial_hidden_states):
                context_net[idx] = context_location

                # Store embeddings in slot predicted by barcode
                old_emb = self.keys[context_location]
                self.keys[context_location] = [torch.squeeze(embedding.data)] + old_emb
            
            # Find most often predicted barcode for trial and store LSTM state in that slot
            # There should ideally only be one prediction if embedder is perfect
            context_avg = int(st.mode(context_net)[0][0])
            self.vals[context_avg] = torch.squeeze(memory_val.data)

        except Exception as e:
            print(e)
            pass

        # Embedding Model Loss Backprop Time
        agent = self.embedder
        loss_vals = [trial_hidden_states[i][2] for i in range(len(trial_hidden_states))]
        episode_loss = torch.stack(loss_vals).mean()
        self.embedder_loss[self.epoch_counter] += (episode_loss/self.exp_settings['episodes_per_epoch'])

        # Unfreeze Embedder to train
        for name, param in agent.named_parameters():
            # print(name, param.grad)
            param.requires_grad = True

        self.embed_optimizer.zero_grad()
        episode_loss.backward(retain_graph=True)
        self.embed_optimizer.step()

        # Freeze Embedder until next memory retrieval
        for name, param in agent.named_parameters():
            # print(name, param.grad)
            param.requires_grad = False

    def save_memory_non_embedder(self, memory_key, barcode_string, memory_val):

        # ----During Trial----
        # If not at end of episode return to training
        if self.encoding_off:
            return

        # All input is stored in memory (QiHong Github version)
        if self.exp_settings['mem_store'] == 'obs/context':
            self.keys.append([torch.squeeze(memory_key.data), barcode_string])
            # Dumb error catching due to how i organize the data for our version
            try:
                test = self.keys[0][0]
            except IndexError:
                self.keys.pop(0)
            self.vals.append(torch.squeeze(memory_val.data))
            # remove the oldest memory, if overflow
            if len(self.keys) > self.dict_len:
                self.keys.pop(0)
                self.vals.pop(0)
            return

        # Only context is stored in memory (Ritter Version)
        elif self.exp_settings['mem_store'] == 'context':
            try:
                key_barcodes = [self.keys[x][0][1] for x in range(len(self.keys))]

                # Is the barcode already in memory?
                try:
                    best_memory_id = key_barcodes.index(barcode_string)
                    self.vals[best_memory_id] = torch.squeeze(memory_val.data)

                # Barcode not in memory, store new memory
                except Exception:                    
                    self.keys.append([(torch.squeeze(memory_key.data), barcode_string)])
                    self.vals.append(torch.squeeze(memory_val.data))

                # Dont have to use tensor stored because now we have barcode strings
                # key_list = [self.keys[x][0][0] for x in range(len(self.keys))]
                # # print("Keys:", key_list)
                # similarities = compute_similarities(
                #     memory_key, key_list, self.kernel)

                # prev_seen_task = torch.gt(torch.max(similarities), 0.9).item()
                # if prev_seen_task:
                #     # print("Key seen before, replacing LSTM")
                #     # get the best-match memory
                #     _, best_memory_id = self._get_memory(similarities)
                #     self.vals[best_memory_id] = torch.squeeze(memory_val.data)
                # else:
                #     # print("New Key Main Branch:", memory_key.data)
                #     self.keys.append([(torch.squeeze(memory_key.data), barcode_string)])
                #     self.vals.append(torch.squeeze(memory_val.data))

            # It's the first episode of the epoch and I'm abusing try/except
            except IndexError:
                self.keys.pop(0)
                # print("New Key Exception Branch:", memory_key.data)
                self.keys.append([(torch.squeeze(memory_key.data), barcode_string)])
                self.vals.append(torch.squeeze(memory_val.data))

        # remove the oldest memory, if overflow
        if len(self.keys) > self.dict_len:
            self.keys.pop(0)
            self.vals.pop(0)
        # print("Key List Short:", self.keys, len(self.keys))
        return
        
    def get_memory(self, query_key, real_label_as_string):
        """
        Embedder memory version:

        Takes an input hidden state (query_key) and a ground truth barcode (context_label)
        Passes the query key into the embedder model to get the predicted barcode
        Uses self.key_context_map and the predicted barcode to retrieve the LSTM state stored for that barcode

        Also handles the embedder model updates, and buffering information for the save_memory function at the end of the episode
        """

        # Get class ID number for real barcode
        real_label_id = torch.tensor([self.sorted_key_list.index(real_label_as_string)], dtype = torch.long, device = self.device)

        # Embedding Model Testing Ground
        agent = self.embedder

        # Unfreeze Embedder to train
        for name, param in agent.named_parameters():
            # print(name, param.grad)
            param.requires_grad = True

        # Model outputs class probabilities
        embedding, model_output = agent(query_key)
        # print("*** Getting New Memory ***")
        # print("Raw:", model_output)
        # print("Embedding:", embedding)     

        # treat model as predicting a single id for a class label, based on the order in self.sorted_key_list

        # Calc Loss for single pull for updates at end of episode
        criterion = nn.CrossEntropyLoss().to(self.device)
        emb_loss = criterion(model_output, real_label_id)

        # Freeze Embedder model until next memory retrieval
        for name, param in agent.named_parameters():
            # print(name, param.grad)
            param.requires_grad = False

        # print("*** Got New Memory ***")

        # Output barcode as string for downstream use
        # Get class ID number for predicted barcode
        soft = torch.softmax(model_output, dim=1)
        # print(soft)
        best_memory_id = torch.argmax(soft)
        # print(best_memory_id)
        # print(key_list)
        predicted_context = self.sorted_key_list[int(best_memory_id)]
        # print('P-CTX:', predicted_context)
        # print('R-CTX:', real_label_as_string)

        # Task not yet seen, no stored LSTM yet
        if predicted_context not in self.key_context_map:
            self.key_context_map[predicted_context] = self.context_counter
            context_location = self.context_counter
            self.keys.append([])
            self.vals.append(0)
            self.context_counter += 1
            best_memory_val = _empty_memory(
                self.hidden_lstm_dim, device=self.device)

        # Task seen before, get LSTM attached to task
        else:
            context_location = self.key_context_map[predicted_context]
            best_memory_val = self.vals[context_location]

            # Task was ID'd in this epoch already, but there hasn't been an LSTM stored for it yet
            if type(best_memory_val) is int:
                best_memory_val = _empty_memory(self.hidden_lstm_dim, device = self.device)

        # Store embedding and predicted class label memory index in trial_buffer
        self.trial_buffer.append((embedding, context_location, emb_loss))
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
            key_list = [self.keys[x][0][0] for x in range(len(self.keys))]
            # print("Keys:", key_list)
            similarities = compute_similarities(query_key, key_list, self.kernel)

            # get the best-match memory
            best_memory_val, best_memory_id = self._get_memory(similarities)

            # get the barcode for that memory
            try:
                barcode = self.keys[best_memory_id][0][1][0]
            except Exception:
                barcode = ''

            # # Split the stored item to get the barcode if needed
            # if self.exp_settings['mem_store'] == 'obs/context':
            #     barcode = key_stored[self.exp_settings['barcode_size']:]
            # else:
            #     barcode = key_stored
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
            # print("Sims:" , similarities)
            # # reverse the similarities list to capture most recent memory
            # rev_sims = torch.flip(similarities.view(1, similarities.size(0)), dims = (0, 1))
            # best_memory_id = rev_sims.shape[1] - 1 - torch.argmax(rev_sims)
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