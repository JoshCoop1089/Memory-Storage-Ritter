"""
A DND-based LSTM based on ...
Ritter, et al. (2018).
Been There, Done That: Meta-Learning with Episodic Recall.
Proceedings of the International Conference on Machine Learning (ICML).
"""
import time
import torch
import torch.nn as nn
from sl_model.DND import DND
from sl_model.A2C import A2C_linear, A2C

# constants
N_GATES = 4

class DNDLSTM(nn.Module):

    def __init__(self, 
                    dim_input_lstm, dim_hidden_lstm, dim_output_lstm,
                    dict_len, exp_settings, device, bias=True):
        super(DNDLSTM, self).__init__()
        self.input_dim = dim_input_lstm
        self.dim_hidden_lstm = dim_hidden_lstm
        self.dim_hidden_a2c = exp_settings['dim_hidden_a2c']
        self.bias = bias
        self.device = device
        self.exp_settings = exp_settings
        # input-hidden weights
        self.i2h = nn.Linear(dim_input_lstm, (N_GATES+1)
                             * dim_hidden_lstm, bias=bias, device = self.device)
        # hidden-hidden weights
        self.h2h = nn.Linear(dim_hidden_lstm, (N_GATES+1) * dim_hidden_lstm, bias=bias, device = self.device)
        # dnd
        self.dnd = DND(dict_len, dim_hidden_lstm, exp_settings, self.device)
        #policy
        # self.a2c = A2C_linear(dim_hidden_lstm, dim_output_lstm).to(self.device)
        self.a2c = A2C(dim_hidden_lstm, self.dim_hidden_a2c, dim_output_lstm, device = self.device)

        # For some reason, if this is activated, the Embedder never learns, even though the embedder layers arent touched by this
        # init
        # self.reset_parameter()

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            # print(name)
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)

    def forward(self, obs_bar_reward, barcode_string, barcode_tensor, barcode_id, h, c):

        forward_start = time.perf_counter()

        # Into LSTM
        if self.exp_settings['agent_input'] == 'obs/context':
            x_t = obs_bar_reward
        elif self.exp_settings['agent_input'] == 'obs':
            # This would only be needed for QiHongs og task, and I don't think this will work like I want it to
            x_t = obs_bar_reward[0][:self.exp_settings['num_arms']].view(1, self.exp_settings['num_arms'])
        else:
            raise ValueError('Incorrect agent_input type')
        # print(x_t)

        # Used for memory search/storage (non embedder versions)
        if self.exp_settings['mem_store'] != 'embedding':
            if self.exp_settings['mem_store'] == 'context':
                q_t = barcode_tensor
            elif self.exp_settings['mem_store'] == 'obs/context':
                q_t = obs_bar_reward
            else:
                raise ValueError('Incorrect mem_store type')
        
        if self.exp_settings['timing']:
            forward_prep = time.perf_counter() - forward_start

        # transform the input info
        Wx = self.i2h(x_t)
        Wh = self.h2h(h)
        preact = Wx + Wh

        if self.exp_settings['timing']:
            forward_preact = time.perf_counter() - forward_prep - forward_start

        # get all gate values
        gates = preact[:, : N_GATES * self.dim_hidden_lstm].sigmoid()

        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.dim_hidden_lstm]
        i_t = gates[:, self.dim_hidden_lstm:2 * self.dim_hidden_lstm]
        o_t = gates[:, 2*self.dim_hidden_lstm:3 * self.dim_hidden_lstm]
        r_t = gates[:, -self.dim_hidden_lstm:]

        # stuff to be written to cell state
        c_t_new = preact[:, N_GATES * self.dim_hidden_lstm:].tanh()

        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_t_new)
        
        if self.exp_settings['timing']:
            forward_gate = time.perf_counter() - forward_prep - forward_preact - forward_start

        if self.exp_settings['mem_store'] == 'embedding':
            # Freeze all LSTM Layers before getting memory
            layers = [self.i2h, self.h2h, self.a2c]
            for layer in layers:
                for name, param in layer.named_parameters():
                    param.requires_grad = False 
                # print(name, param.data, param.grad)

            # print("B-Retrieve:\n", self.a2c.critic.weight)
            # print("B-Retrieve:\n", self.dnd.embedder.e2c.weight)
    
            # Query Memory (hidden state passed into embedder, barcode_string used for embedder loss function)
            mem, predicted_barcode, mem_predicted_bc = self.dnd.get_memory(h, barcode_string, barcode_id)
            m_t = mem.tanh()

            # print("A-Retrieve:\n", self.a2c.critic.weight)
            # print("A-Retrieve:\n", self.dnd.embedder.e2c.weight)

            # Unfreeze LSTM
            for layer in layers:
                for name, param in layer.named_parameters():
                    param.requires_grad = True 
                # print(name, param.data)

        else:
            mem, predicted_barcode, mem_predicted_bc = self.dnd.get_memory_non_embedder(q_t)
            m_t = mem.tanh()
            # print("A:", self.a2c.critic.weight.data, self.a2c.critic.weight.grad)
        
        if self.exp_settings['timing']:
            forward_get_mem = time.perf_counter() - forward_prep - forward_preact - forward_gate - forward_start

        # gate the memory; in general, can be any transformation of it
        c_t = c_t + torch.mul(r_t, m_t)
        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, c_t.tanh())

        forward_save = 0
        # Saving memory happens once at the end of every episode
        if not self.dnd.encoding_off:
            if self.exp_settings['mem_store'] == 'embedding':
                # # Freeze all LSTM Layers before getting memory
                # layers_before = [self.i2h, self.h2h, self.a2c]
                # for layer in layers_before:
                #     for name, param in layer.named_parameters():
                #         param.requires_grad = False 

                # # print("Before-a2c s:\n", self.a2c.critic.weight)
                # # print("Before-emb s:\n", self.dnd.embedder.e2c.weight)

                # Saving Memory (hidden state passed into embedder, embedding is key and c_t is val)
                self.dnd.save_memory(h_t, c_t)

                # # print("After-a2c s:\n", self.a2c.critic.weight)
                # # print("After-emb s:\n", self.dnd.embedder.e2c.weight)

                # layers_after = [self.i2h, self.h2h, self.a2c]
                # # Unfreeze LSTM
                # for layer in layers_after:
                #     for name, param in layer.named_parameters():
                #         param.requires_grad = True 
            else:
                self.dnd.save_memory_non_embedder(q_t, barcode_string, c_t)

            if self.exp_settings['timing']:
                forward_save = time.perf_counter() - forward_get_mem - forward_prep - \
                    forward_preact - forward_gate - forward_start

        # policy
        pi_a_t, v_t, entropy = self.a2c.forward(h_t)
        # pick an action
        a_t, prob_a_t = self.pick_action(pi_a_t)

        timings = {}
        if self.exp_settings['timing']:
            forward_action = time.perf_counter() - forward_save - forward_get_mem - forward_prep - \
                forward_preact - forward_gate - forward_start
            timings = { "1a. Prep": forward_prep, "1b. PreAct": forward_preact, "1c. Gate": forward_gate,
                        "1d. Get_mem": forward_get_mem, "1e. Action": forward_action, "1f. Save_mem": forward_save}
        
        # fetch activity
        output = (a_t, predicted_barcode, mem_predicted_bc, prob_a_t, v_t, entropy, h_t, c_t)
        cache = (f_t, i_t, o_t, r_t, m_t, timings)

        return output, cache

    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.

        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)

        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)
        """
        m = torch.distributions.Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t

    def get_init_states(self, scale=.1):
        h_0 = torch.randn(1, self.dim_hidden_lstm, device = self.device) * scale
        c_0 = torch.randn(1, self.dim_hidden_lstm, device = self.device) * scale
        return h_0, c_0

    def flush_trial_buffer(self):
        self.dnd.trial_buffer = [()]

    def turn_off_encoding(self):
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()

    def get_all_mems(self):
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V

    def get_all_mems_embedder(self):
        mem_keys = self.dnd.keys
        predicted_mapping_to_keys = self.dnd.key_context_map
        return mem_keys, predicted_mapping_to_keys

    def difference_of_weights(self, prior_vals):
        layers = [self.i2h, self.h2h, self.a2c]
        for layer_after, layer_before in zip(layers, prior_vals):
            for (name, param_after), (name, param_before) in zip(layer_after.named_parameters(), layer_before.named_parameters()):
                try:
                    diff = torch.sub(param_after.grad, param_before.grad)
                    print(name, diff)
                except Exception:
                    continue