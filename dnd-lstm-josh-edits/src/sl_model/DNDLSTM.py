"""
A DND-based LSTM based on ...
Ritter, et al. (2018).
Been There, Done That: Meta-Learning with Episodic Recall.
Proceedings of the International Conference on Machine Learning (ICML).
"""
import torch
import torch.nn as nn
from sl_model.DND import DND
from sl_model.A2C import A2C_linear

# constants
N_GATES = 4

class DNDLSTM(nn.Module):

    def __init__(self, 
                    dim_input_lstm, dim_hidden_lstm, dim_output_lstm,
                    dict_len, exp_settings, bias=True):
        super(DNDLSTM, self).__init__()
        self.input_dim = dim_input_lstm
        self.dim_hidden_lstm = dim_hidden_lstm
        self.bias = bias
        self.exp_settings = exp_settings
        # input-hidden weights
        self.i2h = nn.Linear(dim_input_lstm, (N_GATES+1)
                             * dim_hidden_lstm, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(dim_hidden_lstm, (N_GATES+1) * dim_hidden_lstm, bias=bias)
        # dnd
        self.dnd = DND(dict_len, dim_hidden_lstm, exp_settings)
        #policy
        self.a2c = A2C_linear(dim_hidden_lstm, dim_output_lstm)
        # init
        self.reset_parameter()

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)

    def forward(self, observation, barcode, h, c, enable_embedder_layers):
        # unpack activity
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)

        # Form the inputs nicely
        observation = observation.view(1, self.exp_settings['num_arms'])
        context = barcode.view(1, self.exp_settings['barcode_size'])
        print('Obs: ', observation)
        print('CTX: ', context)

        if self.exp_settings['agent_input'] == 'obs/context':
            x_t = torch.cat((observation, context), dim = 1)
        else:  # self.exp_settings['agent_input'] == 'obs'
            x_t = observation

        print(x_t)

        # transform the input info
        Wx = self.i2h(x_t)
        Wh = self.h2h(h)
        preact = Wx + Wh
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

        # Embedding model should be unfrozen if this is the first query of a trial

        # Query Memory (hidden state passed into embedder, context used for embedder loss function)
        mem, predicted_barcode = self.dnd.get_memory(h, context, enable_embedder_layers)
        m_t = mem.tanh()
        # gate the memory; in general, can be any transformation of it
        c_t = c_t + torch.mul(r_t, m_t)
        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, c_t.tanh())

        # Embedding model should be frozen after save_memory if on last moment of trial
        # self.turn_on_encoding() should have been activated in main function
        # Saving Memory (hidden state passed into embedder, embedding is key and c_t is val)
        self.dnd.save_memory(h_t, c_t)

        # policy
        pi_a_t, v_t = self.a2c.forward(h_t)
        # pick an action
        a_t, prob_a_t = self.pick_action(pi_a_t)
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        # fetch activity
        output = [a_t, predicted_barcode, prob_a_t, v_t, h_t, c_t]
        cache = [f_t, i_t, o_t, r_t, m_t]
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
        h_0 = torch.randn(1, 1, self.dim_hidden_lstm) * scale
        c_0 = torch.randn(1, 1, self.dim_hidden_lstm) * scale
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

    def get_all_mems_josh(self):
        # Memory now storing lists of memories in every spot, so we only want the most updated version of each list
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i][0] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V
