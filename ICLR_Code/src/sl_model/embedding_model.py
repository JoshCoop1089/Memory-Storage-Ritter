from numpy import int8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):

    def __init__(self, exp_settings, device, bias=True):
        super(Embedder, self).__init__()
        self.bias = bias
        self.exp_settings = exp_settings
        self.input_dim = exp_settings['dim_hidden_lstm']
        embedding_size = exp_settings['embedding_size']
        self.num_barcodes = exp_settings['num_barcodes']
        self.num_arms = exp_settings['num_arms']

        # Basic Layers
        self.h2m = nn.Linear(self.input_dim, 2*embedding_size, bias=bias, device = device)
        self.e2c = nn.Linear(2*embedding_size, self.num_barcodes, bias=bias, device = device)

        # init
        self.reset_parameter()

    # Model should return an embedding and a context
    def forward(self, h):
        x = self.h2m(h)
        embedding = x
        predicted_context = self.e2c(F.leaky_relu(x))
        return embedding, predicted_context

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            # print("emb: ", name)
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)