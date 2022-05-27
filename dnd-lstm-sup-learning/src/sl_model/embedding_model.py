from numpy import int8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):

    def __init__(self, exp_settings, bias=True):
        super(Embedder, self).__init__()
        self.bias = bias
        self.exp_settings = exp_settings
        self.input_dim = exp_settings['dim_hidden_lstm']
        embedding_size = exp_settings['embedding_size']
        self.barcode_size = exp_settings['barcode_size']

        # Basic Layers
        self.h2e = nn.Linear(self.input_dim, embedding_size, bias=bias)
        self.e2c = nn.Linear(embedding_size, self.barcode_size, bias=bias)

        # init
        self.reset_parameter()

    # Model should return an embedding and a context
    def forward(self, h):
        x = F.relu(self.h2e(h))
        embedding = x
        predicted_context = torch.sigmoid(self.e2c(x))
        return embedding, predicted_context

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)
