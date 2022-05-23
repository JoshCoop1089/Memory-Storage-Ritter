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
        self.fc1 = nn.Linear(self.input_dim, embedding_size)
        self.fc2 = nn.Linear(embedding_size, self.barcode_size)

        # init
        self.reset_parameter()

    # Model should return an embedding and a context
    def forward(self, h):
        x = F.relu(self.fc1(h))
        embedding = x
        predicted_context = F.relu(self.fc2(x))
        return embedding, predicted_context

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)
