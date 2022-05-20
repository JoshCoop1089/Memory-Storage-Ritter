import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):

    def __init__(self,
                 input_dim, num_contexts,
                 exp_settings,
                 bias=True
                 ):
        super(Embedder, self).__init__()
        self.input_dim = input_dim
        self.bias = bias
        self.exp_settings = exp_settings
        hidden_layer_size = exp_settings['hidden_layer_size']

        # Basic Layers
        self.fc1 = nn.Linear(input_dim, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, num_contexts)

        # init
        self.reset_parameter()

    # Model should return an embedding and a context ID
    # When used in get_memory, only the context ID is needed
    # When used in save_memory, the embedding overwrites the current memory in mem[context_id]

    def forward(self, h):
        x = F.relu(self.fc1(h))
        embedding = x
        predicted_context = F.softmax(self.fc2(x), dim = 1)
        return embedding, predicted_context

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)
