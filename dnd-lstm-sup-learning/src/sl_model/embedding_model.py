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
        self.mdrope = nn.Dropout(0.5)
        self.m2e = nn.Linear(2*embedding_size, 1*embedding_size, bias=bias, device = device)
        self.e2c = nn.Linear(1*embedding_size, self.num_barcodes, bias=bias, device = device)
        # self.e2c = nn.Linear(2*embedding_size, self.num_arms, bias=bias, device = device)

        # Use an LSTM??  Future model choice considerations
        self.lstm = nn.LSTM(self.input_dim, self.num_barcodes, bias=bias, device = device)

        # Blow up and shrink down
        # self.h2m = nn.Linear(self.input_dim, 2*embedding_size, bias=bias, device = device)
        self.mdrope = nn.Dropout(0.5)
        self.m2e = nn.Linear(2*embedding_size, 1*embedding_size, bias=bias, device = device)
        # self.edropc = nn.Dropout(0.5)
        # self.e2c = nn.Linear(1*embedding_size, self.num_barcodes, bias=bias, device = device)

        # Shrinking from LSTM Attempt
        # self.h2m = nn.Linear(self.input_dim, embedding_size//2, bias=bias, device = device)
        # self.mdrope = nn.Dropout(0.5)
        # self.m2e = nn.Linear(embedding_size//2, embedding_size//4, bias=bias, device = device)
        # # self.edropc = nn.Dropout(0.5)
        # self.e2c = nn.Linear(embedding_size//4, self.num_barcodes, bias=bias, device = device)

        # init
        self.reset_parameter()

    # Model should return an embedding and a context
    def forward(self, h):
        x = F.leaky_relu(self.h2m(h))
        x = self.mdrope(x)
        x = self.m2e(x)
        embedding = x
        # x1 = self.edropc(x)
        predicted_context = self.e2c(F.leaky_relu(x))
        return embedding, predicted_context

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            # print("emb: ", name)
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)
