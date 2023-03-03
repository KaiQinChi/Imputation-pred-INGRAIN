import math

import torch.nn as nn


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, emb_dim):
        super(LinearEmbedding, self).__init__()
        self.linear_emb = nn.Linear(inp_size, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.linear_emb(x) * math.sqrt(self.emb_dim)


class IDEmbedding(nn.Module):
    def __init__(self, dict_size, emb_dim):
        super(IDEmbedding, self).__init__()
        self.embedding = nn.Embedding(dict_size, emb_dim)  # lookup table
        self.emb_dim = emb_dim

    def forward(self, x, ids):
        x_id = self.embedding(ids) * math.sqrt(self.emb_dim)
        return x + x_id
