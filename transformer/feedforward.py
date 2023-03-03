# -*- coding: utf-8 -*-
# date: 2018-11-30 16:49
import torch.nn as nn
from torch.nn.functional import relu


class Feedforward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, emb_dim, ff_dim, dropout=0.1):
        super(Feedforward, self).__init__()
        self.w_1 = nn.Linear(emb_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(relu(self.w_1(x))))
