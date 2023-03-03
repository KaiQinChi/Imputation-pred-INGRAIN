# -*- coding: utf-8 -*-
# date: 2018-11-30 17:00
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncodingFrame(nn.Module):
    """
    Implement the PE function for time frame awareness in input encoding or decoding.
    """

    def __init__(self, d_model, dropout, max_len, div_dim=10000.0):
        super(PositionalEncodingFrame, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(div_dim) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, time_fra, frame_emb):
        for i in range(x.size(0)):
            fra = time_fra[i].data.cpu().numpy()
            frame_emb[i, :] = self.pe[:, fra]

        x = x + Variable(frame_emb, requires_grad=False)
        return self.dropout(x)
