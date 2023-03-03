# -*- coding: utf-8 -*-
# date: 2018-11-30 15:41
import torch.nn as nn

from .functional import clones
from .sublayer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, obs-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, obs_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.obs_attn = obs_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, obs_mask, imp_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, imp_mask))
        x = self.sublayer[1](x, lambda x: self.obs_attn(x, m, m, obs_mask))
        return self.sublayer[2](x, self.feed_forward)
