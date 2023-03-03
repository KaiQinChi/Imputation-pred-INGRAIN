# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, obs_embed, imp_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.obs_embed = obs_embed
        self.imp_embed = imp_embed
        self.generator = generator

    def forward(self, obs, imp, obs_mask, imp_mask):
        """
        Take in and process masked obs and target sequences.
        """
        obs_emb = self.encode(obs, obs_mask)
        imp_emb = self.decode(obs_emb, imp, obs_mask, imp_mask)
        return obs_emb, imp_emb

        # return self.decode(self.encode(obs, obs_mask), imp, obs_mask, imp_mask)

    def encode(self, obs, obs_mask):
        return self.encoder(self.obs_embed(obs), obs_mask)

    def decode(self, memory, imp, obs_mask, imp_mask):
        return self.decoder(self.imp_embed(imp), memory, obs_mask, imp_mask)
