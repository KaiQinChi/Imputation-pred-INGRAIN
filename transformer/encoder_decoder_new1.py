# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, obs_embed, imp_embed, fra_emb, gru_transfer):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.obs_embed = obs_embed
        self.imp_embed = imp_embed
        self.fra_emb = fra_emb
        self.gru_transfer = gru_transfer

    def forward(self, obs, ini_run, imp, obs_fra, imp_fra, obs_fra_emb, imp_fra_emb, obs_mask, imp_mask):
        """
        Take in and process masked obs and target sequences.
        """
        obs_emb = self.encode(obs, ini_run, obs_fra, obs_fra_emb, obs_mask)
        # obs_emb, _ = self.gru_transfer(obs_emb)
        imp_emb = self.decode(obs_emb, imp, imp_fra, imp_fra_emb, obs_mask, imp_mask)
        return obs_emb, imp_emb

        # return self.decode(self.encode(obs, obs_mask), imp, obs_mask, imp_mask)

    def encode(self, obs, ini_run, obs_fra, obs_fra_emb, obs_mask):
        if ini_run:
            return self.encoder(self.fra_emb(self.obs_embed(obs), obs_fra, obs_fra_emb), obs_mask)
        else:
            return self.encoder(self.fra_emb(obs, obs_fra, obs_fra_emb), obs_mask)

    def decode(self, obs_emb, imp, imp_fra, imp_fra_emb, obs_mask, imp_mask):
        return self.decoder(self.fra_emb(self.imp_embed(imp), imp_fra, imp_fra_emb), obs_emb, obs_mask, imp_mask)
