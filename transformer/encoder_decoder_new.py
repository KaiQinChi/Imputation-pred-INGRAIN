# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, obs_embedding, imp_embedding, GRU_encoder, fra_embedding):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.obs_embedding = obs_embedding
        self.imp_embedding = imp_embedding
        self.GRU_encoder = GRU_encoder
        self.fra_embedding = fra_embedding

    def forward(self, obs, init_imp, obs_fra, imp_fra, obs_fra_emb, imp_fra_emb, rnn_imp_hid, obs_mask, imp_mask):
        """
        Take in and process masked obs and target sequences.
        """
        obs_emb, imp_hid = self.encode(obs, obs_fra, obs_fra_emb, rnn_imp_hid, obs_mask)
        imp_emb = self.decode(obs_emb, init_imp, imp_fra, imp_fra_emb, obs_mask, imp_mask)
        return obs_emb, imp_emb, imp_hid

        # return self.decode(self.encode(obs, obs_mask), imp, obs_mask, imp_mask)

    def encode(self, obs, obs_fra, obs_fra_emb, rnn_imp_hid, obs_mask):
        imp_hid = None
        if rnn_imp_hid is not None:
            obs_emb, imp_hid = self.GRU_encoder(self.obs_embedding(obs), rnn_imp_hid)
        else:
            obs_emb = self.obs_embedding(obs)
        return self.encoder(self.fra_embedding(obs_emb, obs_fra, obs_fra_emb), obs_mask), imp_hid

    def decode(self, memory, imp, imp_fra, imp_fra_emb, obs_mask, imp_mask):
        return self.decoder(self.fra_embedding(self.imp_embedding(imp), imp_fra, imp_fra_emb), memory, obs_mask,
                            imp_mask)
