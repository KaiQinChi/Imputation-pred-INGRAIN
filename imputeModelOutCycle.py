import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformer.multihead_attention import MultiHeadAttention
from transformer.pointerwise_feedforward import PointerwiseFeedforward

from transformer.encoder_decoder_new1 import EncoderDecoder
from transformer.positional_encoding_frame import PositionalEncodingFrame

from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
import numpy as np
import scipy.io
import os
import copy
import math


class ImputeAtten(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, out_size, TF_layers=6, rnn_type="GRU", rnn_layers=1,
                 rnn_hid_dim=256, emb_dim=256, ff_dim=512, heads=8, dropout=0.1, max_pos=9000):
        super(ImputeAtten, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadAttention(heads, emb_dim)
        ff = PointerwiseFeedforward(emb_dim, ff_dim, dropout)
        frame_aware = PositionalEncodingFrame(emb_dim, dropout, max_len=max_pos)
        # position = PositionalEncoding(TF_emb_dim, dropout)
        gru_transfer = nn.GRU(emb_dim, rnn_hid_dim, rnn_layers, batch_first=True, dropout=dropout)
        self.Att_module = EncoderDecoder(Encoder(EncoderLayer(emb_dim, c(attn), c(ff), dropout), TF_layers),
                                         Decoder(DecoderLayer(emb_dim, c(attn), c(attn), c(ff), dropout), TF_layers),
                                         LinearEmbedding(enc_inp_size, emb_dim),
                                         LinearEmbedding(dec_inp_size, emb_dim),
                                         c(frame_aware), gru_transfer)
        self.impute_linear = nn.Linear(emb_dim, out_size)

        # prediction part
        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_layers = rnn_layers
        self.gru_pred = nn.GRU(emb_dim, rnn_hid_dim, rnn_layers, batch_first=True, dropout=dropout)
        self.pred_linear = nn.Linear(rnn_hid_dim, out_size)
        self.relu = nn.ReLU()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # This was important from their code. Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, ini_imp, imp_points_ind, obs_frame, imp_frame, obs_frame_emb, imp_frame_emb, rnn_hid=None,
                obs_mask=None, imp_mask=None):
        obs_emb, imp_emb = self.Att_module(obs, True, ini_imp, obs_frame, imp_frame, obs_frame_emb, imp_frame_emb,
                                           obs_mask, imp_mask)
        imp_out = self.impute_linear(imp_emb)

        if imp_points_ind is not None:
            obs_emb.data[:, imp_points_ind] += imp_emb.data

        if rnn_hid is not None:
            # pre_out, rnn_hid = self.gru_pred(obs_emb, rnn_hid)
            pre_out = self.pred_linear(self.relu(obs_emb[:, -1]))
            return imp_out, pre_out, rnn_hid
        else:
            return imp_out

    def ini_rnn_hid(self, batch_size, device):
        h = Variable(torch.zeros(self.rnn_layers, batch_size, self.rnn_hid_dim))

        # mu = 0
        # sd = 1 / self.rnn_hid_dim
        # h = torch.randn(self.rnn_layers, batch_size, self.rnn_hid_dim, requires_grad=False) * sd + mu
        return h.to(device)

    def loss(self, output, y, loss_type=1):
        if loss_type == 1:
            POI_num = output.shape[2]
            out = output.view(-1, POI_num)
            lose = self.cross_entropy_loss(out, y.view(-1))
            return lose
        else:
            lose = torch.mean(y - output).pow(2)
        return lose

    # def precision(self, output, y):
    #     iter_cnt = 0
    #     recall1 = 0
    #     recall5 = 0
    #     recall10 = 0
    #     average_precision = 0.
    #     for j in range(self.setting.batch_size):
    #         # o contains a per user list of votes for all locations for each sequence entry
    #         o = out[j]
    #
    #         # partition elements
    #         o_n = o.cpu().detach().numpy()
    #         ind = np.argpartition(o_n, -10, axis=1)[:, -10:]  # top 10 elements
    #
    #         y_j = y[:, j]
    #
    #         for k in range(len(y_j)):
    #             # resort indices for k:
    #             ind_k = ind[k]
    #             r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]  # sort top 10 elements descending
    #
    #             r = torch.tensor(r)
    #             t = y_j[k]
    #
    #             # compute MAP:
    #             r_kj = o_n[k, :]
    #             t_val = r_kj[t]
    #             upper = np.where(r_kj > t_val)[0]
    #             precision = 1. / (1 + len(upper))
    #
    #             # store
    #             u_recall1[active_users[j]] += t in r[:1]
    #             u_recall5[active_users[j]] += t in r[:5]
    #             u_recall10[active_users[j]] += t in r[:10]
    #             u_average_precision[active_users[j]] += precision


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, emb_dim):
        super(LinearEmbedding, self).__init__()
        self.linear_enc = nn.Linear(inp_size, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.linear_enc(x) * math.sqrt(self.emb_dim)
