"""
Created on 16/09/2020

@author: Kyle
"""
import argparse
import copy
import time
import torch
import torch.utils.data
import numpy as np
from torch.autograd import Variable

import dataUtils
import imputeModel
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def initSysParameters():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--method', type=str, default='imputeAttention')
    parser.add_argument('--dataset_folder', type=str, default='dataset')
    parser.add_argument('--dataset_name', type=str, default='geolife')
    parser.add_argument('--imp_percent', type=float, default=0.2)
    parser.add_argument('--imp_points_num', type=int, default=1, help='number of points for imputing per time')
    parser.add_argument('--pred_len', type=int, default=1, help='1 is considered only in the test')
    parser.add_argument('--TF_emb_size', type=int, default=256)
    parser.add_argument('--rnn_hid_size', type=int, default=256)
    parser.add_argument('--TF_layers', type=int, default=2)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--rnn_type', type=str, default='GRU')
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learn_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--alpha', type=float, default=1, help='imputation train loss rate')
    parser.add_argument('--beta', type=float, default=1, help='prediction train loss rate')
    parser.add_argument('--beta_s', type=float, default=1, help='speed train loss rate')
    parser.add_argument('--clip', type=int, help='gradient clipping', default=10)
    parser.add_argument('--shuffle_data', type=bool, default=True)
    parser.add_argument('--add_speed_loss', type=bool, default=True)
    parser.add_argument('--loss_func', type=int, default=1, help='1.l2 loss; 2.euclidean; 3.haversine; 4.dtw')
    parser.add_argument('--step', type=int, default=5, help='step length to generate next sub-trajectory')
    parser.add_argument('--mask_dis', type=str, default='uniform', help='mask distribution - uniform, poi2, poi10')
    parser.add_argument('--traj_width', type=str, default='20_1')
    parser.add_argument('--use_portion', type=int, default=1)

    return parser.parse_args()


def getAveSpeed(obs_p, obs_inter, obs_p_ind, imp_p=None, imp_p_ind=None):
    obs_ave_speed = None
    obs_len = len(obs_p_ind)
    add_times = 0

    if imp_p is None:
        for i in range(0, obs_len - 1):
            for j in range(i + 1, obs_len):
                ind1 = obs_p_ind[j]
                ind2 = obs_p_ind[i]
                # obs_dis_dif = torch.norm(obs_p[:, ind1] - obs_p[:, ind2], dim=1)
                obs_dis_dif = dataUtils.haversine_tensor(obs_p[:, ind1], obs_p[:, ind2])
                obs_time_dif = torch.abs(obs_inter[:, ind1] - obs_inter[:, ind2])
                if obs_ave_speed is None:
                    obs_ave_speed = obs_dis_dif / obs_time_dif
                else:
                    obs_ave_speed += obs_dis_dif / obs_time_dif
                add_times += 1
    else:
        for i in range(len(imp_p_ind)):
            for j in obs_p_ind:
                # obs_dis_dif = torch.norm(imp_p[:, i] - obs_p[:, j], dim=1)
                obs_dis_dif = dataUtils.haversine_tensor(imp_p[:, i], obs_p[:, j])
                obs_time_dif = torch.abs(obs_inter[:, imp_p_ind[i]] - obs_inter[:, j])
                if obs_ave_speed is None:
                    obs_ave_speed = obs_dis_dif / obs_time_dif
                else:
                    obs_ave_speed += obs_dis_dif / obs_time_dif
                add_times += 1

    obs_ave_speed = obs_ave_speed / add_times
    return obs_ave_speed


def cal_loss(x, y, loss_type):
    """1.l2 loss; 2.euclidean; 3.haversine; 4.dtw"""

    if loss_type == 2:
        loss = torch.mean(torch.norm(x - y, dim=1))
    elif loss_type == 3:
        loss = torch.mean(dataUtils.haversine_tensor(torch.squeeze(x, 1), torch.squeeze(y, 1)))
        # loss = Variable(dis_loss.data, requires_grad=True)
    elif loss_type == 4:
        loss = 1
    else:
        loss = torch.mean((x - y).pow(2))
    return loss


def runModel(model_, d_loader, opt, b_size, dev, training=True):
    model_.train()
    total_imp_loss = 0
    total_pre_loss = 0
    imp_times = 0
    pre_times = 0

    rnn_h = model_.ini_rnn_hid(b_size, dev)

    for b_data, b_fra, b_inter in d_loader:
        data_obs, frames_obs, inter_obs, data_imp, frames_imp, data_pre, miss_p_ind, obs_p_ind = dataUtils.createMissData(
            b_data, b_fra, b_inter, args.imp_percent, args.pred_len)

        ini_imp_emb, imp_fra_emb = ini_imp_embed(data_obs.shape[0], data_imp.size(0), args.imp_points_num,
                                                 args.TF_emb_size, dev)
        obs_fra_emb = torch.zeros(data_obs.size(0), data_obs.size(1), args.TF_emb_size).to(dev)

        i = 0
        miss_p_len = len(miss_p_ind)
        data_obs_dev = data_obs.to(dev)
        inter_obs_dev = inter_obs.to(dev)
        obs_speed = getAveSpeed(data_obs_dev, inter_obs_dev, obs_p_ind)

        while i < miss_p_len:
            imp_p_end_i = i + args.imp_points_num
            if imp_p_end_i > miss_p_len:
                imp_p_end_i = miss_p_len

            imp_p_ind = miss_p_ind[i:imp_p_end_i]
            if len(imp_p_ind) < args.imp_points_num:
                ini_imp_emb, imp_fra_emb = ini_imp_embed(data_obs.shape[0], data_imp.size(0), len(imp_p_ind),
                                                         args.TF_emb_size, dev)

            if training:  # training mode
                opt.zero_grad()
                imp_out, pre_out, rnn_hid = model_(data_obs.to(dev), ini_imp_emb, None,
                                                   frames_obs.to(dev), frames_imp[:, i:imp_p_end_i].to(dev),
                                                   obs_fra_emb, imp_fra_emb, rnn_h)

                # data_obs.data[:, imp_p_ind] = imp_out.cpu().data

                batch_imp_loss = cal_loss(data_imp[:, i:imp_p_end_i].to(dev), imp_out, args.loss_func)
                batch_pre_loss = cal_loss(data_pre.to(dev), pre_out, args.loss_func)

                if args.add_speed_loss:
                    imp_speed = getAveSpeed(data_obs_dev, inter_obs_dev, obs_p_ind, imp_out, imp_p_ind)
                    b_speed_loss = torch.mean(torch.abs(imp_speed - obs_speed))
                    batch_loss = args.alpha * batch_imp_loss + args.beta * batch_pre_loss + b_speed_loss * args.beta_s
                else:
                    batch_loss = args.alpha * batch_imp_loss + args.beta * batch_pre_loss

                batch_loss.backward()
                nn.utils.clip_grad_norm(model_.parameters(), args.clip)
                opt.step()

                total_imp_loss += batch_imp_loss.item()
                total_pre_loss += batch_pre_loss.item()
                imp_times += 1
                pre_times += 1
            else:  # validating/testing mode
                if imp_p_end_i != miss_p_len:  # it is not the last imputation in a batch of data
                    imp_out = model_(data_obs.to(dev), ini_imp_emb, None,
                                     frames_obs.to(dev), frames_imp[:, i:imp_p_end_i].to(dev),
                                     obs_fra_emb, imp_fra_emb)
                    # data_obs.data[:, imp_p_ind] = imp_out.cpu().data
                else:
                    imp_out, pre_out, rnn_h = model_(data_obs.to(dev), ini_imp_emb, None,
                                                     frames_obs.to(dev), frames_imp[:, i:imp_p_end_i].to(dev),
                                                     obs_fra_emb, imp_fra_emb, rnn_h)

                    batch_pre_loss = cal_loss(data_pre.to(dev), pre_out, args.loss_func)
                    # batch_pre_loss = cal_loss(data_pre.to(dev), pre_out, 3)
                    total_pre_loss += batch_pre_loss.item()
                    pre_times += 1

                batch_imp_loss = cal_loss(data_imp[:, i:imp_p_end_i].to(dev), imp_out, args.loss_func)
                # batch_imp_loss = cal_loss(data_imp[:, i:imp_p_end_i].to(dev), imp_out, 3)
                total_imp_loss += batch_imp_loss.item()
                imp_times += 1

            i = imp_p_end_i

    return total_imp_loss / imp_times, total_pre_loss / pre_times


def ini_imp_embed(obs_num, imp_num, imp_p_num, emb_size, dev):
    ini_imp_emb = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(obs_num, imp_p_num, 1).to(dev)
    imp_fra_emb = torch.zeros(imp_num, imp_p_num, emb_size).to(dev)
    return ini_imp_emb, imp_fra_emb


if __name__ == '__main__':
    args = initSysParameters()
    arg_keys = []
    arg_vals = []

    for arg in vars(args):
        arg_keys.append(arg)
        arg_vals.append(getattr(args, arg))

    # create path for saving parameters of model and loss history
    model_path, log_path = dataUtils.makeOutputDir(args.dataset_name)
    timestamp = time.time()
    timestamp_str = str(timestamp).split(".")
    imp_model_save_file = model_path + 'state_dict_imp_' + str(args.method) + timestamp_str[0] + '.pth'
    pred_model_save_file = model_path + 'state_dict_pred_' + str(args.method) + timestamp_str[0] + '.pth'
    log_file = log_path + str(args.method) + '.csv'

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_data, train_frames, train_intervals, tra_max_frame = dataUtils.loadDataset(args.dataset_folder,
                                                                                     args.dataset_name, True,
                                                                                     args.use_portion,
                                                                                     args.step, args.traj_width)
    test_data, test_frames, test_intervals, tes_max_frame = dataUtils.loadDataset(args.dataset_folder,
                                                                                  args.dataset_name, False,
                                                                                  args.use_portion,
                                                                                  args.step, args.traj_width)

    model = imputeModel.ImputeAtten(2, 2, 2, TF_layers=args.TF_layers, rnn_layers=args.rnn_layers,
                                    rnn_type=args.rnn_type, emb_dim=args.TF_emb_size,
                                    rnn_hid_dim=args.rnn_hid_size, heads=args.heads,
                                    dropout=args.dropout,
                                    max_pos=max(tra_max_frame, tes_max_frame)).to(device)
    # torch.set_printoptions(precision=12)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_frames), torch.from_numpy(train_intervals)),
        shuffle=args.shuffle_data, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_frames), torch.from_numpy(test_intervals)),
        shuffle=args.shuffle_data, batch_size=args.batch_size, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    best_test_imp_loss = 1000
    best_test_pre_loss = 1000

    for epoch in range(1, args.epochs + 1):
        print('Train epoch ' + str(epoch))

        train_imp_loss, train_pre_loss = runModel(model, train_loader, optimizer, args.batch_size, device)
        print('epoch' + str(epoch) + ' train_imp_loss :' + str(train_imp_loss))

        test_imp_loss, test_pre_loss = runModel(model, test_loader, optimizer, args.batch_size, device, False)
        print('epoch' + str(epoch) + ' test_imp_loss :' + str(test_imp_loss))

        # best result on test set
        if best_test_imp_loss == 0 or test_imp_loss < best_test_imp_loss:
            best_test_imp_loss = test_imp_loss
            # torch.save(model.state_dict(), imp_model_save_file)
            print('Best imp model at epoch ' + str(epoch) + ':' + str(best_test_imp_loss))

        if best_test_pre_loss == 0 or test_pre_loss < best_test_pre_loss:
            best_test_pre_loss = test_pre_loss
            print('Best pred model at epoch ' + str(epoch) + ':' + str(best_test_pre_loss))

    dataUtils.saveLogs(log_file, False, best_test_imp_loss, best_test_pre_loss, arg_keys, arg_vals)
