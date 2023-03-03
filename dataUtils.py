"""
Created on 16/09/2020

@author: Kyle
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import pickle
import csv
import math

datasets = ['billiard', 'cuebiq', 'cuebiq-us', 'cuebiq-au', 'zara', 'foursquare', 'geolife', 'basketball', 'gowalla']
# proj_dir = './impute/'  # use it on NCI
proj_dir = ''


def makeOutputDir(dataset_name):
    model_dir = proj_dir + 'models'
    log_dir = proj_dir + 'logs'
    model_path = model_dir + '/' + dataset_name
    log_path = log_dir + '/' + dataset_name

    makeDir(model_dir)
    makeDir(log_dir)
    makeDir(model_path)
    makeDir(log_path)

    model_path = model_path + '/'
    log_path = log_path + '/'
    return model_path, log_path


def makeDir(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)


def loadDataset(dataset_folder, dataset_name, train_set=True, usage_percent=1, step=0, traj_len=None, POI=False):
    data = None
    frames = None
    intervals = None
    POI_num = 0

    if dataset_name not in datasets:
        print('no such dataset!')
        exit()
    elif train_set:
        train_file = dataset_name + '_train_step' + str(step) + '_len' + str(traj_len) + '.p'

        train_file_path = os.path.join(proj_dir, dataset_folder, dataset_name, train_file)
        if dataset_name == datasets[0]:
            data = pickle.load(open(train_file_path, 'rb'), encoding='latin1')
        else:
            data = pickle.load(open(train_file_path, 'rb'))

        if isinstance(data, dict):
            frames = data['frames_train']
            intervals = data['interval_train']

            if POI:
                data = data['POI_trajs_train']
                POI_num = data['POI_num']
            else:
                if "coordinate_trajs_train" in data:
                    data = data['coordinate_trajs_train']
                else:
                    data = data['trajs_train']

    else:
        eval_file = dataset_name + '_test_step' + str(step) + '_len' + str(traj_len) + '.p'

        eval_file_path = os.path.join(proj_dir, dataset_folder, dataset_name, eval_file)
        if dataset_name == datasets[0]:
            data = pickle.load(open(eval_file_path, 'rb'), encoding='latin1')
        else:
            data = pickle.load(open(eval_file_path, 'rb'))

        if isinstance(data, dict):
            frames = data['frames_test']
            intervals = data['interval_test']

            if POI:
                data = data['POI_trajs_test']
                # POI_num = data['POI_num']
            else:
                if "coordinate_trajs_test" in data:
                    data = data['coordinate_trajs_test']
                else:
                    data = data['trajs_test']

    # -----------------------For training model------------------------
    d = data.astype(np.float32)
    input_num = int(d.shape[0] * usage_percent)
    frames = frames[:input_num]
    intervals = intervals[:input_num]
    max_frame = np.amax(frames)

    if POI:
        return d[:input_num], frames, intervals, POI_num
    else:
        return d[:input_num], frames, intervals, max_frame

    # # -----------------------For testing some imputation points on map------------------------
    # d = data.astype(np.float32)
    # input_num = int(d.shape[0] * usage_percent)
    # max_frame = np.amax(frames)
    # return d[-70:], frames[-70:], max_frame


def split_digits(data, split_size=8):
    s0 = data.shape[0]
    s1 = data.shape[1]
    s2 = split_size  # reserve n digits for longitude and latitude
    new_data = np.zeros((s0, s1, s2))
    s2_half = int(s2 / 2)
    digit_shift = 100

    ints = np.trunc(data)
    floats = data - ints
    new_data[:, :, 0] = ints[:, :, 0]  # start save digits for long
    new_data[:, :, s2_half] = ints[:, :, 1]  # start save digits for lat

    i = 1
    while i < s2_half:
        floats = floats * digit_shift
        int_parts = np.trunc(floats)
        floats = floats - int_parts
        new_data[:, :, i] = int_parts[:, :, 0]  # add digits in long
        new_data[:, :, i + s2_half] = int_parts[:, :, 1]  # add digits in lat
        i += 1
    return new_data.astype(np.float32)


def recover_digits(data, input_size=2):
    s0 = data.shape[0]
    s1 = data.shape[1]
    s2 = data.shape[2]
    s2_half = int(s2 / 2)
    digit_shift = 100
    new_data = torch.zeros((s0, s1, input_size)).to(data.device)

    new_data[:, :, 0] = data[:, :, 0]
    new_data[:, :, 1] = data[:, :, s2_half]

    i = 1
    while i < s2_half:
        new_data[:, :, 0] += data[:, :, i] / (digit_shift ** i)
        new_data[:, :, 1] += data[:, :, i + s2_half] / (digit_shift ** i)
        i += 1
    return new_data


def divideDataset(data, imp_percent, pre_len):
    traj_len = data.shape[1] - pre_len
    data_pre = data[:, -pre_len:]
    imp_len = int(traj_len * imp_percent)  # number of points from obs to impute

    # generate mask with a distribution
    # miss_p_ind = torch.from_numpy(random_choice(np.arange(1, traj_len), imp_len)).long()
    miss_p_ind = torch.from_numpy(random_choice(np.arange(1, traj_len), imp_len, lam=10, distribution='poisson')).long()

    data_imp = data[:, miss_p_ind]
    data[:, miss_p_ind] = 0.0
    return data[:, :-pre_len], data_imp, data_pre.squeeze(1), miss_p_ind


def divideDatasetPred(data, imp_percent, pre_len):
    traj_len = data.shape[1] - pre_len
    data_pre = data[:, -pre_len:]
    imp_len = int(traj_len * imp_percent)  # number of points from obs to impute
    missing_points_ind = torch.from_numpy(np.random.choice(np.arange(1, traj_len), imp_len, replace=False)).long()
    data_imp = data[:, missing_points_ind]
    data[:, missing_points_ind] = 0.0
    return data[:, :-pre_len], data_imp, data_pre, missing_points_ind


def divideNormalisedDataset(data, data_ori, imp_percent, pre_len):
    traj_len = data.shape[1] - 1
    data_pre = data[:, -pre_len:]
    data_ori_pre = data_ori[:, -pre_len:]

    imp_len = int(traj_len * imp_percent)  # number of points from obs to impute
    missing_points_ind = torch.from_numpy(np.random.choice(np.arange(1, traj_len), imp_len, replace=False)).long()

    data_imp = data[:, missing_points_ind]
    data[:, missing_points_ind] = 0.0
    data_ori_imp = data_ori[:, missing_points_ind]
    data_ori[:, missing_points_ind] = 0.0
    return data[:, :-pre_len], data_imp, data_pre.squeeze(1), data_ori_imp, data_ori_pre.squeeze(1), missing_points_ind


def saveLogs(file_name, title, test_imp_loss, test_pre_loss, arg_keys, arg_vals, draw_dia=False):
    titles = ['test_imp_loss', 'test_pre_loss'] + arg_keys
    params = [test_imp_loss, test_pre_loss] + arg_vals
    if title:
        results = [titles, params]
    else:
        results = [params]
    with open(file_name, "a") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)

    # if draw_dia:
    #     drawPredictResult(log_path, file_name, learn_method)


def pk_load(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def pk_dump(path, data):
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()


def random_choice(array, out_len, lam=2, replace=False, distribution='uniform'):
    """
    Random choice of values in an array.
    :param replace: the values are repeated or not
    :param distribution: uniform (default), poisson
    """
    if distribution == 'poisson':
        length = array.shape[0]
        # lam == 10 is similar to normal distribution
        mask = np.random.poisson(lam, length)
        mask = mask + 0.0001
        p = mask / mask.sum(axis=0)
        return np.random.choice(array, out_len, p=p, replace=replace)
    else:
        return np.random.choice(array, out_len, replace=replace)


def createMissData(data, frames, intervals, imp_percent, pre_len):
    traj_len = data.shape[1] - 1
    data_pre = data[:, -pre_len:]
    imp_len = int(traj_len * imp_percent)  # number of points from obs to impute

    # generate mask with a distribution
    # miss_p_ind = torch.from_numpy(random_choice(np.arange(1, traj_len), imp_len)).long()
    miss_p_ind = torch.from_numpy(random_choice(np.arange(1, traj_len), imp_len, lam=2, distribution='poisson')).long()

    data_imp = data[:, miss_p_ind]
    data[:, miss_p_ind] = 0.0
    frames_imp = frames[:, miss_p_ind]
    obs_p_ind = [i for i in np.arange(0, traj_len) if i not in miss_p_ind]

    return data[:, :-pre_len], frames[:, :-pre_len], intervals[:, :-pre_len], data_imp, frames_imp, data_pre.squeeze(
        1), miss_p_ind, torch.as_tensor(obs_p_ind)


def divideShuffleDataset(data, min_imp_percent, max_imp_percent, pre_len, comp_prob):
    traj_num = data.shape[0]
    traj_len = data.shape[1] - 1
    data_pre = data[:, -pre_len:]
    data_imp = []
    mis_points_list = []

    for i in range(traj_num):
        if np.random.random_sample() > comp_prob:
            imp_len = int(traj_len * min_imp_percent)  # number of points from obs to impute
            missing_points_ind = torch.from_numpy(
                np.random.choice(np.arange(1, traj_len), imp_len, replace=False)).long()
            data_imp.append(data[i, missing_points_ind].unsqueeze(0))
            data[i, missing_points_ind] = 0.0
            mis_points_list.append(missing_points_ind)
        else:
            data_imp.append(None)
            mis_points_list.append(None)

    return data[:, :-pre_len], torch.cat(data_imp, dim=0)[:, :, :], data_pre.squeeze(1), mis_points_list


def drawPredictResult(dir_, f_name, method_):
    f_name_pref = os.path.splitext(f_name)[0]
    df = pd.read_csv(dir_ + f_name)
    cur_epoch = df['cur_epoch']
    train_pre_loss = df['train_pre_loss']
    test_pre_loss = df['test_pre_loss']
    train_imp_loss = df['train_imp_loss']
    test_imp_loss = df['test_imp_loss']
    imp_percent = df['imp_percent'].iat[0]

    if method_ == 'GRU' or method_ == 'LSTM':
        plt.figure(figsize=(14, 10))
        # plt.plot(cur_epoch, train_pre_loss, '-g', color='blue', label='train_pre_loss')
        plt.plot(cur_epoch, test_pre_loss, '-g', color='black', label='test_pre_loss')
        plt.ylabel('Predict Loss', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)
        plt.legend()
        # plt.axis.set_xticks(np.arange(0, 1, 0.1))
        plt.grid(True)
        plt.title('Train/Test Loss - ' + method_ + str(imp_percent), fontsize=20)
        plt.savefig(dir_ + method_ + str(imp_percent) + '_Predict_' + f_name_pref + '.png', bbox_inches='tight')
        plt.show()
    elif method_ == 'ImputeTFRNN':
        plt.figure(figsize=(14, 10))
        # plt.plot(cur_epoch, train_pre_loss, '-g', color='blue', label='train_pre_loss')
        plt.plot(cur_epoch, test_pre_loss, '-g', color='black', label='test_pre_loss')
        plt.ylabel('Predict Loss', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)
        # plt.set_yticks(np.arange(0, 0.9, 0.1))
        plt.legend()
        plt.grid(True)
        plt.title('Train/Test Loss - ' + method_ + str(imp_percent), fontsize=20)
        plt.savefig(dir_ + method_ + str(imp_percent) + '_Predict_' + f_name_pref + '.png', bbox_inches='tight')
        plt.clf()

        # plt.plot(cur_epoch, train_imp_loss, '-g', color='blue', label='train_imp_loss')
        plt.plot(cur_epoch, test_imp_loss, '-g', color='black', label='test_imp_loss')
        plt.ylabel('Impute Loss', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)
        # plt.set_yticks(np.arange(0, 0.9, 0.1))
        plt.legend()
        plt.grid(True)
        plt.title('Train/Test Loss - ' + method_ + str(imp_percent), fontsize=20)
        plt.savefig(dir_ + method_ + str(imp_percent) + '_Impute_' + f_name_pref + '.png', bbox_inches='tight')
        # plt.show()


def haversine(coord1, coord2, metric=1):
    """
    Calculate the haversine distance between two lon/lat pairs.
    Output distance available in 1.meters, 2.kilometers,  3.miles, and 4.feet.
    Example usage: Haversine([lon1,lat1],[lon2,lat2]).feet
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers
    miles = meters * 0.000621371  # output distance in miles
    feet = miles * 5280  # output distance in feet

    if metric == 2:
        res = km
    elif metric == 3:
        res = miles
    elif metric == 4:
        res = feet
    else:
        res = meters

    return res


def haversine_tensor(coord_list1, coord_list2, metric=1):
    """
    Calculate the haversine distance between each lon/lat pairs.
    Output distance available in 1.meters, 2.kilometers,  3.miles, and 4.feet.
    """
    size = coord_list1.shape[0]
    dis = torch.zeros(size).to(coord_list1.device)
    for i in range(size):
        dis[i] = haversine(coord_list1[i], coord_list2[i], metric)

    return dis
