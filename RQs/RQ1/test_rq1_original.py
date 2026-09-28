"""Run RQ1 on the original systems (data1-data8)."""

import time
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from skmultiflow import drift_detection
from skmultiflow.drift_detection import ADWIN
import datetime
import os
import logging
import sys
# sys.path.append("/mnt/sdaDisk/xzz/dhda")
from DaLModels import DaL_Regressor
from base_model.basemodels import build_incremental_model

seed = 43
work_dir = 'D:\\CodePycharm\\online_dal-main\\new_dhda\\result\\rq1-time'


def create_required_dirs(parent_dir):
    """
    检查并创建指定目录下所需的子目录

    参数:
        parent_dir (str): 父目录路径
    """
    # 需要检查和创建的子目录列表
    required_dirs = ['csv', 'log', 'figures', 'sort', 'stats']

    # 确保父目录存在
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"已创建父目录: {parent_dir}")

    # 检查并创建每个子目录
    for dir_name in required_dirs:
        dir_path = os.path.join(parent_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"已创建子目录: {dir_path}")
        else:
            print(f"子目录已存在: {dir_path}")


create_required_dirs(work_dir)
env_list1 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_4.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_5.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data1\\sac_9.csv"]

env_list2 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_0.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_3.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_4.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_5.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_9.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_10.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_11.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_12.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_13.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_14.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_15.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_16.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_17.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_18.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_19.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data2\\x264_20.csv", ]

env_list3 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj1_feature6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj1_feature7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj1_feature8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj2_feature9.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj2_feature6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj1_feature7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj2_feature8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data3\\storm-obj2_feature9.csv"]

env_list4 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_0.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_3.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_4.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_5.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_6.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_7.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_8.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data4\\spear_9.csv"]

env_list5 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_10.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_11.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_16.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_17.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_18.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_19.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_44.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_45.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_52.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_59.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_73.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_79.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_94.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_96.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data5\\sqlite_97.csv"]

env_list6 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data6\\nginx1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data6\\nginx2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data6\\nginx3.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data6\\nginx4.csv"]

env_list7 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data7\\exastencils1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data7\\exastencils2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data7\\exastencils3.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data7\\exastencils4.csv"]

env_list8 = ["D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data8\\deeparch1.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data8\\deeparch2.csv",
             "D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data8\\deeparch3.csv"
             ]

env_list9 = ['D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_100connections_10s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_100connections_10s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_100connections_120s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_100connections_120s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_100connections_30s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_100connections_60s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_300connections_10s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_300connections_120s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_300connections_30s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_300connections_30s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_12threads_300connections_60s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_100connections_10s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_10connections_10s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_10connections_10s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_10connections_120s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_10connections_120s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_10connections_30s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_10connections_30s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_10connections_60s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_10connections_60s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_300connections_10s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_300connections_120s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_300connections_120s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_300connections_30s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_300connections_60s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_4threads_300connections_60s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_10connections_10s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_10connections_10s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_10connections_120s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_10connections_120s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_10connections_30s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_10connections_30s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_10connections_60s_False.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_10connections_60s_true.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data9\\data_8threads_300connections_10s_False.csv']

env_list10 = ['D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_20tbls_200ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_20tbls_500ksizes_10rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_20tbls_50ksizes_01rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_40tbls_500ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_60tbls_100ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_60tbls_200ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_60tbls_500ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_60tbls_500ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_80tbls_100ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_80tbls_200ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_80tbls_200ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_120s_80tbls_50ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_20tbls_100ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_20tbls_100ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_20tbls_200ksizes_01rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_20tbls_200ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_40tbls_100ksizes_01rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_40tbls_100ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_40tbls_500ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_40tbls_50ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_60tbls_100ksizes_10rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_60tbls_200ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_60tbls_500ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_80tbls_200ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_180s_80tbls_500ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_20tbls_100ksizes_01rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_20tbls_100ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_20tbls_500ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_40tbls_200ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_40tbls_500ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_40tbls_50ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_40tbls_50ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_60tbls_500ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_60tbls_500ksizes_10rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_60tbls_50ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_80tbls_100ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_80tbls_100ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_80tbls_200ksizes_01rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_80tbls_500ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_80tbls_50ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_30s_80tbls_50ksizes_10rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_40tbls_100ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_40tbls_100ksizes_10rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_40tbls_200ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_40tbls_500ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_40tbls_500ksizes_10rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_40tbls_500ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_40tbls_50ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_60tbls_100ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_60tbls_200ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_60tbls_500ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_60tbls_500ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_100ksizes_01rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_100ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_200ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_200ksizes_55rw_32thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_500ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_500ksizes_10rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_500ksizes_55rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_50ksizes_01rw_16thds_size.csv', 'D:\\CodePycharm\\online_dal-main\\data\\drift_data\\data10\\factor_data_60s_80tbls_50ksizes_01rw_32thds_size.csv']







def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y


def generate_unique_random_numbers(n, seed=0):
    # 创建一个包含0到n的数组
    # random.seed(seed)
    numbers = list(range(n + 1))
    # Fisher-Yates shuffle算法
    for i in range(len(numbers) - 1, 0, -1):
        j = random.randint(0, i)
        numbers[i], numbers[j] = numbers[j], numbers[i]

    return numbers

system_sample_rate = {
    "lighttpd": (0.04, 0.08),
    "spear": (0.1, 0.15)
}
def construct_drift_stream(
    system_name,
    envs,
    n_env,
    recurring,
    minr=0.4,
    maxr=0.8,
    max_count=1500,
    min_count=400
):
    """
    Construct a data stream with concept drift.

    Parameters
    ----------
    system_name : str
        Name of the software system.

    envs : list[str]
        List of csv file paths, each representing an environment.

    n_env : int
        Number of environments (drifts) in the stream.

    recurring : bool
        Whether recurring environments exist.

    minr : float
        Minimum sampling ratio.

    maxr : float
        Maximum sampling ratio.

    max_count : int
        Maximum number of samples for one environment.

    min_count : int
        Minimum number of samples for one environment.

    Returns
    -------
    dict
        {
            "system": system_name,
            "init_data": DataFrame,
            "whole_data": list,
            "sequence": list
        }
    """

    if system_name in system_sample_rate:
        minr, maxr = system_sample_rate[system_name]

    n_drift = n_env

    # ---------- determine environment sequence ----------
    if recurring:

        # select a subset of environments
        k = max(1, n_drift // 2)
        k = min(k, len(envs))

        env_subset = random.sample(envs, k)

        env_sequence = []

        # first environment
        prev_env = random.choice(env_subset)
        env_sequence.append(prev_env)

        # remaining environments
        for _ in range(n_drift - 1):

            # avoid consecutive duplicate environments
            candidates = [e for e in env_subset if e != prev_env]

            if len(candidates) == 0:
                next_env = prev_env
            else:
                next_env = random.choice(candidates)

            env_sequence.append(next_env)
            prev_env = next_env

    else:

        if n_drift > len(envs):
            n_drift = len(envs)

        env_sequence = random.sample(envs, n_drift)

    init_data = []
    test_data = []

    print(f"env sequence: {env_sequence}")

    # ---------- build stream ----------
    for i, env_path in enumerate(env_sequence):

        df = pd.read_csv(env_path)

        min_n_samples = int(minr * len(df))
        max_n_samples = int(maxr * len(df))

        # apply sample count constraints
        max_n_samples = min(max_n_samples, max_count)
        min_n_samples = max(min_n_samples, min_count)

        # ensure valid range
        if max_n_samples < min_n_samples:
            max_n_samples = min_n_samples

        sample_size = random.randint(min_n_samples, max_n_samples)
        sample_size = max(1, sample_size)

        # avoid sampling more than dataset size
        sample_size = min(sample_size, len(df))

        sampled_df = df.sample(n=sample_size, replace=False)

        if i == 0:
            # first environment used for initialization

            if len(sampled_df) <= 50:
                init_part = sampled_df
                test_part = pd.DataFrame(columns=sampled_df.columns)

            else:
                init_part = sampled_df.sample(n=50)
                test_part = sampled_df.drop(init_part.index)

            init_data = init_part.values
            test_data.append(test_part.values)

        else:
            test_data.append(sampled_df.values)

    return {
        "system": system_name,
        "init_data": init_data,
        "whole_data": test_data,
        "sequence": env_sequence
    }


def process_training_data(multi_env_list=[], n_drifts=4, min_occupation=0.4, max_occupation=0.8, max_count=2000, min_count=200,
                          init_train_count=50, reuse_test=False, given_list=None):
    if len(multi_env_list)>30:
        max_count = 800
    count_list = {}
    env_count = len(multi_env_list)
    selected_list = generate_unique_random_numbers(env_count - 1)
    whole_data = None
    change_time = 0
    init_data = None
    count = 0
    change_point = list()
    if not reuse_test:
        random_list = selected_list
    elif given_list is not None:
        random_list = given_list
    else:
        selected_list = selected_list[:6]
        random_list = selected_list * 2
    logging.info(f"data list {random_list}")
    """
    处理逻辑：
        如果reuse_test False && given_list = None ,则随机选择n_drifts个数据集,每个数据集随机采样一部分数据,作为流数据;
        如果reuse_test True && given_list != None,则按照given_list的顺序选择数据集,每个数据集随机采样一部分数据,作为流数据;
        如果reuse_test True && given_list = None, 则随机选择3个数据集,每个数据集随机采样一部分数据,作为流数据;然后这3个数据集重复一次,作为流数据;
    """
    for i in random_list:
        if count == n_drifts:
            break

        whole_data_temp = get_whole_data(multi_env_list[i])
        data_num = len(whole_data_temp)
        low = int(data_num * min_occupation)
        high = int(data_num * max_occupation)
        sample_count = random.randint(low, high)
        if sample_count > max_count:
            sample_count = max_count
        dataset_name = multi_env_list[i].split('\\')[-1]
        count_list[count] = [dataset_name, sample_count]
        count += 1

        all_index = np.array(list(range(data_num)))
        # np.random.seed(seed)
        data_index = np.random.choice(all_index, size=sample_count)
        data = whole_data_temp[data_index]
        change_time += 1

        if whole_data is None:
            # whole_data = data
            whole_data = []
            init_count = len(data)
            init_test_count = init_count - init_train_count
            all_index_temp = np.array(list(range(len(data))))

            train_index_temp = np.random.choice(all_index_temp, size=init_train_count)
            train_data = data[train_index_temp]
            test_index_temp = np.setdiff1d(all_index_temp, train_index_temp)
            test_data = data[test_index_temp]
            whole_data.append(test_data)
            init_data = train_data
        else:
            whole_data.append(data)

        change_point.append(len(whole_data))
    # print(f"change time is {change_time}")
    logging.info(f"change time is {change_time}")
    print("data info")
    # print(f"init train data count = {init_train_count}, stream data count = {len(whole_data)}")
    # logging.info(f"init train data count = {init_train_count}, stream data count = {len(whole_data)}")
    true_change_point = []
    all = 0
    for env_data in whole_data:
        all += len(env_data)
        true_change_point.append(len(env_data))
    data_count_list = []
    for data in whole_data:
        data_count_list.append(len(data))
    logging.info(f"test env list: {random_list}, test env count: {data_count_list}")
    return {'init_data': init_data, 'whole_data': whole_data, 'true_change_point': true_change_point}


def plot_show(performance_dict, save_fig, save_path=None, title=None):
    plt.figure()
    mean_perf = {}
    for k, v in performance_dict.items():
        length = len(v)
        x = list(range(length))
        plt.plot(x, v, label=k)
        mean_perf[k] = round(np.mean(v), 4)
    plt.xlabel(mean_perf)
    plt.legend()
    plt.title(title)
    if save_fig:
        plt.savefig(save_path)
        # logging.info(f"figure is saved, path={file_path}")
    # plt.show()

def test_dhda_reuse(name, test_data):
    data = construct_drift_stream(name, test_data, 6, recurring=True)
    test_data_list = data['whole_data']
    train_data = data['init_data']
    train_x, train_y = get_attributes_result(train_data)
    performance_dict = {}
    time_cost_dict = {}

    logging.info("--------dhda reuse model with all--------")
    random.seed(seed)
    np.random.seed(seed)
    dhda_reuse_ALL_model = DaL_Regressor(MODEL_REUSE=True, Smooth=True, CPD=True)
    # dhda_model.fit_model(train_x, np.ravel(train_y))
    dhda_reuse_ALL_model.fit_model(train_x, train_y)

    dhda_reuse_ALL_error_list = []
    dhda_reuse_ALL_y = []

    c = 0
    time0 = time.time()
    for index, env_data in enumerate(test_data_list):
        env_test_x, env_test_y = get_attributes_result(env_data)
        test_size = len(env_test_y)

        for i in range(test_size):
            c += 1
            y_pre = dhda_reuse_ALL_model.predict(env_test_x[i][np.newaxis, :])
            val = 0
            # val = abs(y_pre - y_test[i])[0][0] change
            if env_test_y[i] != 0:
                val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                dhda_reuse_ALL_error_list.append(val)
                dhda_reuse_ALL_model.learn_data(env_test_x[i][np.newaxis, :], env_test_y[i][np.newaxis, :], val)
            if c % 32 == 0:
                err_reuse_ALL_dhda = np.mean(dhda_reuse_ALL_error_list)
                dhda_reuse_ALL_y.append(err_reuse_ALL_dhda)
                dhda_reuse_ALL_error_list = list()
    time1 = time.time()
    performance_dict["dhda_reuse_ALL_model"] = dhda_reuse_ALL_y
    time_cost_dict["dhda_reuse_ALL_model"] = time1 - time0

    # The original data4-data8 scripts also evaluate the default min_sample.
    if name in {"spear", "sqlite", "nginx", "exastencils", "deeparch"}:
        logging.info("--------dhda model--------")
        random.seed(seed)
        np.random.seed(seed)
        dhda_error_list = []
        dhda_y = []

        c = 0
        dhda_model = DaL_Regressor(MODEL_REUSE=False, Smooth=False, CPD=False)
        dhda_model.fit_model(train_x, train_y)
        time0 = time.time()
        for index, env_data in enumerate(test_data_list):
            env_test_x, env_test_y = get_attributes_result(env_data)
            test_size = len(env_test_y)

            for i in range(test_size):
                c += 1
                y_pre = dhda_model.predict(env_test_x[i][np.newaxis, :])
                val = 0
                if env_test_y[i] != 0:
                    val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                    dhda_error_list.append(val)
                    dhda_model.learn_data(
                        env_test_x[i][np.newaxis, :],
                        env_test_y[i][np.newaxis, :],
                        val,
                    )
                if c % 32 == 0:
                    err_dhda = np.mean(dhda_error_list)
                    dhda_y.append(err_dhda)
                    dhda_error_list = []
        time1 = time.time()
        performance_dict["dhda_model"] = dhda_y
        time_cost_dict["dhda_model"] = time1 - time0

    logging.info("--------dhda model--------")
    random.seed(seed)
    np.random.seed(seed)
    dhda_error_list = []
    dhda_y = []

    c = 0
    dhda_model = DaL_Regressor(MODEL_REUSE=False, Smooth=False, CPD=False, min_sample=8)
    dhda_model.fit_model(train_x, train_y)
    time0 = time.time()
    for index, env_data in enumerate(test_data_list):
        env_test_x, env_test_y = get_attributes_result(env_data)
        test_size = len(env_test_y)

        for i in range(test_size):
            c += 1
            y_pre = dhda_model.predict(env_test_x[i][np.newaxis, :])
            val = 0
            # val = abs(y_pre - y_test[i])[0][0] change
            if env_test_y[i] != 0:
                val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                dhda_error_list.append(val)
                dhda_model.learn_data(env_test_x[i][np.newaxis, :], env_test_y[i][np.newaxis, :], val)
            if c % 32 == 0:
                err_dhda = np.mean(dhda_error_list)
                dhda_y.append(err_dhda)
                dhda_error_list = list()
    time1 = time.time()
    performance_dict["dhda_model8"] = dhda_y
    time_cost_dict["dhda_model8"] = time1 - time0


    online_sotas = ["ARF", "SRP"]
    for sota in online_sotas:
        print(sota)

        random.seed(seed)
        np.random.seed(seed)
        online_error_list = []
        online_y = []
        c = 0
        online_model = build_incremental_model(sota)
        online_model.fit(train_x, train_y)
        time0 = time.time()
        for index, env_data in enumerate(test_data_list):
            env_test_x, env_test_y = get_attributes_result(env_data)
            test_size = len(env_test_y)

            for i in range(test_size):
                c += 1
                y_pre = online_model.predict(env_test_x[i][np.newaxis, :])
                val = 0
                # val = abs(y_pre - y_test[i])[0][0] change
                if env_test_y[i] != 0:
                    val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                    online_error_list.append(val)
                newx = env_test_x[i][np.newaxis, :]
                newy = env_test_y[i][np.newaxis, :]
                online_model.update(newx, newy)
                if c % 32 == 0:
                    err_online = np.mean(online_error_list)
                    online_y.append(err_online)
                    online_error_list = list()
        time1 = time.time()
        performance_dict[sota] = online_y
        time_cost_dict[sota] = time1 - time0

    re_baseline = ["DaL"]
    for baseline in re_baseline:
        print(baseline)
        random.seed(seed)
        np.random.seed(seed)
        baseline_error_list = []
        baseline_y = []
        c = 0
        x_accessible = train_x
        y_accessible = train_y
        baseline_model = DaL_Regressor()
        baseline_model.fit_model(train_x, train_y)
        time0 = time.time()
        for index, env_data in enumerate(test_data_list):
            env_test_x, env_test_y = get_attributes_result(env_data)
            test_size = len(env_test_y)

            for i in range(test_size):
                c += 1
                y_pre = baseline_model.predict(env_test_x[i][np.newaxis, :])
                val = 0
                # val = abs(y_pre - y_test[i])[0][0] change
                if env_test_y[i] != 0:
                    val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                    baseline_error_list.append(val)

                x_accessible = np.vstack((x_accessible, env_test_x[i][np.newaxis, :]))
                y_accessible = np.vstack((y_accessible, env_test_y[i][np.newaxis, :]))

                if c % 32 == 0:
                    baseline_model = DaL_Regressor()
                    baseline_model.fit_model(x_accessible, y_accessible)
                    err_baseline = np.mean(baseline_error_list)
                    baseline_y.append(err_baseline)
                    baseline_error_list = list()
        time1 = time.time()
        performance_dict[baseline] = baseline_y
        time_cost_dict[baseline] = time1 - time0
    return performance_dict, time_cost_dict




def set_logging(filename):
    # work_dir = 'D:\\CodePycharm\\online_dal-main\\new_dhda\\result\\reuse'
    # 构造日志文件路径
    save_path = work_dir + f"\\log\\{filename}_log.txt"
    # 获取根日志器
    logger = logging.getLogger()
    # 移除所有已存在的处理器（避免重复输出到之前的文件）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建新的文件处理器
    file_handler = logging.FileHandler(save_path, mode='w')
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 添加新的处理器
    logger.addHandler(file_handler)


ENV_INDEX2NAME = {0: "sac", 1: "x264", 2: "storm", 3: "spear", 4: "sqlite", 5: "nginx", 6: "exastencils", 7: "deeparch", 8: "lighttpd", 9: "mysql"}

if __name__ == '__main__':

    random.seed(seed)
    np.random.seed(seed)

    envs = [env_list1, env_list2, env_list3, env_list4,
            env_list5, env_list6, env_list7, env_list8]

    for i, env in enumerate(envs):

        # Preserve the old 1-3 / 4-6 / 7-8 script boundaries.
        if i in {0, 3, 6}:
            seed = 43
            random.seed(seed)
            np.random.seed(seed)
        log_filename = ENV_INDEX2NAME[i]
        csv_save_path = work_dir + f"\\csv\\{log_filename}_time_result.csv"
        result_record = defaultdict(list)
        for index in range(30):
            seed += 2*index + 1
            random.seed(seed)
            np.random.seed(seed)
            set_logging(log_filename + f"_{index}_test")
            logging.info(f"the {i}-th env is testing")
            logging.info(f"the {index}-th testing of the {i}-th env")
            performance_dict, time_cost_dict = test_dhda_reuse(log_filename, env)
            fig_save_path = work_dir + f"\\figures\\"
            fig_save_path_filename = fig_save_path + f"{log_filename}_{index}_test_figure.png"
            plot_show(performance_dict, save_fig=True, title=f"the {index} test of {log_filename}",
                      save_path=fig_save_path_filename)
            """result_record['dhda'].append(np.mean(performance_dict['dhda_model']))
            result_record['dhda_reuse'].append(np.mean(performance_dict['dhda_reuse_model']))
            result_record['dhda_reuse_ALL'].append(np.mean(performance_dict['dhda_reuse_ALL_model']))
            result_record['dhda_reuse_CPD'].append(np.mean(performance_dict['dhda_reuse_CPD_model']))
            result_record['dhda_reuse_SMOOTH'].append(np.mean(performance_dict['dhda_reuse_SMOOTH_model']))"""
            """for key, value in performance_dict.items():
                result_record[key].append(np.mean(value))""" # for err
            for key, value in time_cost_dict.items():
                result_record[key].append(value)  # for time
            print(f"all method time cost: {time_cost_dict}")
        result_df = pd.DataFrame(result_record)
        result_df.to_csv(csv_save_path)
