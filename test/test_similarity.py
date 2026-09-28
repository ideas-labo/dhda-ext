import math
import random
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from skmultiflow.drift_detection import ADWIN
from scipy.stats import wilcoxon
import logging

### its a test
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


def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y


def split_data(arr, m, replace=False):
    """
    从数组中抽取m条数据，并同时返回抽取的数据和未被抽取的数据

    参数:
        arr: 输入的ndarray，形状为(2048, n)
        m: 要抽取的数据条数
        replace: 是否有放回抽样，默认为False（无放回）

    返回:
        sampled: 抽取的m条数据，形状为(m, n)
        unsampled: 未被抽取的数据，形状为(2048-m, n)
    """
    # 获取总行数
    total = arr.shape[0]

    # 验证参数合法性
    if replace is False and m > total:
        raise ValueError("当replace=False时，m不能大于数据总条数")

    # 生成随机索引
    sampled_indices = np.random.choice(total, size=m, replace=replace)

    # 计算未被抽取的索引（补集）
    # 创建全量索引
    all_indices = np.arange(total)
    # 使用布尔掩码找到未被选中的索引
    mask = np.isin(all_indices, sampled_indices, invert=True)
    unsampled_indices = all_indices[mask]

    # 获取抽取的数据和未被抽取的数据
    sampled = arr[sampled_indices]
    unsampled = arr[unsampled_indices]

    return sampled, unsampled


def kl_div(p, q):
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(rel_entr(p, q))


def similarity_compute_k(data1, data2, mode, smooth, sigma=2, k=5):

    if mode == 0:
        x1, y1 = get_attributes_result(data1)
        x2, y2 = get_attributes_result(data2)

        y1 = y1.reshape(-1, 1)
        y2 = y2.reshape(-1, 1)

        K_test_result = []

        # 使用KFold替代train_test_split
        kf1 = KFold(n_splits=k, shuffle=True, random_state=42)
        kf2 = KFold(n_splits=k, shuffle=True, random_state=42)

        for (train_idx1, test_idx1), (train_idx2, test_idx2) in zip(kf1.split(x1), kf2.split(x2)):

            one_result = 1

            x1_train, x1_test = x1[train_idx1], x1[test_idx1]
            y1_train, y1_test = y1[train_idx1], y1[test_idx1]

            x2_train, x2_test = x2[train_idx2], x2[test_idx2]
            y2_train, y2_test = y2[train_idx2], y2[test_idx2]

            rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
            rf1.fit(x1_train, np.ravel(y1_train))

            rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
            rf2.fit(x2_train, np.ravel(y2_train))

            all_test_x = np.concatenate((x1_test, x2_test), axis=0)
            all_test_y = np.concatenate((y1_test, y2_test), axis=0)

            test1_len = len(y1_test)

            all_pre_y1 = rf1.predict(all_test_x)
            all_pre_y2 = rf2.predict(all_test_x)

            all_test_y_squeezed = all_test_y.squeeze()

            non_zero_mask = all_test_y_squeezed != 0

            non_zero_pre1 = all_pre_y1[non_zero_mask]
            non_zero_pre2 = all_pre_y2[non_zero_mask]
            non_zero_test = all_test_y_squeezed[non_zero_mask]

            err_list1 = np.abs(non_zero_pre1 - non_zero_test) / non_zero_test
            err_list2 = np.abs(non_zero_pre2 - non_zero_test) / non_zero_test

            if len(err_list1) < 256 or len(err_list2) < 256:
                w1 = math.ceil(256 / len(err_list1))
                w2 = math.ceil(256 / len(err_list2))
                err_list1 = np.tile(err_list1, w1)
                err_list2 = np.tile(err_list2, w2)

            logging.info(
                f"select to combine: \nerr_list1 info:\nmean: {np.mean(err_list1[:test1_len])} "
                f"to {np.mean(err_list1[test1_len:])}\nstd: {np.std(err_list1[:test1_len])} "
                f"to {np.std(err_list1[test1_len:])}"
            )

            logging.info(
                f"\nerr_list2 info:\nmean: {np.mean(err_list2[:test1_len])} "
                f"to {np.mean(err_list2[test1_len:])}\nstd: {np.std(err_list2[:test1_len])} "
                f"to {np.std(err_list2[test1_len:])}"
            )

            adwin1 = ADWIN()
            adwin2 = ADWIN()

            if smooth:
                err_list1 = gaussian_filter1d(err_list1, sigma)
                err_list2 = gaussian_filter1d(err_list2, sigma)

            for i in range(len(err_list1)):
                adwin1.add_element(err_list1[i])
                adwin2.add_element(err_list2[i])

                if (adwin1.detected_change() or adwin2.detected_change()) and i > test1_len:
                    one_result = 0
                    break

            K_test_result.append(one_result)

        logging.info(f"select to combine: vote result: {K_test_result}")

        return np.sum(K_test_result) > (k // 2)

    if mode == 1:
        if len(data1) > len(data2):
            support_data = data1
            valid_data = data2
        else:
            support_data = data2
            valid_data = data1

        K_test_result = []
        for i in range(k):
            one_result = 1
            compare_data, train_data = split_data(support_data, int(0.2 * (len(support_data))))
            x_train, y_train = get_attributes_result(train_data)
            x_valid, y_valid = get_attributes_result(valid_data)
            x_compare, y_compare = get_attributes_result(compare_data)

            y_compare = y_compare.reshape(-1, 1)
            mask_compare = np.ravel(y_compare) != 0
            y_train = y_train.reshape(-1, 1)
            y_valid = y_valid.reshape(-1, 1)
            mask_valid = np.ravel(y_valid) != 0

            rf_support = RandomForestRegressor()
            rf_support.fit(x_train, np.ravel(y_train))

            all_valid_pre = rf_support.predict(x_valid)
            all_compare_pre = rf_support.predict(x_compare)

            non_zero_y_valid = np.ravel(y_valid)[mask_valid]
            non_zero_pre_valid = all_valid_pre[mask_valid]

            non_zero_y_compare = np.ravel(y_compare)[mask_compare]
            non_zero_pre_compare = all_compare_pre[mask_compare]

            valid_err = np.abs(non_zero_y_valid - non_zero_pre_valid) / non_zero_y_valid
            compare_err = np.abs(non_zero_y_compare - non_zero_pre_compare) / non_zero_y_compare
            logging.info(
                f"select to reuse: \nvalid_err info:\nrange: {min(valid_err)} to {max(valid_err)}\nmean: {np.mean(valid_err)}\nstd: {np.std(valid_err)}\n")
            logging.info(
                f"select to reuse: \ncompare_err info:\nrange: {min(compare_err)} to {max(compare_err)}\nmean: {np.mean(compare_err)}\nstd: {np.std(compare_err)}\n")

            adwin = ADWIN()
            if len(valid_err) < 256:
                wv = math.ceil(256 / len(valid_err))
                wc = math.ceil(256 / len(compare_err))

                bs_valid_err = np.tile(valid_err, wv)
                bs_compare_err = np.tile(compare_err, wc)
            else:
                bs_valid_err = valid_err
                bs_compare_err = compare_err

            if smooth:
                bs_compare_err = gaussian_filter1d(bs_compare_err, sigma)
                bs_valid_err = gaussian_filter1d(bs_valid_err, sigma)

            for err1 in bs_compare_err:
                adwin.add_element(err1)

            for err2 in bs_valid_err:
                adwin.add_element(err2)
                if adwin.detected_change():
                    one_result = 0  # diff
                    break

            K_test_result.append(one_result)
        logging.info(f"vote result: {K_test_result}")
        # return true when same
        return np.sum(K_test_result) > 3
    return None


def shuffle_and_split(arr, k):
    """
    将给定数组打乱后平均分成 k 段（最后一段可能数量不同）

    参数:
        arr (list): 要处理的原始数组
        k (int): 要分成的段数

    返回:
        list: 包含 k 个子数组的列表，每个子数组是打乱后的数据段
    """
    # 1. 检查输入合法性
    if not isinstance(arr, list):
        raise TypeError("第一个参数必须是列表类型")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("第二个参数必须是正整数")
    if k > len(arr):
        raise ValueError(f"段数 k={k} 不能大于数组长度 {len(arr)}")

    # 2. 复制原始数组并打乱（避免修改原数组）
    shuffled_arr = arr.copy()
    random.shuffle(shuffled_arr)

    # 3. 计算每段的基础长度和剩余元素数量
    n = len(shuffled_arr)
    base_length = n // k  # 每段的基础长度
    remainder = n % k  # 剩余的元素数量（前 remainder 段会多1个元素）

    # 4. 分割数组
    result = []
    start = 0
    for i in range(k):
        # 计算当前段的长度：前 remainder 段长度为 base_length + 1，其余为 base_length
        current_length = base_length + 1 if i < remainder else base_length
        # 截取当前段
        current_segment = shuffled_arr[start:start + current_length]
        result.append(current_segment)
        # 更新下一段的起始索引
        start += current_length

    return result


def similarity_compute(data1, data2, mode, smooth, sigma=2, k=3):
    if mode == 0:
        x1, y1 = get_attributes_result(data1)
        x2, y2 = get_attributes_result(data2)
        y1 = y1.squeeze()
        y2 = y2.squeeze()
        data1_indexes = shuffle_and_split(list(range(len(data1))), k)
        data2_indexes = shuffle_and_split(list(range(len(data2))), k)
        error1_list = []
        error2_list = []
        for i in range(k):

            x1_test = x1[data1_indexes[i]]
            x2_test = x2[data2_indexes[i]]
            y1_test = y1[data1_indexes[i]]
            y2_test = y2[data2_indexes[i]]
            data1_train_index = [item for lst in data1_indexes for item in lst if data1_indexes.index(lst) != i]
            data2_train_index = [item for lst in data2_indexes for item in lst if data2_indexes.index(lst) != i]
            x1_train = x1[data1_train_index]
            x2_train = x2[data2_train_index]
            y1_train = y1[data1_train_index]
            y2_train = y2[data2_train_index]

            rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
            rf1.fit(x1_train, y1_train)
            rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
            rf2.fit(x2_train, y2_train)

            pre_y1 = rf1.predict(x1_test)
            pre_y2 = rf2.predict(x2_test)

            # 创建掩码：筛选出all_test_y中非零的元素
            non_zero_mask1 = y1_test != 0
            non_zero_mask2 = y2_test != 0

            # 只保留非零元素对应的预测值和真实值
            non_zero_pre1 = pre_y1[non_zero_mask1]
            non_zero_pre2 = pre_y2[non_zero_mask2]
            non_zero_test1 = y1_test[non_zero_mask1]
            non_zero_test2 = y2_test[non_zero_mask2]

            # 计算非零元素的相对误差
            err_list1 = np.abs(non_zero_pre1 - non_zero_test1) / non_zero_test1
            err_list2 = np.abs(non_zero_pre2 - non_zero_test2) / non_zero_test2

            error1_list.extend(err_list1)
            error2_list.extend(err_list2)

        # error1 list & error2 list are the concept performance on itself, computing to compare to others
        #
        if len(error1_list) < 256 or len(error2_list) < 256:
            w1 = math.ceil(256 / len(error1_list))
            w2 = math.ceil(256 / len(error2_list))

            error1_list = np.tile(error1_list, w1)
            error2_list = np.tile(error2_list, w2)

        # compare error1
        rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf1.fit(x1, y1)
        rf2.fit(x2, y2)
        model1_data2_pre = rf1.predict(x2)
        model2_data1_pre = rf2.predict(x1)
        none_zero_mask12 = y2 != 0
        none_zero_mask21 = y1 != 0

        none_zero_pre12 = model1_data2_pre[none_zero_mask12]
        none_zero_pre21 = model2_data1_pre[none_zero_mask21]
        none_zero_test12 = y2[none_zero_mask12]
        none_zero_test21 = y1[none_zero_mask21]

        none_zero_mape12 = np.abs(none_zero_pre12 - none_zero_test12) / none_zero_test12
        none_zero_mape21 = np.abs(none_zero_pre21 - none_zero_test21) / none_zero_test21


        logging.info(
            f"select to combine: \nerr_list1 info:\nmean: {np.mean(error1_list)} to {np.mean(none_zero_mape12)}\nstd: {np.std(error1_list)} to {np.std(none_zero_mape12)}")
        logging.info(
            f"\nerr_list2 info:\nmean: {np.mean(error2_list)} to {np.mean(none_zero_mape21)}\nstd: {np.std(error2_list)} to {np.std(none_zero_mape21)}")

        adwin1 = ADWIN()
        adwin2 = ADWIN()
        if smooth:
            error1_list = gaussian_filter1d(error1_list, sigma)
            error2_list = gaussian_filter1d(error2_list, sigma)
            none_zero_mape12 = gaussian_filter1d(none_zero_mape12, sigma)
            none_zero_mape21 = gaussian_filter1d(none_zero_mape21, sigma)
            logging.info(
                f"select to combine: \n(smooth)err_list1 info:\nmean: {np.mean(error1_list)} to {np.mean(none_zero_mape12)}\nstd: {np.std(error1_list)} to {np.std(none_zero_mape12)}")
            logging.info(
                f"\n(smooth)err_list2 info:\nmean: {np.mean(error2_list)} to {np.mean(none_zero_mape21)}\nstd: {np.std(error2_list)} to {np.std(none_zero_mape21)}")
        ob_seq1 = np.concatenate((error1_list, none_zero_mape12))
        ob_seq2 = np.concatenate((error2_list, none_zero_mape21))
        for i in range(min(len(ob_seq1), len(ob_seq2))):
            adwin1.add_element(ob_seq1[i])
            adwin2.add_element(ob_seq2[i])
            if adwin1.detected_change() or adwin2.detected_change():
                logging.info("select to combine: False")
                return False
        return True

    if mode == 1:
        if len(data1) > len(data2):
            support_data = data1
            valid_data = data2
        else:
            support_data = data2
            valid_data = data1

        support_indexes = shuffle_and_split(list(range(len(support_data))), k)
        compare_errs = []
        valid_errs = []
        for i in range(k):
            compare_data = support_data[support_indexes[i]]
            train_data_index = [item for lst in support_indexes for item in lst if support_indexes.index(lst) != i] # todo: check
            train_data = support_data[train_data_index]

            x_train, y_train = get_attributes_result(train_data)
            x_valid, y_valid = get_attributes_result(valid_data)
            x_compare, y_compare = get_attributes_result(compare_data)
            y_train = y_train.squeeze()
            y_valid = y_valid.squeeze()
            y_compare = y_compare.squeeze()
            mask_compare = y_compare != 0

            mask_valid = y_valid != 0

            rf_support = RandomForestRegressor()
            rf_support.fit(x_train, y_train)

            all_valid_pre = rf_support.predict(x_valid)
            all_compare_pre = rf_support.predict(x_compare)

            non_zero_y_valid = y_valid[mask_valid]
            non_zero_pre_valid = all_valid_pre[mask_valid]

            non_zero_y_compare = y_compare[mask_compare]
            non_zero_pre_compare = all_compare_pre[mask_compare]

            valid_err = np.abs(non_zero_y_valid - non_zero_pre_valid) / non_zero_y_valid
            compare_err = np.abs(non_zero_y_compare - non_zero_pre_compare) / non_zero_y_compare

            compare_errs.extend(compare_err)
            valid_errs.extend(valid_err)

        logging.info(f"select to reuse: \nerror list info:\nmean: {np.mean(compare_errs)} to {np.mean(valid_errs)}\nstd: {np.std(compare_errs)} to {np.std(valid_errs)}")

        adwin = ADWIN()
        if len(valid_errs) < 256:
            wv = math.ceil(256 / len(valid_errs))
            wc = math.ceil(256 / len(compare_errs))

            bs_valid_err = np.tile(valid_errs, wv)
            bs_compare_err = np.tile(compare_errs, wc)
        else:
            bs_valid_err = valid_errs
            bs_compare_err = compare_errs

        if smooth:
            bs_compare_err = gaussian_filter1d(bs_compare_err, sigma)
            bs_valid_err = gaussian_filter1d(bs_valid_err, sigma)

        ob_seq = np.concatenate((bs_compare_err, bs_valid_err))

        for err in ob_seq:
            adwin.add_element(err)
            if adwin.detected_change():
                return False
        return True


def compute_similarity_by_ML(data1, data2, test_mode=0, method='KL', measure='APE', smooth=True):
    if smooth:
        smooth_parm = {'method': 'Gaussian',
                       'is_smooth': True,
                       'sigma': 2
                       }
    if test_mode == 0:
        # both data1 and data2 is enough
        # happen when a drift happen and old data was forgotten and stored in history set, compare it with other history concept
        x1, y1 = get_attributes_result(data1)
        x2, y2 = get_attributes_result(data2)
        y1 = y1.reshape(-1, 1)
        y2 = y2.reshape(-1, 1)

        x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20, random_state=42)
        x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.20, random_state=42)

        rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf1.fit(x1_train, np.ravel(y1_train))
        rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf2.fit(x2_train, np.ravel(y2_train))

        err_list1 = []
        err_list2 = []

        all_test_x = np.concatenate((x1_test, x2_test), axis=0)
        all_test_y = np.concatenate((y1_test, y2_test), axis=0)

        test1_len = len(y1_test)
        test2_len = len(y2_test)

        all_pre_y1 = rf1.predict(all_test_x)
        all_pre_y2 = rf2.predict(all_test_x)

        if measure == 'APE':
            all_test_y_squeezed = all_test_y.squeeze()

            # 创建掩码：筛选出all_test_y中非零的元素
            non_zero_mask = ~np.isclose(all_test_y_squeezed, 0)

            # 只保留非零元素对应的预测值和真实值
            non_zero_pre1 = all_pre_y1[non_zero_mask]
            non_zero_test = all_test_y_squeezed[non_zero_mask]

            # 计算非零元素的相对误差
            err_list1 = np.abs(non_zero_pre1 - non_zero_test) / non_zero_test

            # 确保数据形状一致
            all_test_y_squeezed = all_test_y.squeeze()

            # 只保留非零元素对应的预测值和真实值
            non_zero_pre2 = all_pre_y2[non_zero_mask]

            # 计算非零元素的相对误差
            err_list2 = np.abs(non_zero_pre2 - non_zero_test) / non_zero_test
        if measure == 'AE':
            err_list1 = np.abs(all_pre_y1 - all_test_y.squeeze())
            err_list2 = np.abs(all_pre_y2 - all_test_y.squeeze())
        if measure == 'SE':
            err_list1 = (all_pre_y1 - all_test_y.squeeze()) ** 2
            err_list2 = (all_pre_y2 - all_test_y.squeeze()) ** 2

        if method == 'KL':
            """err_list2_norm = err_list2 / sum(err_list2)
            err_list1_norm = err_list1 / sum(err_list1)
            kl_divergence = np.sum(rel_entr(err_list1_norm, err_list2_norm))"""
            kl_divergence = kl_div(err_list1, err_list2)
            print("KL divergence: ", kl_divergence)
            return kl_divergence < 0.1

        if method == 'ADWIN':
            adwin1 = ADWIN()
            adwin2 = ADWIN()
            if smooth:
                err_list1 = gaussian_filter1d(err_list1, smooth_parm['sigma'])
                err_list2 = gaussian_filter1d(err_list2, smooth_parm['sigma'])
            for i in range(len(err_list1)):
                adwin1.add_element(err_list1[i])
                adwin2.add_element(err_list2[i])
                if (adwin1.detected_change() or adwin2.detected_change()) and i > test1_len:
                    return False
            return True

        if method == 'MAPE':
            # 确保数据形状一致
            all_test_y_squeezed = all_test_y.squeeze()

            # 创建掩码：筛选出all_test_y中非零的元素
            non_zero_mask = ~np.isclose(all_test_y_squeezed, 0)

            # 只保留非零元素对应的预测值和真实值
            non_zero_pre1 = all_pre_y1[non_zero_mask]
            non_zero_test = all_test_y_squeezed[non_zero_mask]

            # 计算非零元素的相对误差
            err_list1 = np.abs(non_zero_pre1 - non_zero_test) / non_zero_test

            # 确保数据形状一致
            all_test_y_squeezed = all_test_y.squeeze()

            # 只保留非零元素对应的预测值和真实值
            non_zero_pre2 = all_pre_y2[non_zero_mask]

            # 计算非零元素的相对误差
            err_list2 = np.abs(non_zero_pre2 - non_zero_test) / non_zero_test

            return (np.mean(err_list1) < 0.1) and (np.mean(err_list2) < 0.1)

        if method == 'WILCOXON':
            diff = err_list1 - err_list2
            diff1 = diff[:test1_len]
            diff2 = diff[test1_len:]
            res1 = wilcoxon(diff1)
            res2 = wilcoxon(diff2)
            print(f"p value1 = {res1.pvalue}")
            print(f"p value2 = {res2.pvalue}")
            return res1.pvalue > 0.05 and res2.pvalue > 0.05

        if method == 'AE-APE-ADWIN':
            adwin1 = ADWIN()
            adwin2 = ADWIN()
            result_ape = True
            result_ae = True

            all_test_y_squeezed = all_test_y.squeeze()

            # 创建掩码：筛选出all_test_y中非零的元素
            non_zero_mask = ~np.isclose(all_test_y_squeezed, 0)

            # 只保留非零元素对应的预测值和真实值
            non_zero_pre1 = all_pre_y1[non_zero_mask]
            non_zero_test = all_test_y_squeezed[non_zero_mask]

            # 计算非零元素的相对误差
            err_list1_ape = np.abs(non_zero_pre1 - non_zero_test) / non_zero_test

            # 确保数据形状一致
            all_test_y_squeezed = all_test_y.squeeze()

            # 只保留非零元素对应的预测值和真实值
            non_zero_pre2 = all_pre_y2[non_zero_mask]

            # 计算非零元素的相对误差
            err_list2_ape = np.abs(non_zero_pre2 - non_zero_test) / non_zero_test
            err_list1_ae = np.abs(all_pre_y1 - all_test_y.squeeze())
            err_list2_ae = np.abs(all_pre_y2 - all_test_y.squeeze())

            for i in range(len(err_list1_ae)):
                adwin1.add_element(err_list1_ae[i])
                adwin2.add_element(err_list2_ae[i])
                if (adwin1.detected_change() or adwin2.detected_change()) and i > test1_len:
                    result_ae = False

            adwin1 = ADWIN()
            adwin2 = ADWIN()

            for i in range(len(err_list1_ape)):
                adwin1.add_element(err_list1_ape[i])
                adwin2.add_element(err_list2_ape[i])
                if (adwin1.detected_change() or adwin2.detected_change()) and i > test1_len:
                    result_ape = False

            return result_ape and result_ae

    if test_mode == 1:
        if len(data1) > len(data2):
            support_data = data1
            valid_data = data2
        else:
            support_data = data2
            valid_data = data1

        compare_data, train_data = split_data(support_data, len(valid_data))

        x_train, y_train = get_attributes_result(train_data)
        x_valid, y_valid = get_attributes_result(valid_data)
        x_compare, y_compare = get_attributes_result(compare_data)

        y_compare = y_compare.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)

        rf_support = RandomForestRegressor()
        rf_support.fit(x_train, np.ravel(y_train))

        all_valid_pre = rf_support.predict(x_valid)
        all_compare_pre = rf_support.predict(x_compare)

        valid_err = []
        compare_err = []

        if measure == 'AE':
            valid_err = abs(all_valid_pre - y_valid.squeeze())
            compare_err = abs(all_compare_pre - y_compare.squeeze())
        elif measure == 'APE':
            non_zero_mask_valid = ~np.isclose(y_valid.squeeze(), 0)
            non_zero_y_valid = y_valid.squeeze()[non_zero_mask_valid]
            non_zero_pre_valid = all_valid_pre[non_zero_mask_valid]

            non_zero_mask_compare = ~np.isclose(y_compare.squeeze(), 0)
            non_zero_y_compare = y_compare.squeeze()[non_zero_mask_compare]
            non_zero_pre_compare = all_compare_pre[non_zero_mask_compare]

            valid_err = np.abs(non_zero_y_valid - non_zero_pre_valid) / non_zero_y_valid
            compare_err = np.abs(non_zero_y_compare - non_zero_pre_compare) / non_zero_y_compare

        elif measure == 'SE':
            valid_err = (all_valid_pre - y_valid.squeeze()) ** 2
            compare_err = (all_compare_pre - y_compare.squeeze()) ** 2

        if method == 'ADWIN':
            adwin = ADWIN()
            if len(valid_err) < 128:
                bs_valid_err = random.choices(valid_err, k=128)
                bs_compare_err = random.choices(compare_err, k=128)
            else:
                bs_valid_err = valid_err
                bs_compare_err = compare_err

            if smooth:
                bs_compare_err = gaussian_filter1d(bs_compare_err, smooth_parm['sigma'])
                bs_valid_err = gaussian_filter1d(bs_valid_err, smooth_parm['sigma'])

            for err1 in bs_compare_err:
                adwin.add_element(err1)

            for err2 in bs_valid_err:
                adwin.add_element(err2)
                if adwin.detected_change():
                    return False  # diff
            return True  # same

        if method == 'MAPE':
            valid_err = abs(all_valid_pre - y_valid.squeeze()) / y_valid.squeeze()
            return np.mean(valid_err) < 0.1

        if method == 'WILCOXON':
            c_len = min(len(compare_err), len(valid_err))
            diff = compare_err[:c_len] - valid_err[:c_len]
            res = wilcoxon(diff)

            print(f"p value = {res.pvalue}")

            return res.pvalue > 0.05

        if method == 'KL':
            err_compare_norm = compare_err / sum(compare_err)
            err_valid_norm = valid_err / sum(valid_err)
            kl_divergence = np.sum(rel_entr(err_compare_norm, err_valid_norm))
            print("KL divergence: ", kl_divergence)
            return kl_divergence < 0.1

        if method == 'AE-APE-ADWIN':
            valid_err_ae = abs(all_valid_pre - y_valid.squeeze())
            compare_err_ae = abs(all_compare_pre - y_compare.squeeze())
            non_zero_mask_valid = ~np.isclose(y_valid.squeeze(), 0)
            non_zero_y_valid = y_valid.squeeze()[non_zero_mask_valid]
            non_zero_pre_valid = all_valid_pre[non_zero_mask_valid]

            non_zero_mask_compare = ~np.isclose(y_compare.squeeze(), 0)
            non_zero_y_compare = y_compare.squeeze()[non_zero_mask_compare]
            non_zero_pre_compare = all_compare_pre[non_zero_mask_compare]

            valid_err_ape = np.abs(non_zero_y_valid - non_zero_pre_valid) / non_zero_y_valid
            compare_err_ape = np.abs(non_zero_y_compare - non_zero_pre_compare) / non_zero_y_compare

            adwin1_result = True
            adwin2_result = True
            adwin = ADWIN()
            if len(valid_err_ae) < 128:
                bs_valid_err = random.choices(valid_err_ae, k=128)
                bs_compare_err = random.choices(compare_err_ae, k=128)
            else:
                bs_valid_err = valid_err_ae
                bs_compare_err = compare_err_ae

            for err1 in bs_compare_err:
                adwin.add_element(err1)

            for err2 in bs_valid_err:
                adwin.add_element(err2)
                if adwin.detected_change():
                    adwin1_result = False

            adwin = ADWIN()
            if len(valid_err_ape) < 128:
                bs_valid_err = random.choices(valid_err_ape, k=128)
                bs_compare_err = random.choices(compare_err_ape, k=128)
            else:
                bs_valid_err = valid_err_ape
                bs_compare_err = compare_err_ape

            for err1 in bs_compare_err:
                adwin.add_element(err1)

            for err2 in bs_valid_err:
                adwin.add_element(err2)
                if adwin.detected_change():
                    adwin2_result = False

            return adwin1_result and adwin2_result


if __name__ == '__main__':
    random.seed(43)
    np.random.seed(43)
    result_path = 'D:\\CodePycharm\\online_dal-main\\new_dhda\\result\\similarity2\\'
    all_systems = [env_list1, env_list2, env_list3, env_list4, env_list5, env_list6, env_list7, env_list8]
    all_method = ['WILCOXON', 'ADWIN', 'AE-APE-ADWIN']
    all_measure = ['APE', 'AE']
    test_mode = [0, 1]
    test_time = 3
    for mode in test_mode:
        for index, sys in enumerate(all_systems):
            df = {'test_time': [], 'test_mode': [], 'env1': [], 'env2': [], 'method': [], 'measure': [],
                  'test_result': []}
            sys_count = len(sys)
            for index1, path1 in enumerate(sys):
                for index2, path2 in enumerate(sys):
                    true_result = (index1 == index2)
                    for method in all_method:
                        for measure in all_measure:
                            if method == 'AE-APE-ADWIN' and measure == 'AE':
                                break
                            for i in range(test_time):
                                data1 = get_whole_data(path1)
                                data2 = get_whole_data(path2)
                                if mode == 0:
                                    sample_size1 = int(len(data1) * 0.5)
                                    if sample_size1 > 2000:
                                        sample_size1 = 2000
                                    random_indices = np.random.choice(data1.shape[0], size=sample_size1, replace=False)
                                    data1 = data1[random_indices]
                                    sample_size2 = int(len(data2) * 0.5)
                                    if sample_size2 > 2000:
                                        sample_size2 = 2000
                                    random_indices = np.random.choice(data2.shape[0], size=sample_size2, replace=False)
                                    data2 = data2[random_indices]
                                    pre_result = compute_similarity_by_ML(data1=data1, data2=data2, test_mode=mode,
                                                                          method=method, measure=measure)
                                    test_result = (true_result == pre_result)
                                    df['test_time'].append(i)
                                    df['test_mode'].append(mode)
                                    df['env1'].append(index1)
                                    df['env2'].append(index2)
                                    df['method'].append(method)
                                    df['measure'].append(measure)
                                    df['test_result'].append(test_result)
                                if mode == 1:
                                    sample_size1 = int(len(data1) * 0.5)
                                    if sample_size1 > 2000:
                                        sample_size1 = 2000
                                    random_indices = np.random.choice(data1.shape[0], size=sample_size1, replace=False)
                                    data1 = data1[random_indices]
                                    sample_size2 = 6
                                    random_indices = np.random.choice(data2.shape[0], size=sample_size2, replace=False)
                                    data2 = data2[random_indices]
                                    pre_result = compute_similarity_by_ML(data1=data1, data2=data2, test_mode=mode,
                                                                          method=method, measure=measure)
                                    test_result = (true_result == pre_result)
                                    df['test_time'].append(i)
                                    df['test_mode'].append(mode)
                                    df['env1'].append(index1)
                                    df['env2'].append(index2)
                                    df['method'].append(method)
                                    df['measure'].append(measure)
                                    df['test_result'].append(test_result)
            df = pd.DataFrame(df)
            filename = f"system{index}_mode{mode}_result.csv"
            finalpath = result_path + filename
            df.to_csv(finalpath, index=False)

def similarity_compute_light(data1, data2, mode, smooth=True, sigma=2, regressor='DT'):
    if mode == 0:
        # ---------------------------
        # 选择回归器
        # ---------------------------
        if regressor == 'DT':
            Regressor = DecisionTreeRegressor
        else:
            Regressor = RandomForestRegressor

        # ---------------------------
        # 数据准备
        # ---------------------------
        x1, y1 = get_attributes_result(data1)
        x2, y2 = get_attributes_result(data2)

        y1 = y1.reshape(-1, 1)
        y2 = y2.reshape(-1, 1)

        kf = KFold(n_splits=2, shuffle=True, random_state=42)

        # 取2折
        (idx11, idx12), (idx21, idx22) = list(kf.split(x1))[0], list(kf.split(x2))[0]

        # data1 两折
        x11, x12 = x1[idx11], x1[idx12]
        y11, y12 = y1[idx11], y1[idx12]

        # data2 两折
        x21, x22 = x2[idx21], x2[idx22]
        y21, y22 = y2[idx21], y2[idx22]

        eps = 1e-8

        # ==================================================
        # 1️⃣ err11：data1 自分布误差
        # ==================================================
        model_11 = Regressor()
        model_11.fit(x11, np.ravel(y11))
        pred_12 = model_11.predict(x12)

        model_12 = Regressor()
        model_12.fit(x12, np.ravel(y12))
        pred_11 = model_12.predict(x11)

        err_12 = np.abs(pred_12 - np.ravel(y12)) / (np.abs(np.ravel(y12)) + eps)
        err_11_ = np.abs(pred_11 - np.ravel(y11)) / (np.abs(np.ravel(y11)) + eps)

        err11 = np.concatenate([err_11_, err_12])

        # ==================================================
        # 2️⃣ err21：data2 模型在 data1 上误差
        # ==================================================
        model_2_all = Regressor()
        model_2_all.fit(x2, np.ravel(y2))
        pred_21 = model_2_all.predict(x1)

        err21 = np.abs(pred_21 - np.ravel(y1)) / (np.abs(np.ravel(y1)) + eps)

        # ==================================================
        # 3️⃣ err22：data2 自分布误差
        # ==================================================
        model_21 = Regressor()
        model_21.fit(x21, np.ravel(y21))
        pred_22 = model_21.predict(x22)

        model_22 = Regressor()
        model_22.fit(x22, np.ravel(y22))
        pred_21_ = model_22.predict(x21)

        err_22 = np.abs(pred_22 - np.ravel(y22)) / (np.abs(np.ravel(y22)) + eps)
        err_21_self = np.abs(pred_21_ - np.ravel(y21)) / (np.abs(np.ravel(y21)) + eps)

        err22 = np.concatenate([err_22, err_21_self])

        # ==================================================
        # 4️⃣ err12：data1 模型在 data2 上误差
        # ==================================================
        model_1_all = Regressor()
        model_1_all.fit(x1, np.ravel(y1))
        pred_12_cross = model_1_all.predict(x2)

        err12 = np.abs(pred_12_cross - np.ravel(y2)) / (np.abs(np.ravel(y2)) + eps)

        # ==================================================
        # 长度不足256则扩充
        # ==================================================
        def expand_to_256(arr):
            if len(arr) < 256:
                w = math.ceil(256 / len(arr))
                arr = np.tile(arr, w)
            return arr[:256]

        err11 = expand_to_256(err11)
        err21 = expand_to_256(err21)
        err22 = expand_to_256(err22)
        err12 = expand_to_256(err12)

        # ==================================================
        # 平滑
        # ==================================================
        if smooth:
            err11 = gaussian_filter1d(err11, sigma)
            err21 = gaussian_filter1d(err21, sigma)
            err22 = gaussian_filter1d(err22, sigma)
            err12 = gaussian_filter1d(err12, sigma)

        # ==================================================
        # ADWIN检测
        # ==================================================
        adwin1 = ADWIN()
        adwin2 = ADWIN()

        # adwin1: err11 -> err21
        for e in err11:
            adwin1.add_element(e)
        drift1 = False
        for e in err21:
            adwin1.add_element(e)
            if adwin1.detected_change():
                drift1 = True
                break

        # adwin2: err22 -> err12
        for e in err22:
            adwin2.add_element(e)
        drift2 = False
        for e in err12:
            adwin2.add_element(e)
            if adwin2.detected_change():
                drift2 = True
                break

        # ==================================================
        # 判定
        # ==================================================
        if (not drift1) and (not drift2):
            return True   # 分布相似
        else:
            return False  # 分布不同
    if mode == 1:
        # ---------------------------
        # 选择回归器
        # ---------------------------
        if regressor == 'DT':
            Regressor = DecisionTreeRegressor
        else:
            Regressor = RandomForestRegressor

        # ---------------------------
        # support / valid 选择
        # ---------------------------
        if len(data1) >= len(data2):
            support_data = data1
            valid_data = data2
        else:
            support_data = data2
            valid_data = data1

        xs, ys = get_attributes_result(support_data)
        xv, yv = get_attributes_result(valid_data)

        ys = ys.reshape(-1, 1)
        yv = yv.reshape(-1, 1)

        eps = 1e-8

        # ==================================================
        # 1️⃣ support 二折 → err_support
        # ==================================================
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        idx_s1, idx_s2 = list(kf.split(xs))[0]

        xs1, xs2 = xs[idx_s1], xs[idx_s2]
        ys1, ys2 = ys[idx_s1], ys[idx_s2]

        model_s1 = Regressor()
        model_s1.fit(xs1, np.ravel(ys1))
        pred_s2 = model_s1.predict(xs2)

        model_s2 = Regressor()
        model_s2.fit(xs2, np.ravel(ys2))
        pred_s1 = model_s2.predict(xs1)

        err_s2 = np.abs(pred_s2 - np.ravel(ys2)) / (np.abs(np.ravel(ys2)) + eps)
        err_s1 = np.abs(pred_s1 - np.ravel(ys1)) / (np.abs(np.ravel(ys1)) + eps)

        err_support = np.concatenate([err_s2, err_s1])

        # ==================================================
        # 2️⃣ support 全量 → valid → err_valid
        # ==================================================
        model_support_full = Regressor()
        model_support_full.fit(xs, np.ravel(ys))

        pred_valid = model_support_full.predict(xv)

        err_valid = np.abs(pred_valid - np.ravel(yv)) / (np.abs(np.ravel(yv)) + eps)

        if np.mean(err_valid)<np.mean(err_support):
            return True
        # ==================================================
        # 长度扩充到 256
        # ==================================================
        def expand_to_256(arr):
            if len(arr) < 256:
                w = math.ceil(256 / len(arr))
                arr = np.tile(arr, w)
            return arr[:256]

        err_support = expand_to_256(err_support)
        err_valid = expand_to_256(err_valid)

        # ==================================================
        # 平滑
        # ==================================================
        if smooth:
            err_support = gaussian_filter1d(err_support, sigma)
            err_valid = gaussian_filter1d(err_valid, sigma)

        # ==================================================
        # ADWIN检测
        # ==================================================
        adwin = ADWIN()

        drift = False

        # 先输入 support
        for e in err_support:
            adwin.add_element(e)

        # 再输入 valid
        for e in err_valid:
            adwin.add_element(e)
            if adwin.detected_change():
                drift = True
                break

        # ==================================================
        # 判定
        # ==================================================
        if not drift:
            return True   # 分布相似
        else:
            return False  # 分布不同

    return None
