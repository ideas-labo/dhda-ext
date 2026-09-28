import copy
import math
import os.path
import time
from sklearn.tree import _tree
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_absolute_percentage_error
import configs
from skmultiflow import drift_detection
from base_model.basemodels import build_incremental_model
import logging
import ruptures as rpt
from new_dhda.test.test_similarity import compute_similarity_by_ML, similarity_compute_light


def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y


def combine_x_y(x, y):
    return np.hstack((x, y))


def combine_data(old, new):
    return np.vstack((old, new))


def split_data(data, rate):
    pass


def smo_data(x, ylabel):
    smo = SMOTE(random_state=1, k_neighbors=3)
    return smo.fit_resample(x, ylabel)


def get_tree_info(tree, d):
    node_count = 2 ** d - 1
    if d == 2:
        left_node = tree.left_child[0]
        right_node = tree.right_child[0]
        left_feature = tree.left_feature[left_node]
        right_feature = tree.right_feature[right_node]
        feature = [tree.feature[0], tree.feature[left_feature], tree.feature[right_feature]]
        threshold = [tree.threshold[0], tree.threshold[left_feature], tree.threshold[right_feature]]
    else:
        # warning: unsupported d != 1 or 2
        feature = [tree.feature[0]]
        threshold = [tree.threshold[0]]
    return feature, threshold


def info_compare(old, new):
    """
    :param old:
    :param new:
    :return: if old and new are the same
    """
    result = [old[0].__eq__(new[0]), old[1].__eq__(new[1])]
    return result  # todo:bug


def clusters_compare(old, new):
    similarity_list = []
    for index, cluster in enumerate(new):
        new_set = set(cluster)
        old_set = set(old[index])
        same_count = len(new_set.intersection(old_set))
        similarity_list.append(same_count / len(new_set))
    return similarity_list


def cart_change_test(G_best, G_second, n, DELTA):
    hoeffding_bound = math.sqrt((math.log(1.0 / DELTA)) / (2.0 * n))
    return G_best - G_second > hoeffding_bound


def my_cart_change_test(G_best, G_second, n1, n2, DELTA):
    my_hoeffding_bound = math.sqrt(math.log(1.0 / DELTA) / (4.0 / (1.0 / n1 + 1.0 / n2)))
    # hoeffding_bound = math.sqrt((math.log(1.0 / DELTA)) / (4.0 * (2.0 / (1.0 / n1 + 1.0 / n2))))
    return G_best - G_second > my_hoeffding_bound


def compute_hoeffding_bound(G_best, G_second, range_val, confidence, n):
    hoeffding_bound = np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))
    return G_best - G_second > hoeffding_bound

def divide_by_root(tree_, X, samples=None):
    """
    仅使用决策树根节点的第一条规则，
    将 X（子集）划分为两个簇，并返回原始样本 index

    Parameters
    ----------
    tree_ : sklearn.tree._tree.Tree
        已训练决策树的 tree_ 对象
    X : ndarray, shape (k, n_features)
        子集样本的特征矩阵
    samples : list or array-like, length k
        子集样本在全量数据中的原始 index，与 X 行一一对应

    Returns
    -------
    left_idx : list
        满足规则的原始样本 index
    right_idx : list
        不满足规则的原始样本 index
    """

    node = 0  # 根节点

    # 根节点不可分裂，也强制返回两个簇
    if tree_.feature[node] == _tree.TREE_UNDEFINED:
        return list(samples), []

    feature = tree_.feature[node]
    threshold = tree_.threshold[node]

    left_idx = []
    right_idx = []

    # 注意：这里的 i 是 X 的行号，而不是全量 index
    for i in range(len(samples)):
        if X[i, feature] <= threshold:
            left_idx.append(samples[i])
        else:
            right_idx.append(samples[i])

    return [left_idx, right_idx]


"""
1. 数据不足等待数据，足则直接训练，在检测到下次drift之前保持静止 -> 
2. 立即训练，随着到达的新数据增量训练 MLP -> 
3. 立即训练，随着新数据累计到达重新训练 MLP n ->  
4. 保存两个模型，长期模型和当前模型 
"""

"""
第一级是一个current data的总集合
第二级是各个cluster子集
1-》2: clusters change使得第二级变化
2-》1: local current data收缩导致第一级变化
第一级没有主动的遗忘策略，依赖第二级遗忘
数据必须遗忘，否则总体数据无法反应当前的数据
模型可以保留，作为老数据（老概念）的抽象记忆
"""
baseline_models = ['LR', 'HT', 'KNN', 'RF', 'ARF', 'SRP', 'SVR', 'KRR']


class DaL_Regressor:
    def __init__(self, depth=1, min_clock=32, MODEL_REUSE=False, base_model="RF", Alignment_frequency=3, min_sample = 8,
                 compare_method_1='ADWIN', compare_method_2='ADWIN', CPD=True, HYBRID_UPDATE=True, Upper_Adapt=True,
                 Lower_Adapt=True, Smooth=False, ignore_mode1=False, mape=False):
        self.base_model = base_model
        self.test_mode = True
        self.MODEL_REUSE = MODEL_REUSE
        """self.CART_FILTER = CART_FILTER
        self.CART_CHANGE = CART_CHANGE"""
        self.Upper_Adapt = Upper_Adapt
        self.Lower_Adapt = Lower_Adapt
        self.HYBRID_UPDATE = HYBRID_UPDATE
        # self.CHANGE_FILTER = CHANGE_FILTER
        self.stable_represent = False
        self.init = True
        self.models = list()
        # self.detectors = list()
        self.CART = None
        self.rfc = None
        self.clusters_index = list()
        self.min_sample = min_sample
        self.config = dict()
        self.lr = 0.001
        # self.clusters = list()
        self.weights = None
        self.depth = depth
        self.clusters_num = 0
        self.feature_count = 0
        self.max_X = 0
        self.max_Y = 0
        self.current_data_index = None
        self.whole_data = None
        self.clock = 0
        self.relative_clock = 0
        self.min_clock = min_clock
        self.detected_change_point_dict = dict()
        self.detected_cart_change_point_list = list()
        self.learn_threshold = configs.LEARNING_THRESHOLD
        self.recent_error = list()
        self.mean_error = list()
        self.count_test = list()
        self.last_data_index = list()  # record the tail item of the last change window

        self.warning_detectors = []
        self.drift_detectors = []
        self.modes = []
        self.data_size = []
        # self.model_manager = ModelsManager()
        self.data_manager = DataManager(compare_method_1, compare_method_2, smooth=Smooth, ignore_mode1=ignore_mode1, mape=mape)
        self.Smooth = Smooth
        self.retrain_clock = Alignment_frequency * min_clock

        self.all_error = [[], []]
        self.reuse_stat = [False, False]
        self.CPD = CPD
        self.mape = mape
        self.time_cost_record = {"reuse": 0, "update": 0}
        if CPD:
            self.CPD_MODEL = rpt.BottomUp(model='l1', jump=1, min_size=2)
        if self.MODEL_REUSE:
            if self.Smooth:
                self.model_name = 'DHDA-Reuse-Smooth'
            else:
                self.model_name = 'DHDA-Reuse-Raw'
        else:
            if self.Smooth:
                self.model_name = 'DHDA-Smooth'
            else:
                self.model_name = 'DHDA-Raw'

    def get_weight(self, x, y):
        feature_weights = mutual_info_regression(x, y, random_state=0)
        self.weights = feature_weights
        return feature_weights.tolist()

    def divide_cluster(self, x, y):
        DT = DecisionTreeRegressor(random_state=3, criterion='squared_error', splitter='best')
        DT.fit(x, y)
        self.CART = DT
        tree_ = DT.tree_
        cluster_indexes_all = divide_by_root(tree_, x, range(len(x)))
        return cluster_indexes_all

    def fit_RFRs(self, clustersx, clustersy):
        self.models = list()
        for i in range(len(self.clusters_index)):
            # print('Training DNN for division {}... ({} samples)'.format(i + 1, len(clustersx[i])))
            model = build_incremental_model(self.base_model)
            model.fit(clustersx[i], clustersy[i])
            self.models.append(model)

    def fit_basemodels(self, clustersx, clustersy):
        self.models = list()
        for i in range(len(self.clusters_index)):
            # print('Training DNN for division {}... ({} samples)'.format(i + 1, len(clustersx[i])))
            model = build_incremental_model(self.base_model, model_reuse=self.MODEL_REUSE)
            model.fit(clustersx[i], clustersy[i])
            self.models.append(model)

    def init_detectors(self):
        for i in range(len(self.models)):
            """detector = drift_detection.ADWIN(delta=configs.DELTA)
            self.detectors.append(detector)"""
            warning_detector = drift_detection.ADWIN(0.01)
            warning_detector.set_clock(5)
            warning_detector.mint_min_window_length = 3
            drift_detector = drift_detection.ADWIN(0.005)
            drift_detector.set_clock(8)
            drift_detector.mint_min_window_length = 3
            self.warning_detectors.append(warning_detector)
            self.drift_detectors.append(drift_detector)
            self.modes.append(False)
            self.data_size.append(0)

    def fit_rfc(self, x, y):
        rfc = RandomForestClassifier(criterion='gini', random_state=3)
        self.rfc = rfc
        if len(x) != len(y):
            print()
        self.rfc.fit(x, y)
        return

    def fit_model(self, x, y):  # remove smote and weight
        self.whole_data = combine_x_y(x, y)
        self.current_data_index = list(range(len(self.whole_data)))
        self.clock = len(y)
        x = np.array(x)
        (self.data_count, self.feature_count) = x.shape
        y = np.array(y)
        # weights = self.get_weight(x, y)
        self.clusters_index = self.divide_cluster(x, y)
        y_label = list()
        for i in range(len(x)):
            for label, cluster in enumerate(self.clusters_index):
                if i in cluster:
                    y_label.append(label)
                    continue

        if len(self.clusters_index) == 2:
            if len(self.clusters_index[0]) > 5 and len(self.clusters_index[1]) > 5:
                rfc_x, rfc_y = smo_data(x, y_label)
            else:
                rfc_x = x
                rfc_y = y_label
        else:
            rfc_x = x
            rfc_y = y_label
        self.fit_rfc(rfc_x, rfc_y)
        # dividing
        train_x = list()
        train_y = list()
        for c in self.clusters_index:
            train_data_temp = self.whole_data[c]
            x_temp, y_temp = get_attributes_result(train_data_temp)
            train_x.append(x_temp)
            train_y.append(y_temp)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        if self.base_model in baseline_models:
            self.fit_basemodels(train_x, train_y)
        else:
            print("err base model")
            exit()
        self.init_detectors()
        for i in range(len(self.clusters_index)):
            self.recent_error.append(list())
            self.mean_error.append(0)
            self.count_test.append(0)

    def predict(self, x):
        # x_test = self.predict_data_preprocessing(x)
        # get the cluster index of test data
        # weighted_x = np.array(x_test) * self.weights
        test_labels = self.rfc.predict(x)
        # logging.info(f"predict local model index: {test_labels}")
        # 以下是按照x顺序进行predict
        pre_y = []
        for i, label in enumerate(test_labels):
            # print(f"pre in cluster{test_labels}")
            y = self.models[label].predict(x[i][np.newaxis, :])
            pre_y.append(y[0])
        return pre_y

    def learn_data(self, x, y, val):  #维护 whole_data current_data clusters_index
        # learn when new data are available
        new_data = combine_x_y(x, y)
        self.whole_data = combine_data(self.whole_data, new_data)

        for i, data in enumerate(x):
            self.clock += 1
            self.relative_clock += 1
            new_single_data_index = len(self.whole_data) - 1
            self.current_data_index.append(new_single_data_index)
            cluster_index = self.rfc.predict(data[np.newaxis, :])[0]
            self.clusters_index[cluster_index].append(new_single_data_index)
            self.all_error[cluster_index].append(val)
            # self.clusters_index[cluster_index].append(len(self.whole_data) - 1)
            self.recent_error[cluster_index].append(val)
            n_count = self.count_test[cluster_index]
            self.count_test[cluster_index] += 1
            self.mean_error[cluster_index] = (self.mean_error[cluster_index] * n_count + val) / (n_count + 1)
            if len(self.recent_error[cluster_index]) > 32:
                self.recent_error[cluster_index].pop(0)
            sq_val = val
            if self.Lower_Adapt:
                if self.modes[cluster_index]:
                    self.data_size[cluster_index] += 1
                for times in range(2):
                    self.warning_detectors[cluster_index].add_element(sq_val)
                    if self.warning_detectors[cluster_index].detected_change():
                        print(f"warning happen in index{cluster_index} at index{self.clock} at{self.clock / 32}")
                        logging.info(
                            f"a warning is detected in cluster {cluster_index} at global clock {self.relative_clock - 1} now")
                        self.modes[cluster_index] = True
                        self.data_size[cluster_index] = 0
                        self.warning_detectors[cluster_index].reset()

                    self.drift_detectors[cluster_index].add_element(sq_val)

                    if self.drift_detectors[cluster_index].detected_change():
                        mean_recent = np.mean(self.recent_error[cluster_index])
                        mean_old = self.mean_error[cluster_index]

                        if mean_recent < mean_old:  # 向下漂移 忽略
                            self.modes[cluster_index] = False
                            self.drift_detectors[cluster_index].reset()
                            self.warning_detectors[cluster_index].reset()
                            for val in self.recent_error[cluster_index]:
                                self.warning_detectors[cluster_index].add_element(val)
                                self.drift_detectors[cluster_index].add_element(val)
                                self.warning_detectors[cluster_index].add_element(val)
                                self.drift_detectors[cluster_index].add_element(val)
                            print(
                                f"drift happen in index{cluster_index} at index{self.clock} at{self.clock / 32} but ignore")
                            logging.info(
                                f"a drift is detected but ignore in cluster {cluster_index} at global clock {self.relative_clock - 1} now, mean old error = {mean_old}, recent error = {mean_recent}, count = {self.count_test[cluster_index]}")
                        else:
                            print(f"drift happen in index{cluster_index} at index{self.clock} at{self.clock / 32}")
                            logging.info(
                                f"a drift is detected in cluster {cluster_index} at global clock {self.relative_clock - 1} now, mean old error = {mean_old}, recent error = {mean_recent}, count = {self.count_test[cluster_index]}")
                            # logging.info(f"the error list {self.all_error[cluster_index][-64:]}")
                            self.modes[cluster_index] = False
                            self.drift_detectors[cluster_index].reset()
                            self.warning_detectors[cluster_index].reset()

                            if cluster_index not in self.detected_change_point_dict:  # record the change point detection result
                                self.detected_change_point_dict[cluster_index] = list()
                            self.detected_change_point_dict[cluster_index].append(self.clock)

                            whole_size = len(self.clusters_index[cluster_index])
                            current_window_size = self.data_size[cluster_index]
                            self.data_size[cluster_index] = 0

                            """if current_window_size < configs.MIN_CLUSTERS_SAMPLES:
                                # current_window_size = configs.MIN_CLUSTERS_SAMPLES
                                current_window_size = 0
                                errors = np.array(self.all_error[cluster_index])[-64:]
                                self.all_error[cluster_index] = []
                                self.CPD_MODEL.fit(errors)
                                cp = self.CPD_MODEL.predict(n_bkps=1)
                                rpt.display(errors, cp)
                                # plt.show()
                                plt.savefig(os.path.join(configs.CPD_FIGURE_PATH, f"clock_{self.relative_clock}_{time.time()}.png"))
                                plt.close()
                                current_window_size = len(errors) - cp[0]
                                if current_window_size < configs.MIN_CLUSTERS_SAMPLES:
                                    current_window_size = configs.MIN_CLUSTERS_SAMPLES
                            if current_window_size > configs.MAX_CLUSTERS_SAMPLES:
                                current_window_size = configs.MAX_CLUSTERS_SAMPLES"""
                            if self.CPD and self.MODEL_REUSE:
                                errors = np.array(self.all_error[cluster_index])[-64:]
                                self.all_error[cluster_index] = []
                                self.CPD_MODEL.fit(errors)
                                cp = self.CPD_MODEL.predict(n_bkps=1)
                                rpt.display(errors, cp)
                                # plt.show()
                                current_window_size = len(errors) - cp[0] - 1


                            if current_window_size < self.min_sample:
                                current_window_size = self.min_sample
                            if current_window_size > configs.MAX_CLUSTERS_SAMPLES:
                                current_window_size = configs.MAX_CLUSTERS_SAMPLES

                            delete_data_size = whole_size - current_window_size
                            delete_data = self.clusters_index[cluster_index][:delete_data_size]
                            self.mean_error[cluster_index] = 0
                            self.count_test[cluster_index] = 0
                            # self.recent_error[cluster_index] = []
                            # print(f"contain data count = {len(self.clusters_index[cluster_index])}")
                            # print(f"delete old data count = {delete_data_size}, start index = {delete_data[0]}({delete_data[0] / 32}), end index = {delete_data[-1]}({delete_data[-1] / 32})")
                            remain_data = self.clusters_index[cluster_index][-current_window_size:]
                            # update cluster index
                            self.clusters_index[cluster_index] = remain_data
                            # self.current_data_index =  [item for item in self.current_data_index if item not in delete_data]
                            # update current data

                            old_data = delete_data[int(delete_data_size * 0.05):int(
                                delete_data_size * 0.8)]  # change point detection
                            for item in delete_data:
                                while item in self.current_data_index:
                                    self.current_data_index.remove(item)

                            # retrain local model
                            # dnn_x, dnn_y = self.single_DNN_data_preprocessing(new_x, new_y)

                            if self.MODEL_REUSE and self.HYBRID_UPDATE:
                                time0 = time.time()
                                if len(remain_data) == 0:
                                    print()
                                reuse_data_index = self.data_manager.select_one(self.whole_data, remain_data)
                                self.data_manager.add_concept(self.whole_data, old_data)
                                if reuse_data_index is not None:
                                    print("reuse successfully")
                                    logging.info(f"reuse successfully")
                                    # todo: add code
                                    self.clusters_index[cluster_index].extend(reuse_data_index)
                                    self.current_data_index.extend(reuse_data_index)
                                    train_x, train_y = get_attributes_result(
                                        self.whole_data[self.clusters_index[cluster_index]])
                                    self.models[cluster_index].fit(train_x, train_y)
                                    self.reuse_stat[cluster_index] = True
                                else:
                                    print(f"reuse unsuccessfully")
                                    self.reuse_stat[cluster_index] = False
                                    train_x, train_y = get_attributes_result(
                                        self.whole_data[self.clusters_index[cluster_index]])
                                    self.models[cluster_index].fit(train_x, train_y)
                                time1 = time.time()
                                self.time_cost_record['reuse'] += time1 - time0

                            else:  # when model reuse is not allowed:

                                train_x, train_y = get_attributes_result(
                                    self.whole_data[self.clusters_index[cluster_index]])
                                self.models[cluster_index].fit(train_x, train_y)  # todo: check if change
                            print(f"warning window size: {len(train_y)}")
                            logging.info(f"warning window size: {len(train_y)}")
                            logging.info(f"cluster info:{self.clusters_index}")
                            logging.info(f"the history info:\n {self.data_manager.get_info()}")
                            # model = self.model_manager.get_model_by_accuracy_threshold(train_x, train_y, cluster_index)

                            # self.retrain_model_rfr(cluster_index, train_x, train_y)  # change
            else:  #use most resent samples
                self.current_data_index = self.current_data_index[-256:]

            if self.clock % self.min_clock == 0:
                time0 = time.time()
                current_x, current_y = get_attributes_result(self.whole_data[self.current_data_index])
                current_DT = DecisionTreeRegressor(random_state=3, criterion='squared_error', splitter='best')
                current_DT.fit(current_x, current_y)
                current_tree_info = get_tree_info(current_DT.tree_, 1)
                old_tree_info = get_tree_info(self.CART.tree_, 1)

                # 如果当前误差不够小 or 数据少->需要学习新数据
                if np.mean(self.recent_error[cluster_index]) >= self.learn_threshold or len(
                        self.clusters_index[cluster_index]) < 128:
                    if self.HYBRID_UPDATE:
                        clusters_index = list()
                        clusters_index = divide_by_root(current_DT.tree_, current_x, self.current_data_index)
                        self.clusters_index = clusters_index
                        if len(self.clusters_index[0]) == 0 or len(self.clusters_index[1]) == 0:
                            print()
                    y_label = list()
                    if self.Upper_Adapt:
                        for index in self.current_data_index:
                            for label, cluster in enumerate(self.clusters_index):
                                if index in cluster:
                                    y_label.append(label)
                                    break
                        self.fit_rfc(current_x, y_label)
                    # todo: 有bug reuse会导致某一数据在两个簇中都存在 continue-》break后bug不在了但是似乎不符合dal
                    train_x = list()
                    train_y = list()
                    for c in self.clusters_index:
                        train_data_temp = self.whole_data[c]
                        x_temp, y_temp = get_attributes_result(train_data_temp)
                        train_x.append(x_temp)
                        train_y.append(y_temp)

                    if self.HYBRID_UPDATE:  # 下层更新
                        if self.clock % self.retrain_clock == 0:
                            # logging.info(f"model retrain update")
                            # self.fit_RFRs(train_x, train_y)
                            self.models = list()
                            for mi in range(2):
                                model = build_incremental_model(regressor=self.base_model,
                                                                model_reuse=self.MODEL_REUSE)
                                model.fit(train_x[mi], train_y[mi])
                                self.models.append(model)
                        else:
                            # logging.info(f"model incremental update")
                            for index, xs in enumerate(train_x):
                                new_x = xs[-self.min_clock:]
                                new_y = train_y[index][-self.min_clock:]
                                self.models[index].update(new_x, new_y)
                time1 = time.time()
                self.time_cost_record['update'] += time1 - time0
                # whether cart change happens
                cart_result = info_compare(old_tree_info, current_tree_info)
                if (not (cart_result[0] and cart_result[1])) and (self.reuse_stat[0] == self.reuse_stat[1]):
                    isChange = True
                    old_best_feature = self.CART.tree_.feature[0]
                    new_best_feature = current_DT.tree_.feature[0]
                    G_best = current_DT.feature_importances_[new_best_feature]
                    G_second = current_DT.feature_importances_[old_best_feature]
                    if len(self.clusters_index) == 2:
                        if (len(self.clusters_index[0]) / len(self.clusters_index[1]) > 3 or
                                len(self.clusters_index[1]) / len(self.clusters_index[0]) > 3):
                            isChange = False

                        if not my_cart_change_test(G_best, G_second, len(self.clusters_index[0]),
                                                   len(self.clusters_index[1]), 0.0000001):
                            isChange = False
                    if isChange and self.Upper_Adapt:
                        print(
                            f"------------------------------CART change detected in {self.clock}------------------------------")
                        logging.info(
                            f"CART change detected now, the old CART feature: {old_best_feature}, the new CART feature: {new_best_feature}")
                        # continue
                        print(f"the old CART feature: {old_best_feature}, with weight{G_second}")
                        print(f"the new CART feature: {new_best_feature}, with weight{G_best}")
                        print("the old clusters info:")
                        for index in range(len(self.clusters_index)):
                            print(
                                f"the {index}-th cluster contain samples count {len(self.clusters_index[index])}, from {self.clusters_index[index][0]} to {self.clusters_index[index][-1]}")
                        self.detected_cart_change_point_list.append(self.clock)
                        # self.clusters_index = self.divide_cluster(current_x, current_y) # buggggggggg!!!!
                        self.CART = current_DT
                        clusters_index = divide_by_root(current_DT.tree_, current_x, self.current_data_index)

                        self.clusters_index = clusters_index
                        print("the new clusters info:")
                        for index in range(len(self.clusters_index)):
                            print(
                                f"the {index}-th cluster contain samples count {len(self.clusters_index[index])}, from {self.clusters_index[index][0]} to {self.clusters_index[index][-1]}")
                        """for index, num in enumerate(similar_list):
                            print(f"the {index}-th old&new clusters has {num} percent of the same data.")"""
                        y_label = list()

                        # in order to keep window size and data consistent that may be breached when cart change
                        """for index, d in enumerate(self.detectors):
                            d.reset()
                            for data_index in self.clusters_index[index]:
                                d.add_element(self.val_list[data_index])
                            window_size = d.width
                            delete_size = len(self.clusters_index[index]) - window_size
                            delete_data = self.clusters_index[index][:delete_size]
                            self.clusters_index[index] = self.clusters_index[index][-window_size:]

                            for item in delete_data:
                                while item in self.current_data_index:
                                    self.current_data_index.remove(item)"""

                        """for index, d in enumerate(self.warning_detectors):
                            d.reset()
                        for index, d in enumerate(self.drift_detectors):
                            d.reset()"""
                        # label attrs with cluster index
                        for index in self.current_data_index:
                            for label, cluster in enumerate(self.clusters_index):
                                if index in cluster:
                                    y_label.append(label)
                                    continue
                        self.fit_rfc(current_x, y_label)

                        # dnn_x, dnn_y = self.DNNs_data_preprocessing_online(current_x, current_y)
                        # self.fit_DNNs(dnn_x, dnn_y) change
                        train_x = list()
                        train_y = list()
                        # get new clusters
                        for c in self.clusters_index:
                            train_data_temp = self.whole_data[c]
                            x_temp, y_temp = get_attributes_result(train_data_temp)
                            train_x.append(x_temp)
                            train_y.append(y_temp)
                        self.fit_basemodels(train_x, train_y)

    def retrain_model_rfr(self, index, x, y):
        # print('Training DNN for division {}... ({} samples)'.format(index, len(x)))
        model = build_incremental_model(self.base_model)
        model.fit(x, y)
        self.models[index] = model  # 替换old model
        # self.detectors[index].reset()

    def retrain_all(self, x, y):
        pass


class ModelsManager:
    def __init__(self):
        self.current_cart_index = 0
        self.carts_list = []
        self.models = []
        self.similar_threshold = 0.02

    def cart_info_exit(self, cart_info):
        for cart in self.carts_list:
            result = info_compare(cart, cart_info)
            if result[0] and result[1]:
                return True
        return False

    def get_cart_index(self, cart_info):
        for index, cart in enumerate(self.carts_list):
            result = info_compare(cart, cart_info)
            if result[0] and result[1]:
                return index
        return None

    def change_to_cart(self, cart_info):
        if not self.cart_info_exit(cart_info):
            self.carts_list.append(cart_info)
            self.models.append([[], []])
            self.current_cart_index = len(self.carts_list) - 1
        else:
            self.current_cart_index = self.get_cart_index(cart_info)

    def add_model_in_current_cart(self, model, index):
        self.models[self.current_cart_index][index].append(model)
        """if self.has_similar_model(model):
            # todo: replace it or discard
            pass
        else:
            # todo: add it in history pool
            self.models[self.current_cart_index][index].append(model)"""

    def has_similar_model(self, model):
        pass

    def get_model_by_accuracy(self, x, y, cluster_index):
        if len(self.models[self.current_cart_index][cluster_index]) == 0:
            return None
        err_list = []
        y = np.array(y)
        for model in self.models[self.current_cart_index][cluster_index]:
            pre_y = []
            for i, config in enumerate(x):
                y_temp = model.predict(x[i][np.newaxis, :])
                pre_y.append(y_temp[0])
            pre_y = np.array(pre_y)
            err = mean_absolute_percentage_error(y_true=y, y_pred=pre_y)
            err_list.append(err)
        min_err = 100000
        min_index = -1
        for index, err in enumerate(err_list):
            if err < min_err:
                min_err = err
                min_index = index
        # self.models[self.current_cart_index][index].append(self.models[self.current_cart_index][index][max_index])
        if min_err > 0.06:
            return None
        return copy.deepcopy(self.models[self.current_cart_index][cluster_index][min_index])

    def get_model_by_accuracy_threshold(self, x, y, cluster_index):
        if len(self.models[self.current_cart_index][cluster_index]) == 0:
            return None
        err_list = []
        y = np.array(y)
        for model in self.models[self.current_cart_index][cluster_index]:
            pre_y = []
            for i, config in enumerate(x):
                y = model.predict(x[i][np.newaxis, :])
                pre_y.append(y[0])
            pre_y = np.array(pre_y)
            err = np.sum(abs(y - pre_y))
            err_list.append(err)
        min_err = 0
        min_index = -1
        for index, err in enumerate(err_list):
            if err < min_err:
                min_err = err
                min_index = index
        # self.models[self.current_cart_index][index].append(self.models[self.current_cart_index][index][max_index])
        if min_err < self.similar_threshold:
            return copy.deepcopy(self.models[self.current_cart_index][cluster_index][min_index])
        else:
            return None


def data_match(target: np.ndarray, compare: np.ndarray) -> dict:
    """
    比较两个ndarray数据集，统计目标数据集（target）的关键匹配指标。

    数据规则：
    - 每个数据集的最后一列视为标签（y），其余列视为特征（x）
    - 仅处理特征列数量一致的输入（避免维度不匹配导致的错误）

    参数：
        target: np.ndarray - 目标数据集（用于统计的基准数据集）
        compare: np.ndarray - 对比数据集（用于与目标数据集匹配的参照数据集）

    返回：
        dict - 包含4个统计指标的字典：
            - "target_total_count": target总数据个数
            - "same_x_diff_y_count": target中与compare特征相同但标签不同的个数
            - "same_x_same_y_count": target中与compare特征相同且标签相同的个数
            - "x_not_in_compare_count": target中特征在compare中完全不存在的个数
    """
    # -------------------------- 1. 数据合法性校验与拆分 --------------------------
    # 校验输入是否为ndarray
    if not isinstance(target, np.ndarray) or not isinstance(compare, np.ndarray):
        raise TypeError("target和compare必须是numpy.ndarray类型")

    # 校验数据集是否为空
    if target.size == 0:
        raise ValueError("target数据集不能为空")
    if compare.size == 0:
        raise ValueError("compare数据集不能为空")

    # 校验特征列数量一致（最后一列为y，剩余列为x，故x的维度 = 总列数 - 1）
    target_x_dim = target.shape[1] - 1
    compare_x_dim = compare.shape[1] - 1
    if target_x_dim != compare_x_dim:
        raise ValueError(
            f"两个数据集的特征列数量不匹配："
            f"target的特征列数为{target_x_dim}，compare的特征列数为{compare_x_dim}"
        )

    # 拆分x（特征）和y（标签）
    target_x = target[:, :-1]  # target的所有行，除最后一列外的所有列
    target_y = target[:, -1]  # target的所有行，仅最后一列
    compare_x = compare[:, :-1]
    compare_y = compare[:, -1]

    # -------------------------- 2. 核心指标统计 --------------------------
    # 指标1：target总数据个数
    target_total = target.shape[0]

    # 关键步骤：找到target_x中每个样本在compare_x中的匹配索引
    # 原理：通过np.equal比较每行，再用np.all确认所有特征值匹配，最后获取匹配的行索引
    match_mask = np.all(np.equal(target_x[:, None, :], compare_x[None, :, :]), axis=-1)
    # match_mask[i][j] = True 表示 target_x[i] 与 compare_x[j] 完全匹配

    # 指标2：target中x与compare相同但y不同的个数
    same_x_diff_y = 0
    # 指标3：target中x与compare相同且y相同的个数
    same_x_same_y = 0

    for i in range(target_total):
        # 获取当前target样本在compare中的所有匹配行索引
        compare_match_indices = np.where(match_mask[i])[0]
        if len(compare_match_indices) == 0:
            continue  # 无匹配，跳过

        # 检查是否存在至少一个匹配的y值不同
        has_diff_y = any(target_y[i] != compare_y[j] for j in compare_match_indices)
        # 检查是否存在至少一个匹配的y值相同
        has_same_y = any(target_y[i] == compare_y[j] for j in compare_match_indices)

        # 统计规则：若存在任一匹配行y不同，则计入same_x_diff_y；若存在任一匹配行y相同，则计入same_x_same_y
        if has_diff_y:
            same_x_diff_y += 1
        if has_same_y:
            same_x_same_y += 1

    # 指标4：target中的x在compare中不存在的个数（无任何匹配的样本数）
    x_not_in_compare = target_total - (same_x_diff_y + same_x_same_y)

    # -------------------------- 3. 返回结果 --------------------------
    return {
        "target_total_count": target_total,
        "same_x_diff_y_count": same_x_diff_y,
        "same_x_same_y_count": same_x_same_y,
        "x_not_in_compare_count": x_not_in_compare
    }


class DataManager:
    def __init__(self, method1, method2, smooth=True, ignore_mode1=False, f=False, mape=False):
        self.concepts = []
        self.method1 = method1
        self.method2 = method2
        self.smooth = smooth
        self.ignore_mode1 = ignore_mode1
        self.f = f
        self.topn = 3
        self.mape = mape

    def get_info(self):
        info_str = ''
        for i, c in enumerate(self.concepts):
            info_str += f"the {i} concept: from {min(c)} to {max(c)}, count {len(c)}\n"

        return info_str

    def select_one(self, whole_data, data, observation='APE', topn=3):
        topn_concepts = None
        mape_list = []
        concept_model_list = []

        test_x, test_y = get_attributes_result(whole_data[data])

        # 训练每个concept对应的模型
        for index, concept in enumerate(self.concepts):
            rf = RandomForestRegressor()
            x, y = get_attributes_result(whole_data[concept])
            rf.fit(x, y)
            concept_model_list.append((index, rf))  # 同时存index

        # 计算每个模型的MAPE
        for index, model in concept_model_list:
            pre = model.predict(test_x)
            mape = np.mean(np.abs((pre - test_y) / (np.abs(test_y) + 1e-8)))  # 标准MAPE
            mape_list.append((index, mape))

        # 按MAPE升序排序
        mape_list.sort(key=lambda x: x[1])

        if self.mape:
            if len(self.concepts) == 0:
                return None
            return self.concepts[mape_list[0][0]]
        # 取最小的n个
        topn_tri = mape_list[:topn]

        # 取对应index
        topn_concepts = [item[0] for item in topn_tri]

        candidate_index = []
        for index in topn_concepts:
            concept = self.concepts[index]
            if similarity_compute_light(whole_data[concept], whole_data[data], 1, smooth=self.smooth):
                candidate_index.append(index)

        if len(candidate_index)==0:
            return None
        logging.info(f"reuse successfully! the {candidate_index[0]} concept is selected to resue")
        print(f"reuse successfully! the {candidate_index[0]} concept is selected to resue")

        return self.concepts[candidate_index[0]]

    def add_concept(self, whole_data, data, observation='APE', topn=3):
        if self.ignore_mode1:
            self.concepts.append(data)
            return None

        if len(data) < 32:
            return None  # todo: filter when add new concept

        mape_list = []

        train_x, train_y = get_attributes_result(whole_data[data])
        model = RandomForestRegressor()
        model.fit(train_x, train_y)

        for index, concept in enumerate(self.concepts):
            test_x, test_y = get_attributes_result(whole_data[concept])
            pre = model.predict(test_x)
            mape = np.mean(np.abs((pre - test_y) / (np.abs(test_y) + 1e-8)))
            mape_list.append((index, mape))

        mape_list.sort(key=lambda x: x[1])

        # Ablation: when self.mape=True, skip similarity check
        # and directly merge with the concept having the lowest MAPE.
        if self.mape:
            if len(mape_list) == 0:
                self.concepts.append(data)
            else:
                best_index = mape_list[0][0]
                logging.info(
                    f"(MAPE Ablation) old concept was combined with concept {best_index}, "
                    f"all concept error list: {mape_list}"
                )
                combined = np.concatenate([self.concepts[best_index], data])
                self.concepts[best_index] = np.unique(combined, axis=0)
            return None

        topn_tri = mape_list[:topn]
        topn_concepts = [item[0] for item in topn_tri]

        candidate_index = []
        for index in topn_concepts:
            if similarity_compute_light(
                    whole_data[self.concepts[index]],
                    whole_data[data],
                    0,
                    smooth=self.smooth
            ):
                candidate_index.append(index)

        if len(candidate_index) == 0:
            if len(data) > 32:
                self.concepts.append(data)
        else:
            logging.info(
                f"old concept was combined with concept {candidate_index[0]}, "
                f"all concept error list: {mape_list}"
            )
            combined = np.concatenate([self.concepts[candidate_index[0]], data])
            self.concepts[candidate_index[0]] = np.unique(combined, axis=0)

        return None


class HistoryModel:
    def __init__(self):
        self.model = None
        self.mean = 0
        self.val = 0
