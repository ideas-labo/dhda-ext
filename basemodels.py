from abc import ABC, abstractmethod
from copy import deepcopy
from sklearn.model_selection import train_test_split
# from river.neighbors import KNNRegressor
from skmultiflow.lazy import KNNRegressor
import numpy as np
from river.linear_model import LinearRegression
from river.tree import HoeffdingTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from utils.Meta_sparse_model_tf2 import MTLSparseModel
from utils.hyperparameter_tuning_MTSM import nn_l1_val
from utils.mlp_plain_model_tf2 import MLPPlainModel
from skmultiflow.meta import AdaptiveRandomForestRegressor
from river.ensemble import SRPRegressor as SRP
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import pandas as pd




def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y


def hyper_tune(X_train, Y_train):
    X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X_train, Y_train, test_size=0.333)
    (N, n) = X_train.shape
    print('Tuning hyperparameters...')
    print('Step 1: Tuning the number of layers and the learning rate ...')
    config = dict()
    config['num_input'] = n
    config['num_neuron'] = 128
    config['lambda'] = 'NA'
    config['decay'] = 'NA'
    config['verbose'] = 0
    abs_error_all = np.zeros((15, 4))
    abs_error_all_train = np.zeros((15, 4))
    abs_error_layer_lr = np.zeros((15, 2))
    abs_err_layer_lr_min = 100
    count = 0
    layer_range = range(2, 15)
    # lr_range = [0.0025, 0.005, 0.0075, 0.01]
    lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
    # lr_range = np.logspace(np.log10(0.0001), np.log10(0.1), 5)
    for n_layer in layer_range:
        config['num_layer'] = n_layer
        for lr_index, lr_initial in enumerate(lr_range):
            model = MLPPlainModel(config)
            model.build_train()
            model.train(X_train1, Y_train1, lr_initial)

            Y_pred_train = model.predict(X_train1)
            abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
            abs_error_all_train[int(n_layer), lr_index] = abs_error_train

            Y_pred_val = model.predict(X_train2)
            abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
            abs_error_all[int(n_layer), lr_index] = abs_error

            # Pick the learning rate that has the smallest train cost
            # Save testing abs_error correspond to the chosen learning_rate
        temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
        temp_idx = np.where(abs(temp) < 0.0001)[0]
        if len(temp_idx) > 0:
            lr_best = lr_range[np.max(temp_idx)]
            err_val_best = abs_error_all[int(n_layer), np.max(temp_idx)]
        else:
            lr_best = lr_range[np.argmin(temp)]
            err_val_best = abs_error_all[int(n_layer), np.argmin(temp)]

        abs_error_layer_lr[int(n_layer), 0] = err_val_best
        abs_error_layer_lr[int(n_layer), 1] = lr_best

        if abs_err_layer_lr_min >= abs_error_all[int(n_layer), np.argmin(temp)]:
            abs_err_layer_lr_min = abs_error_all[int(n_layer),
            np.argmin(temp)]
            count = 0
        else:
            count += 1

        if count >= 2:
            break

    abs_error_layer_lr = abs_error_layer_lr[abs_error_layer_lr[:, 1] != 0]

    # Get the optimal number of layers
    n_layer_opt = layer_range[np.argmin(abs_error_layer_lr[:, 0])] + 3

    # Find the optimal learning rate of the specific layer
    config['num_layer'] = n_layer_opt
    for lr_index, lr_initial in enumerate(lr_range):
        model = MLPPlainModel(config)
        model.build_train()
        model.train(X_train1, Y_train1, lr_initial)

        Y_pred_train = model.predict(X_train1)
        abs_error_train = np.mean(np.abs(Y_pred_train - Y_train1))
        abs_error_all_train[int(n_layer), lr_index] = abs_error_train

        Y_pred_val = model.predict(X_train2)
        abs_error = np.mean(np.abs(Y_pred_val - Y_train2))
        abs_error_all[int(n_layer), lr_index] = abs_error

    temp = abs_error_all_train[int(n_layer), :] / np.max(abs_error_all_train)
    temp_idx = np.where(abs(temp) < 0.0001)[0]
    if len(temp_idx) > 0:
        lr_best = lr_range[np.max(temp_idx)]
    else:
        lr_best = lr_range[np.argmin(temp)]

    lr_opt = lr_best
    print('The optimal number of layers: {}'.format(n_layer_opt))
    print('The optimal learning rate: {:.4f}'.format(lr_opt))

    # Use grid search to find the right value of lambda
    lambda_range = np.logspace(-2, np.log10(1000), 30)
    error_min = np.zeros((1, len(lambda_range)))
    rel_error_min = np.zeros((1, len(lambda_range)))
    decay = 'NA'
    for idx, lambd in enumerate(lambda_range):
        val_abserror, val_relerror = nn_l1_val(X_train1, Y_train1,
                                               X_train2, Y_train2,
                                               n_layer_opt, lambd, lr_opt, max_epoch=2000)
        error_min[0, idx] = val_abserror
        rel_error_min[0, idx] = val_relerror

        # Find the value of lambda that minimize error_min
    lambda_f = lambda_range[np.argmin(error_min)]
    print('Step 2: Tuning the l1 regularized hyperparameter ...')
    print('The optimal l1 regularizer: {:.4f}'.format(lambda_f))

    # Solve the final NN with the chosen lambda_f on the training data
    config = dict()
    config['num_neuron'] = 128
    config['num_input'] = n
    config['num_layer'] = n_layer_opt
    config['lambda'] = lambda_f
    config['verbose'] = 1
    return config, lr_opt




class BaseModel(ABC):
    @abstractmethod
    def update(self, x, y):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class LinearRegressor(BaseModel):
    def __init__(self):
        self.n_feature = None
        self.model = LinearRegression(intercept_lr=.1)
        self.max_y = None

    def update(self, x, y):
        if len(y) == 1:
            data_structure = {}
            for index, data in enumerate(x):
                for i, feature in enumerate(data):
                    data_structure[i] = feature
                self.model.learn_one(data_structure, y[index])
        else:
            y = np.divide(y, self.max_y)
            n_count, n_feature = x.shape
            self.n_feature = n_feature
            columns = list(range(self.n_feature))
            df = pd.DataFrame(x, columns=columns)
            y_s = np.squeeze(y)
            s = pd.Series(y_s, name='output')
            self.model.learn_many(df, s)

    def fit(self, x, y):
        self.max_y = max(y)
        self.model = LinearRegression(intercept_lr=.1)
        self.update(x, y)

    def predict(self, x):
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
        return [self.model.predict_one(x=data_structure) * self.max_y]


class KNNRegression(BaseModel):
    def __init__(self):
        self.n_feature = None
        self.model = KNNRegressor()

    def update(self, x, y):
        """n_count, self.n_feature = x.shape
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
            self.model.learn_one(data_structure, y[index])"""
        self.model.partial_fit(x, y)

    def fit(self, x, y):
        self.model = KNNRegressor()
        self.model.fit(x, y)

    def predict(self, x):
        """data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature"""
        pre_y = self.model.predict(x)
        return pre_y


class HoeffdingTree(BaseModel):
    def __init__(self):
        self.n_feature = None
        self.model = HoeffdingTreeRegressor(grace_period=100)

    def update(self, x, y):
        n_count, self.n_feature = x.shape
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
            self.model.learn_one(data_structure, y[index])

    def fit(self, x, y):
        self.model = HoeffdingTreeRegressor(grace_period=100)
        self.update(x, y)

    def predict(self, x):
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
        pre_y = self.model.predict_one(x=data_structure)
        return [pre_y]


class ECPF(BaseModel):
    """Enhanced Concept Profiling Framework for regression streams.

    This is a Python adaptation of ECPF's model-management idea. It keeps a
    pool of historical models, reuses the best model on warning-buffer data
    after drift, and trains a fresh model alongside the reused model.
    """

    STABLE = "stable"
    WARNING = "warning"
    DRIFT = "drift"

    def __init__(self, base_model_cls=HoeffdingTree, similarity_margin=0.95,
                 fade_points=15, fade_models=True, model_check_freq=1,
                 prediction_similarity_tol=0.05, min_buffer_size=1):
        self.base_model_cls = base_model_cls
        self.similarity_margin = similarity_margin
        self.fade_points = fade_points
        self.fade_models_enabled = fade_models
        self.model_check_freq = max(1, int(model_check_freq))
        self.prediction_similarity_tol = prediction_similarity_tol
        self.min_buffer_size = max(1, int(min_buffer_size))
        self.reset()

    def reset(self):
        self.classifier_collection = [self._new_base_model()]
        self.current_classifier = 0
        self.new_model = None
        self.buffer_x = []
        self.buffer_y = []
        self.model_fade_scores = {0: self.fade_points}
        self.model_error_sum = {0: 0.0}
        self.model_seen = {0: 0}
        self.model_pair_seen = {}
        self.model_pair_agree = {}
        self.curr_error_sum = 0.0
        self.new_error_sum = 0.0
        self.total_inst = 0
        self.num_drifts = 0
        self.model_reuses = 0
        self.model_merges = 0
        self.models_faded = 0

    def _new_base_model(self):
        return self.base_model_cls()

    def _copy_model(self, model):
        return deepcopy(model)

    def _as_arrays(self, x, y=None):
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if y is None:
            return x_arr, None
        y_arr = np.asarray(y).reshape(-1)
        if len(y_arr) != len(x_arr):
            raise ValueError("x and y must contain the same number of samples")
        return x_arr, y_arr

    def _predict_one(self, model, x_row):
        pred = model.predict(np.asarray(x_row).reshape(1, -1))[0]
        if pred is None:
            return 0.0
        return float(pred)

    def _model_error(self, model, x_arr, y_arr):
        if len(x_arr) == 0:
            return float("inf")
        errors = [abs(self._predict_one(model, row) - float(y_arr[i]))
                  for i, row in enumerate(x_arr)]
        return float(np.mean(errors))

    def _train_model(self, model, x_arr, y_arr):
        if len(x_arr) > 0:
            model.update(x_arr, y_arr)

    def _active_model_indices(self):
        return [i for i, model in enumerate(self.classifier_collection)
                if model is not None]

    def _add_model(self, model, fade_score=0):
        model_id = len(self.classifier_collection)
        self.classifier_collection.append(model)
        self.model_fade_scores[model_id] = fade_score
        self.model_error_sum[model_id] = 0.0
        self.model_seen[model_id] = 0
        return model_id

    def _remove_model(self, model_id):
        if model_id == self.current_classifier:
            return
        if 0 <= model_id < len(self.classifier_collection):
            self.classifier_collection[model_id] = None
        self.model_fade_scores.pop(model_id, None)
        self.model_error_sum.pop(model_id, None)
        self.model_seen.pop(model_id, None)

    def _same_prediction(self, pred_a, pred_b, y_scale):
        tolerance = self.prediction_similarity_tol * max(1.0, abs(y_scale))
        return abs(pred_a - pred_b) <= tolerance

    def _update_similarity_stats(self, current_models, x_arr, y_arr):
        if len(x_arr) == 0:
            return
        y_scale = float(np.mean(np.abs(y_arr))) if y_arr is not None else 1.0
        preds = {}
        for model_id in current_models:
            model = self.classifier_collection[model_id]
            preds[model_id] = [self._predict_one(model, row) for row in x_arr]

        for left_pos, model_a in enumerate(current_models):
            for model_b in current_models[left_pos + 1:]:
                key = tuple(sorted((model_a, model_b)))
                agree = 0
                for pred_a, pred_b in zip(preds[model_a], preds[model_b]):
                    if self._same_prediction(pred_a, pred_b, y_scale):
                        agree += 1
                self.model_pair_seen[key] = self.model_pair_seen.get(key, 0) + len(x_arr)
                self.model_pair_agree[key] = self.model_pair_agree.get(key, 0) + agree

    def _mean_historical_error(self, model_id):
        seen = self.model_seen.get(model_id, 0)
        if seen == 0:
            return float("inf")
        return self.model_error_sum.get(model_id, 0.0) / seen

    def _merge_similar_models(self, current_models):
        removed = set()
        for left_pos, model_a in enumerate(list(current_models)):
            if model_a in removed or self.classifier_collection[model_a] is None:
                continue
            for model_b in list(current_models)[left_pos + 1:]:
                if model_b in removed or self.classifier_collection[model_b] is None:
                    continue
                key = tuple(sorted((model_a, model_b)))
                seen = self.model_pair_seen.get(key, 0)
                if seen == 0:
                    continue
                agreement = self.model_pair_agree.get(key, 0) / seen
                if agreement < self.similarity_margin:
                    continue

                err_a = self._mean_historical_error(model_a)
                err_b = self._mean_historical_error(model_b)
                keep, remove = (model_a, model_b) if err_a <= err_b else (model_b, model_a)
                if remove == self.current_classifier:
                    keep, remove = remove, keep
                removed_fade_score = self.model_fade_scores.get(remove, 0)
                self._remove_model(remove)
                removed.add(remove)
                self.model_merges += 1
                if self.fade_models_enabled:
                    self.model_fade_scores[keep] = (
                        self.model_fade_scores.get(keep, 0)
                        + removed_fade_score
                    )
        return [idx for idx in current_models if idx not in removed]

    def _fade_models(self, current_models):
        if not self.fade_models_enabled:
            return
        for model_id in list(current_models):
            if self.classifier_collection[model_id] is None:
                continue
            if model_id == self.current_classifier:
                self.model_fade_scores[model_id] = (
                    self.model_fade_scores.get(model_id, 0) + self.fade_points
                )
            else:
                self.model_fade_scores[model_id] = self.model_fade_scores.get(model_id, 0) - 1
                if self.model_fade_scores[model_id] <= 0:
                    self._remove_model(model_id)
                    self.models_faded += 1

    def _compare_current_and_new(self):
        if self.new_model is None:
            return
        if self.total_inst % self.model_check_freq != 0:
            return
        if self.new_error_sum < self.curr_error_sum:
            current_model = self.classifier_collection[self.current_classifier]
            self.classifier_collection[self.current_classifier] = self.new_model
            self.new_model = current_model
            self.curr_error_sum, self.new_error_sum = self.new_error_sum, self.curr_error_sum

    def _record_competition_error(self, x_row, y_value):
        current_model = self.classifier_collection[self.current_classifier]
        curr_pred = self._predict_one(current_model, x_row)
        curr_error = abs(curr_pred - float(y_value))
        self.curr_error_sum += curr_error
        self.total_inst += 1

        model_id = self.current_classifier
        self.model_error_sum[model_id] = self.model_error_sum.get(model_id, 0.0) + curr_error
        self.model_seen[model_id] = self.model_seen.get(model_id, 0) + 1

        if self.new_model is not None:
            new_pred = self._predict_one(self.new_model, x_row)
            self.new_error_sum += abs(new_pred - float(y_value))
        self._compare_current_and_new()

    def _handle_stable_batch(self, x_arr, y_arr):
        for index, x_row in enumerate(x_arr):
            self._record_competition_error(x_row, y_arr[index])
            self._train_model(
                self.classifier_collection[self.current_classifier],
                x_row.reshape(1, -1),
                np.asarray([y_arr[index]])
            )
            if self.new_model is not None:
                self._train_model(self.new_model, x_row.reshape(1, -1), np.asarray([y_arr[index]]))

    def _handle_warning_batch(self, x_arr, y_arr):
        for index, x_row in enumerate(x_arr):
            self.buffer_x.append(np.asarray(x_row, dtype=float).copy())
            self.buffer_y.append(float(y_arr[index]))

    def _handle_drift(self, x_arr=None, y_arr=None):
        if x_arr is not None and y_arr is not None:
            self._handle_warning_batch(x_arr, y_arr)
        if len(self.buffer_x) < self.min_buffer_size:
            return

        self.num_drifts += 1
        current_models = self._active_model_indices()
        buffer_x = np.asarray(self.buffer_x)
        buffer_y = np.asarray(self.buffer_y)

        self._update_similarity_stats(current_models, buffer_x, buffer_y)
        current_models = self._merge_similar_models(current_models)
        if not current_models:
            current_models = [self.current_classifier]

        errors = {
            model_id: self._model_error(self.classifier_collection[model_id], buffer_x, buffer_y)
            for model_id in current_models
        }
        best_model_id = min(errors, key=errors.get)

        self.new_model = self._new_base_model()
        self._train_model(self.new_model, buffer_x, buffer_y)

        reused_model = self._copy_model(self.classifier_collection[best_model_id])
        self.current_classifier = self._add_model(reused_model, fade_score=self.fade_points)
        self.model_reuses += 1

        current_models.append(self.current_classifier)
        self._fade_models(current_models)

        self.buffer_x = []
        self.buffer_y = []
        self.curr_error_sum = 0.0
        self.new_error_sum = 0.0
        self.total_inst = 0

    def handle_signal(self, signal, x=None, y=None):
        """React to an external drift-state signal.

        signal accepts "stable", "warning", or "drift". When x/y are provided
        with a drift signal, they are first appended to the warning buffer.
        """
        normalized = str(signal).lower()
        x_arr, y_arr = (None, None)
        if x is not None and y is not None:
            x_arr, y_arr = self._as_arrays(x, y)

        if normalized in ("stable", "in_control", "normal", "平稳"):
            if x_arr is not None:
                self._handle_stable_batch(x_arr, y_arr)
        elif normalized in ("warning", "warn", "警告"):
            if x_arr is not None:
                self._handle_warning_batch(x_arr, y_arr)
        elif normalized in ("drift", "change", "out_control", "漂移"):
            self._handle_drift(x_arr, y_arr)
        else:
            raise ValueError("signal must be stable, warning, or drift")

    def update(self, x, y):
        self.handle_signal(self.STABLE, x, y)

    def fit(self, x, y):
        self.reset()
        x_arr, y_arr = self._as_arrays(x, y)
        self._train_model(self.classifier_collection[self.current_classifier], x_arr, y_arr)

    def predict(self, x):
        x_arr, _ = self._as_arrays(x)
        model = self.classifier_collection[self.current_classifier]
        return np.asarray([self._predict_one(model, row) for row in x_arr])

    def get_status(self):
        return {
            "current_classifier": self.current_classifier,
            "num_models": len(self._active_model_indices()),
            "num_drifts": self.num_drifts,
            "model_reuses": self.model_reuses,
            "model_merges": self.model_merges,
            "models_faded": self.models_faded,
            "buffer_size": len(self.buffer_x),
            "has_new_model": self.new_model is not None,
        }


class RandomForest(BaseModel):
    def __init__(self, model_reuse=True):
        self.n_feature = None
        self.model_reuse = model_reuse
        self.model = RandomForestRegressor(warm_start=True, random_state=3)

    def update(self, x, y):
        self.model.n_estimators += 5
        self.model.fit(x, y)

    def fit(self, x, y):
        self.model = RandomForestRegressor(warm_start=True, random_state=3)
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class DeepPerf(BaseModel):
    def __init__(self, test_model=True):
        self.test_model = test_model
        self.n_feature = None
        self.n_count = None
        self.model = None
        self.config = None
        self.lr = None

    def update(self, x, y):
        weights, bias = self.model.get_weights()
        model = MTLSparseModel(self.config)
        model.read_weights(weights, bias)
        model.train(x, y, self.lr, max_epoch=1000)
        self.model = model

    def fit(self, x, y):
        self.n_count, self.n_feature = x.shape
        if not self.test_model:
            config, lr = hyper_tune(x, y)
        else:
            config = dict()
            config['num_neuron'] = 128
            config['num_input'] = self.n_feature
            config['num_layer'] = 8
            config['lambda'] = 0.123
            config['verbose'] = 1
            lr = 0.01
        self.model = MTLSparseModel(config)
        self.config = config
        self.lr = lr
        self.model.build_train()
        self.model.train(x, y, self.lr, max_epoch=2000)

    def predict(self, x):
        return self.model.predict(x)


class ARFRegressor(BaseModel):
    def __init__(self):
        # Set detection_method to None to disable detection
        self.model = AdaptiveRandomForestRegressor(random_state=3)

    def update(self, x, y):
        self.model.partial_fit(x, y)

    def fit(self, x, y):
        self.model.reset()
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class SRPRegressor(BaseModel):
    def __init__(self):
        self.model = SRP(model=HoeffdingTreeRegressor(grace_period=50))
        self.n_feature = None

    def update(self, x, y):
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
            self.model.learn_one(data_structure, y[index])

    def fit(self, x, y):
        self.model = SRP(model=HoeffdingTreeRegressor(grace_period=50))
        self.update(x, y)

    def predict(self, x):
        data_structure = {}
        for index, data in enumerate(x):
            for i, feature in enumerate(data):
                data_structure[i] = feature
        pre_y = self.model.predict_one(x=data_structure)
        return [pre_y]




class KRRegressor(BaseModel):
    def __init__(self):
        self.model = KernelRidge()

    def update(self, x, y):
        self.model.fit(x, y)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class SVRegressor(BaseModel):
    def __init__(self):
        self.model = SVR()

    def update(self, x, y):
        self.model.fit(x, y)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)









"""class DaLRegressor:
    def __init__(self):
        self.model = DaL_Regressor(1)

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        self.model.fit_model(x, y)"""


def build_incremental_model(regressor='RF', model_reuse=True, train_x=None, train_y=None):
    model = None
    if regressor == 'RF':
        return RandomForest(model_reuse)
    if regressor == 'LR':
        return LinearRegressor()
    if regressor == 'HT':
        model = HoeffdingTree()
    if regressor == 'ECPF':
        model = ECPF()
    if regressor == 'KNN':
        model = KNNRegression()
    if regressor == 'DeepPerf':
        model = DeepPerf()
    if regressor == 'ARF':
        model = ARFRegressor()
    if regressor == 'SRP':
        model = SRPRegressor()
    if regressor == 'SVR':
        model = SVRegressor()
    if regressor == 'KRR':
        model = KRRegressor()
    return model
