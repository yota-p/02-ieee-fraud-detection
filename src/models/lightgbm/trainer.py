from numba import jit
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit

from utils.mylog import timer
from .model import Model


class Trainer:

    def __init__(self, config):
        self.c = config
        self.model = Model(self.c.model)

    @timer
    def train(self, X_train, y_train):
        X_test = X_train  # TODO: replace!

        n_fold = 5
        folds = TimeSeriesSplit(n_splits=n_fold)
        folds = KFold(n_splits=5)

        params = {'num_leaves': 256,
                  'min_child_samples': 79,
                  'objective': 'binary',
                  'max_depth': 13,
                  'learning_rate': 0.03,
                  "boosting_type": "gbdt",
                  "subsample_freq": 3,
                  "subsample": 0.9,
                  "bagging_seed": 11,
                  "metric": 'auc',
                  "verbosity": -1,
                  'reg_alpha': 0.3,
                  'reg_lambda': 0.3,
                  'colsample_bytree': 0.9,
                  #'categorical_feature': cat_cols
                  }
        result_dict_lgb = self.model.train_model_classification(
            X=X_train,
            X_test=X_test,
            y=y_train,
            params=params,
            folds=folds,
            model_type='lgb',
            eval_metric='auc',
            plot_feature_importance=True,
            verbose=500,
            early_stopping_rounds=200,
            n_estimators=5000,
            averaging='usual',
            n_jobs=-1)

    @timer
    def get_model(self):
        return self.model

    @jit
    def fast_auc(self, y_true, y_prob):
        """
        fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
        """
        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_prob)]
        nfalse = 0
        auc = 0
        n = len(y_true)
        for i in range(n):
            y_i = y_true[i]
            nfalse += (1 - y_i)
            auc += y_i * nfalse
        auc /= (nfalse * (n - nfalse))
        return auc

    def eval_auc(self, y_true, y_pred):
        """
        Fast auc eval function for lgb.
        """
        return 'auc', self.fast_auc(y_true, y_pred), True
