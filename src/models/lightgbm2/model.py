# https://www.kaggle.com/artgor/eda-and-models
import numpy as np
import pandas as pd
import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from logging import getLogger
from mylog import timer
logger_train = getLogger('train')


class Model:
    c = None
    clf = None

    def __init__(self, config):
        self.c = config

    @timer
    def train(self, X, y):

        folds = TimeSeriesSplit(n_splits=5)

        aucs = list()
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = X.columns

        for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
            logger_train.debug(f'Training on fold {fold + 1}')

            trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
            val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
            clf = lgb.train(self.c.params, trn_data, 10000, valid_sets=[
                            trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)

            feature_importances[f'fold_{fold + 1}'] = clf.feature_importance()
            aucs.append(clf.best_score['valid_1']['auc'])

            logger_train.debug(f'Fold {fold + 1} finished')

        logger_train.debug('Training has finished.')
        logger_train.debug(f'Mean AUC: {np.mean(aucs)}')

        best_iter = clf.best_iteration
        clf = lgb.LGBMClassifier(**self.c.params, num_boost_round=best_iter)
        clf.fit(X, y)
        self.clf = clf
        # TODO: save model to file
