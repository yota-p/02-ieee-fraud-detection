# TODO: save model to file, separate logics
# https://www.kaggle.com/artgor/eda-and-models
import numpy as np
import lightgbm as lgb
from logging import getLogger
from utils.mylog import timer
logger_train = getLogger('train')


class LGB_Model:
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

        self.clf = lgb.train(self.c.params, trn_data, 10000, valid_sets=[
                            trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)

        feature_importances[f'fold_{fold + 1}'] = self.clf.feature_importance()
        aucs.append(self.clf.best_score['valid_1']['auc'])

        logger_train.debug(f'Fold {fold + 1} finished')

        logger_train.debug('Training has finished.')
        logger_train.debug(f'Mean AUC: {np.mean(aucs)}')

        best_iter = self.clf.best_iteration
        self.clf = lgb.LGBMClassifier(**self.c.params, num_boost_round=best_iter)
        self.clf.fit(X, y)
