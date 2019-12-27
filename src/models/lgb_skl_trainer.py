import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from logging import getLogger
logger_train = getLogger('train')
import lightgbm as lgb

from utils.mylog import timer
from models.base_trainer import BaseTrainer
from models.lgb_skl_model import LGB_Skl_Model


class LGB_Skl_Trainer(BaseTrainer):

    @timer
    def train(self, train):

        # split train into X, y
        train.reset_index(inplace=True)
        train.set_index('TransactionID', drop=False, inplace=True)
        cols = train.columns.drop(['isFraud', 'TransactionDT', 'TransactionID'])
        X = train[cols]
        y = train['isFraud']

        folds = TimeSeriesSplit(n_splits=5)

        aucs = list()
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = X.columns

        # split data into train, validation
        for fold, (idx_train, idx_val) in enumerate(folds.split(X, y)):
            logger_train.debug(f'Training on fold {fold + 1}')

            X_train = X.iloc[idx_train]
            y_train = y.iloc[idx_train]
            X_val = X.iloc[idx_val]
            y_val = y.iloc[idx_val]

            # train
            self.model.train(X_train, y_train, X_val, y_val,
                             self.c.early_stopping_rounds)

            feature_importances[f'fold_{fold + 1}'] = self.model.get_feature_importances_()
            aucs.append(self.model.get_best_score_()['valid_1']['auc'])
            logger_train.debug(f'Fold {fold + 1} finished')

        logger_train.debug('Training has finished.')
        logger_train.debug(f'Mean AUC: {np.mean(aucs)}')

        best_iter = self.model.get_best_iteration_()
        print(best_iter)

        self.model.clf = lgb.LGBMClassifier(**self.c.model.params, num_boost_round=best_iter)
        self.model.clf.fit(X, y)

        return self.model

    @timer
    def set_model(self):
        self.model = LGB_Skl_Model(self.c.model)
