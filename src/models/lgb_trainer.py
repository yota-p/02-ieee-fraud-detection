import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from logging import getLogger
import lightgbm as lgb

from utils.mylog import timer
from models.base_trainer import BaseTrainer
from models.lgb_model import LGB_Model

logger = getLogger('main')
logger_train = getLogger('train')


class LGB_Trainer(BaseTrainer):

    @timer
    def train(self, train):

        # split train into X, y
        train.reset_index(inplace=True)
        train.set_index('TransactionID', drop=False, inplace=True)
        cols = train.columns.drop(['isFraud', 'TransactionDT', 'TransactionID'])
        X = train[cols]
        y = train['isFraud']

        folds = TimeSeriesSplit(n_splits=self.c.n_splits)

        aucs = list()
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = X.columns

        # log header
        logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'eval_auc'))

        # split data into train, validation
        for fold, (idx_train, idx_val) in enumerate(folds.split(X, y)):
            logger.info(f'Training on fold {fold + 1}')

            X_train = X.iloc[idx_train]
            y_train = y.iloc[idx_train]
            X_val = X.iloc[idx_val]
            y_val = y.iloc[idx_val]

            # train
            self.model.train(X_train, y_train, X_val, y_val, fold+1)

            feature_importances[f'fold_{fold + 1}'] = self.model.clf.feature_importances_
            aucs.append(self.model.clf.best_score_['valid_1']['auc'])
            logger.debug(f'Fold {fold + 1} finished')

        logger.info('Training has finished.')
        logger.debug(f'Mean AUC: {np.mean(aucs)}')

        best_params = self.c.model.params
        del best_params['early_stopping_rounds']

        best_iteration = self.model.clf.best_iteration_
        # use best iteration found if converged
        if best_iteration is not None:
            best_params['n_estimators'] = best_iteration
        else:
            logger.warn('Training did not converge. Try increasing n_estimators')

        self.model.clf = lgb.LGBMClassifier(**best_params)
        self.model.clf.fit(X, y)

        return self.model

    @timer
    def set_model(self):
        self.model = LGB_Model(self.c.model)
