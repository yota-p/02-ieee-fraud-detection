# https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from logging import getLogger, DEBUG
import lightgbm as lgb

from utils.mylog import timer
from models.base_trainer import BaseTrainer

logger = getLogger('main')
logger_train = getLogger('train')


class LGB_Trainer(BaseTrainer):
    @timer
    def train(self, X, y):
        aucs = list()
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = X.columns

        # log header
        logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'eval_auc'))

        clf = lgb.LGBMClassifier(**self.c.model.params)

        # split data into train, validation
        folds = TimeSeriesSplit(n_splits=self.c.n_splits)
        for fold, (idx_train, idx_val) in enumerate(folds.split(X, y)):

            # prepare inputs for train
            logger.info(f'Training on fold {fold + 1}')
            X_train = X.iloc[idx_train]
            y_train = y.iloc[idx_train]
            X_val = X.iloc[idx_val]
            y_val = y.iloc[idx_val]
            callbacks = [self.log_evaluation(logger_train, period=10, fold=fold+1)]

            # train
            clf.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=1000,
                    callbacks=callbacks)

            # record result
            feature_importances[f'fold_{fold + 1}'] = clf.feature_importances_
            aucs.append(clf.best_score_['valid_1']['auc'])
            logger.debug(f'Fold {fold + 1} finished')

        logger.info('Training has finished.')
        logger.debug(f'Mean AUC: {np.mean(aucs)}')

        best_params = self.c.model.params
        del best_params['early_stopping_rounds']

        best_iteration = self.model.clf.best_iteration_
        if best_iteration is not None:
            best_params['n_estimators'] = best_iteration
        else:
            logger.warn('Training did not converge. Try larger n_estimators.')

        # TODO: save feature importance and other

        clf = lgb.LGBMClassifier(**best_params)
        clf.fit(X, y)
        self.model = clf

    def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG, fold=1):
        def _callback(env):
            if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
                train_auc = env.evaluation_result_list[0][2]
                eval_auc = env.evaluation_result_list[1][2]
                logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}\t{train_auc:.6f}\t{eval_auc:.6f}')
        _callback.order = 10
        return _callback
