# https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
import lightgbm as lgb
from logging import getLogger, DEBUG
from utils.mylog import timer

logger_train = getLogger('train')


def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG, fold=1):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            train_auc = env.evaluation_result_list[0][2]
            eval_auc = env.evaluation_result_list[1][2]
            logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}\t{train_auc:.6f}\t{eval_auc:.6f}')
    _callback.order = 10
    return _callback


class LGB_Model:

    def __init__(self, config):
        self.c = config
        self.clf = lgb.LGBMClassifier(**self.c.params)

    @timer
    def train(self, X_train, y_train, X_val, y_val, fold):
        callbacks = [log_evaluation(logger_train, period=10, fold=fold)]
        self.clf.fit(X_train, y_train,
                     eval_set=[(X_train, y_train), (X_val, y_val)],
                     verbose=1000,
                     callbacks=callbacks)
        '''
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        clf = lgb.train(self.c.params, trn_data,
                        num_boost_round=10000,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=1000,
                        early_stopping_rounds=500,
                        callbacks=callbacks)
        self.clf = clf
        '''

    @timer
    def predict(self, X_test):
        prediction = self.clf.predict_proba(X_test)[:, 1]
        return prediction
