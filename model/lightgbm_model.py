import lightgbm as lgb
from logging import DEBUG

from util.mylog import timer
from model.base_model import BaseModel


class LightGBM(BaseModel):
    '''
    Wrapper class of LightGBM.
    self.core contains Booster.
    '''

    @timer
    def __init__(self, config):
        self.config = config

    @timer
    def train(self, X_train, y_train):
        train_set = lgb.Dataset(X_train, y_train)

        self.core = lgb.train(params=self.config.params,
                              train_set=train_set,
                              verbose_eval=True,
                              )
        return self

    @timer
    def train_and_validate(self, X_train, y_train, X_val, y_val, logger, fold):
        train_set = lgb.Dataset(X_train, y_train)
        valid_set = lgb.Dataset(X_val, y_val)
        valid_sets = [train_set, valid_set]
        callbacks = [log_evaluation(logger, period=10, fold=fold)]

        self.core = lgb.train(params=self.config.params,
                              train_set=train_set,
                              valid_sets=valid_sets,
                              callbacks=callbacks
                              )
        return self

    @timer
    def predict(self, X_test):
        y_test = self.core.predict(X_test)
        return y_test

    @property
    def feature_importance(self):
        return self.core.feature_importance(importance_type='gain')

    @property
    def train_auc(self):
        return self.core.best_score['training']['auc']

    @property
    def val_auc(self):
        return self.core.best_score['valid_1']['auc']

    @property
    def best_iteration(self):
        return self.core.best_iteration


# for lightgbm.Booster
def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG, fold=1):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            train_auc = env.evaluation_result_list[0][2]
            eval_auc = env.evaluation_result_list[1][2]
            logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}\t{train_auc:.6f}\t{eval_auc:.6f}')
    _callback.order = 10
    return _callback
