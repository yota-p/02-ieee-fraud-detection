# https://lightgbm.readthedocs.io/en/latest/Python-API.html
import lightgbm as lgb
from logging import DEBUG, getLogger

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
    def train(self,
              X_train, y_train,
              X_val=None, y_val=None,
              num_boost_round=100,
              early_stopping_rounds=None,
              fold=0):
        train_set = lgb.Dataset(X_train, y_train)
        valid_set = lgb.Dataset(X_val, y_val)
        if X_val is not None and y_val is not None:
            valid_sets = [train_set, valid_set]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_set]
            valid_names = ['train']
        logger = getLogger('train')
        callbacks = [log_evaluation(logger, period=1, fold=fold, valid_sets=valid_sets)]

        self.core = lgb.train(params=self.config.params,
                              train_set=train_set,
                              valid_sets=valid_sets,
                              valid_names=valid_names,
                              num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds,
                              fobj=None,
                              feval=None,
                              init_model=None,
                              feature_name='auto',
                              categorical_feature='auto',
                              evals_result=None,
                              verbose_eval=False,
                              learning_rates=None,
                              keep_training_booster=False,
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
def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG, fold=1, valid_sets=None):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = ''
            for i in range(len(valid_sets)):
                result = result + f'\t{env.evaluation_result_list[i][2]:6f}'
            logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}{result}')
    _callback.order = 10
    return _callback
