# https://catboost.ai/docs/concepts/python-reference_train.html#python-reference_train
# https://github.com/catboost/catboost/blob/master/catboost/python-package/catboost/core.py
# Unfortunately CatBoost does not seems to support train callbacks
# Catboost.train has poor features and documentation, thus using Scikit-learn API.
from catboost import CatBoostClassifier
# from catboost import Pool
# from logging import DEBUG, getLogger

from util.mylog import timer
from model.base_model import BaseModel


class CatBoost(BaseModel):
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
              categorical_features=None,
              num_boost_round=100,
              early_stopping_rounds=None,
              fold=0):
        '''
        train_set = Pool(X_train, label=y_train)
        valid_set = Pool(X_val, label=y_val)
        if X_val is not None and y_val is not None:
            eval_set = [train_set, valid_set]
        else:
            eval_set = [train_set]
        # logger = getLogger('train')
        # callbacks = [log_evaluation(logger, period=1, fold=fold, eval_set=eval_set)]

        self.core = ctb.train(params=self.config.params,
                              dtrain=train_set,
                              logging_level=None,
                              verbose=None,
                              num_boost_round=num_boost_round,
                              eval_set=eval_set,
                              plot=None,
                              metric_period=None,
                              early_stopping_rounds=early_stopping_rounds,
                              save_snapshot=None,
                              snapshot_file=None,
                              snapshot_interval=None,
                              init_model=None,
                              )
        '''
        self.core = CatBoostClassifier(
            **self.config.params,
            num_boost_round=num_boost_round
        )
        self.core.fit(X=X_train,
                      y=y_train,
                      cat_features=categorical_features,
                      eval_set=(X_val, y_val),
                      verbose=True,
                      early_stopping_rounds=early_stopping_rounds,
                      )
        return self

    @timer
    def predict(self, X_test):
        y_test = self.core.predict_proba(X_test)[:, 1]
        return y_test

    @property
    def feature_importance(self):
        return self.core.get_feature_importance()

    @property
    def best_iteration(self):
        return self.core.get_best_iteration()

    @property
    def evals_result(self):
        return self.core.get_evals_result()


'''
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
'''
