# https://xgboost.readthedocs.io/en/latest/python/python_api.html#
import xgboost as xgb
from logging import DEBUG, getLogger

from util.mylog import timer
from model.base_model import BaseModel


class XGBoost(BaseModel):
    '''
    Wrapper class of XGBoost.
    self.core contains Booster.
    '''

    @timer
    def __init__(self, config):
        self.config = config

    @timer
    def train(self,
              X_train, y_train,
              num_boost_round):

        dtrain = xgb.DMatrix(X_train, label=y_train)

        self.core = xgb.train(params=self.config.params,
                              dtrain=dtrain,
                              verbose_eval=False,
                              num_boost_round=num_boost_round,
                              )
        return self

    @timer
    def train_and_validate(self, X_train, y_train,
                           X_val, y_val,
                           num_boost_round=100,
                           early_stopping_rounds=None,
                           logger=None,
                           fold=1,
                           ):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'eval')]
        logger = getLogger('train')
        callbacks = [log_evaluation(logger, period=1, fold=fold)]

        self.core = xgb.train(params=self.config.params,
                              dtrain=dtrain,
                              num_boost_round=num_boost_round,
                              evals=evals,
                              obj=None,
                              feval=None,
                              maximize=False,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=None,
                              verbose_eval=False,
                              callbacks=callbacks,
                              )
        return self

    @timer
    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        y_test = self.core.predict(dtest)
        return y_test

    @property
    def feature_importance(self):
        return self.core.get_score(importance_type='gain')

    @property
    def train_auc(self):
        return self.core.best_score['training']['auc']

    @property
    def val_auc(self):
        return self.core.evals_result['eval']['auc']

    @property
    def best_iteration(self):
        return self.core.best_iteration


# for XGBoost.Booster
def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG, fold=1):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            # XGBClassifier.evaluation_result_list contains values as below:
            #  env.evaluation_result_list = [('validation_0-auc', 0.882833), ('validation_1-auc', 0.827249)]
            train_auc = env.evaluation_result_list[0][1]
            eval_auc = env.evaluation_result_list[1][1]
            logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}\t{train_auc:.6f}\t{eval_auc:.6f}')
    _callback.order = 10
    return _callback
