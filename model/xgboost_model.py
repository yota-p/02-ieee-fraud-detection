# https://xgboost.readthedocs.io/en/latest/python/python_api.html#
# https://xgboost.readthedocs.io/en/latest/parameter.html
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
              X_val=None, y_val=None,
              num_boost_round=100,
              early_stopping_rounds=None,
              fold=0):
        # np.nan are treated as missing value by default
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        if X_val is not None and y_val is not None:
            evals = [(dtrain, 'train'), (dval, 'valid')]
        else:
            evals = [(dtrain, 'train')]
        logger = getLogger('train')
        callbacks = [log_evaluation(logger, period=1, fold=fold, evals=evals)]

        self.core = xgb.train(params=self.config.params,
                              dtrain=dtrain,
                              evals=evals,
                              num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds,
                              obj=None,
                              feval=None,
                              maximize=False,
                              evals_result=None,
                              verbose_eval=False,
                              xgb_model=None,
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


# for xgboost.Booster
def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG, fold=0, evals=None):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            # XGBClassifier.evaluation_result_list contains values as below:
            #  env.evaluation_result_list = [('validation_0-auc', 0.882833), ('validation_1-auc', 0.827249)]
            result = ''
            for i in range(len(evals)):
                result = result + f'\t{env.evaluation_result_list[i][1]:6f}'
            logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}{result}')
    _callback.order = 10
    return _callback
