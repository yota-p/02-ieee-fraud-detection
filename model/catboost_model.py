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
