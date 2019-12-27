# https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
import lightgbm as lgb
from logging import getLogger
from utils.mylog import timer

logger_train = getLogger('train')


class LGB_Skl_Model:

    def __init__(self, config):
        self.c = config
        self.clf = lgb.LGBMClassifier(**self.c.params)

    @timer
    def train(self, X_train, y_train, X_val, y_val, early_stopping_rounds):
        self.clf.fit(X_train, y_train,
                     eval_set=[(X_train, y_train), (X_val, y_val)],
                     verbose=1000,
                     early_stopping_rounds=early_stopping_rounds)

    @timer
    def predict(self, X_test):
        prediction = self.clf.predict_proba(X_test)[:, 1]
        return prediction

    @timer
    def get_feature_importances_(self):
        return self.clf.feature_importances_

    @timer
    def get_best_iteration_(self):
        return self.clf.best_iteration_

    @timer
    def get_best_score_(self):
        return self.clf.best_score_
