# https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
import lightgbm as lgb
from logging import getLogger
from utils.mylog import timer

logger_train = getLogger('train')


class LGB_Model:

    def __init__(self, config):
        self.c = config
        self.clf = lgb.LGBMClassifier(**self.c.params)

    @timer
    def train(self, X_train, y_train, X_val, y_val, early_stopping_rounds):
        '''
        self.clf.fit(X_train, y_train,
                     eval_set=[(X_train, y_train), (X_val, y_val)],
                     num_boost_rounds=10000,
                     verbose_eval=1000,
                     early_stopping_rounds=early_stopping_rounds)
        '''
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        clf = lgb.train(self.c.params, trn_data, num_boost_round=10000, valid_sets=[
                            trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
        self.clf = clf

    @timer
    def predict(self, X_test):
        prediction = self.clf.predict_proba(X_test)[:, 1]
        return prediction

    @timer
    def get_feature_importances_(self):
        return self.clf.feature_importance()

    @timer
    def get_best_iteration_(self):
        return self.clf.best_iteration

    @timer
    def get_best_score_(self):
        return self.clf.best_score
