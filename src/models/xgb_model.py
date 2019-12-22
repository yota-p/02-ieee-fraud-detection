import xgboost as xgb
from logging import getLogger
from utils.mylog import timer

logger = getLogger('main')
logger_train = getLogger('train')


class XGB_Model:

    def __init__(self, config):
        self.config = config
        self.clf = xgb.XGBClassifier(**config.params)

    @timer
    def train(self, X_train, y_train, X_val, y_val, early_stopping_rounds):
        self.clf.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     verbose=50,
                     early_stopping_rounds=early_stopping_rounds)
