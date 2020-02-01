import xgboost as xgb
from logging import getLogger, DEBUG
from util.mylog import timer

logger_train = getLogger('train')


def log_evaluation(logger, period=1, show_stdv=True, level=DEBUG, fold=1):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            # XGBClassifier.evaluation_result_list contains values as below
            # env.evaluation_result_list = [('validation_0-auc', 0.882833), ('validation_1-auc', 0.827249)]
            train_auc = env.evaluation_result_list[0][1]
            eval_auc = env.evaluation_result_list[1][1]
            logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}\t{train_auc:.6f}\t{eval_auc:.6f}')
    _callback.order = 10
    return _callback


class XGB_Model:

    def __init__(self, config):
        self.c = config
        self.clf = xgb.XGBClassifier(**self.c.params)

    @timer
    def train(self, X_train, y_train, X_val, y_val, fold):
        callbacks = [log_evaluation(logger_train, period=10, fold=fold)]
        self.clf.fit(X_train, y_train,
                     eval_set=[(X_train, y_train), (X_val, y_val)],
                     verbose=1000,
                     early_stopping_rounds=self.c.early_stopping_rounds,
                     callbacks=callbacks)

    @timer
    def predict(self, X_test):
        prediction = self.clf.predict_proba(X_test)[:, 1]
        return prediction
