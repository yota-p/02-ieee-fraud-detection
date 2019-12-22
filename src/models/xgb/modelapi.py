# XGB
from utils.mylog import timer


class ModelAPI:
    c = None
    model = None

    def __init__(self, config):
        self.c = config

    @timer
    def predict(self, test):
        test.reset_index(inplace=True)
        test.set_index('TransactionID', drop=False, inplace=True)
        X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

        return self.model.clf.predict_proba(X_test)[:, 1]

    @timer
    def set_model(self, model):
        self.model = model
