from util.mylog import timer
from model.base_modelapi import BaseModelAPI


class XGB_ModelAPI(BaseModelAPI):

    def __init__(self, config):
        self.c = config

    @timer
    def predict(self, test):
        test.reset_index(inplace=True)
        test.set_index('TransactionID', drop=False, inplace=True)
        X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

        y_test = self.model.predict(X_test)
        return y_test
