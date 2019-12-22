from utils.mylog import timer
from models.base_modelapi import BaseModelAPI


class XGB_ModelAPI(BaseModelAPI):

    def __init__(self, config):
        self.c = config

    @timer
    def predict(self, test):
        test.reset_index(inplace=True)
        test.set_index('TransactionID', drop=False, inplace=True)
        X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

        self.prediction = self.model.clf.predict_proba(X_test)[:, 1]  # TODO: put this into model

    @timer
    def set_model(self, model):
        self.model = model
