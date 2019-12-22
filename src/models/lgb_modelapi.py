# lightgbm2
from utils.mylog import timer
from models.base_modelapi import BaseModelAPI


class LGB_ModelAPI(BaseModelAPI):
    c = None
    model = None

    def __init__(self, config):
        self.c = config

    @timer
    def predict(self, X_test):
        return self.model.clf.predict_proba(X_test)[:, 1]

    @timer
    def set_model(self, model):
        self.model = model
