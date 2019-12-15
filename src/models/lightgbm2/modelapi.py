# lightgbm2
from utils.mylog import timer


class ModelAPI:
    c = None
    model = None

    def __init__(self, config):
        self.c = config

    @timer
    def predict(self, X_test):
        # clf right now is the last model, trained with 80% of data and validated with 20%
        return self.model.clf.predict_proba(X_test)[:, 1]

    @timer
    def set_model(self, model):
        self.model = model
