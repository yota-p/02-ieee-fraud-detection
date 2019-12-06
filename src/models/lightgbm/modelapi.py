from mylog import timer


class ModelAPI:

    def __init__(self, config):
        self.c = config

    @timer
    def predict(self, df_test):
        pass

    @timer
    def set_model(self, model):
        self.model = model
