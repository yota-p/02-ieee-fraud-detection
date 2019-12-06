from mylog import timer
from .model import Model


class Trainer:

    def __init__(self, config):
        self.c = config
        self.model = Model(self.c.model)

    @timer
    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)
        return self.model
