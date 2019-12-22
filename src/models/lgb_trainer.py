from sklearn.model_selection import TimeSeriesSplit
from utils.mylog import timer
from models.base_trainer import BaseTrainer
from models.lgb_model import LGB_Model


class LGB_Trainer(BaseTrainer):

    @timer
    def train(self, X_train, y_train):

        self.model.train(X_train, y_train)

        return self.model

    @timer
    def set_model(self):
        self.model = LGB_Model(self.c.model)
