# XGB
from utils.mylog import timer
from .model import Model


class Trainer:

    def __init__(self, config):
        self.c = config
        self.model = Model(self.c.model)

    @timer
    def train(self, train):
        # split data into train and validation set
        train.reset_index(inplace=True)
        train.set_index('TransactionID', drop=False, inplace=True)
        idx_train = train.index[:3*len(train)//4]
        idx_val = train.index[3*len(train)//4:]

        cols = train.columns.drop(['isFraud', 'TransactionDT', 'TransactionID'])
        X_train = train.loc[idx_train, cols]
        y_train = train.loc[idx_train, 'isFraud']
        X_val = train.loc[idx_val, cols]
        y_val = train.loc[idx_val, 'isFraud']

        self.model.train(X_train, y_train, X_val, y_val,
                         early_stopping_rounds=self.c.early_stopping_rounds)
        return self.model