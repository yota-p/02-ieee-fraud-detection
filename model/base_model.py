from abc import ABCMeta, abstractmethod
import pickle

from util.mylog import timer


class BaseModel(metaclass=ABCMeta):
    '''
    Class to wrap models.
    Contain specific models from frameworks at self.core.
    '''
    core = None

    @abstractmethod
    def train(self, X_train, y_train):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError

    @timer
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.core, f)
        return self

    @timer
    def load(self, path):
        with open(path, 'rb') as f:
            self.core = pickle.load(f)
        return self
