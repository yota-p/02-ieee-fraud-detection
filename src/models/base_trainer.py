from abc import ABCMeta, abstractmethod
import pickle

import sys
from pathlib import Path
ROOTDIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOTDIR / 'src'))

from utils.mylog import timer
from logging import getLogger
logger = getLogger('main')


class BaseTrainer(metaclass=ABCMeta):
    '''
    Abstract base class for trainer.
    Every concrete trainer class must inherit this class and
    implement train() method.
    Call train() to train, save, return model.
    '''
    model = None

    def __init__(self, config):
        self.c = config
        self.name = self.c.model.TYPE.lower()
        self.model_path = ROOTDIR / f'models/{self.name}_model.pkl'

    @timer
    def run(self, X_train, y_train):
        # Skip training & load if output is 'latest'
        if self.is_latest():
            logger.debug(f'Skip training {self.name}')
            self.load()
            return self.model

        self.train(X_train, y_train)
        self.save()
        logger.debug(f'Saved {self.name}_model')

    @abstractmethod
    def train(self, X_train, y_train):
        raise NotImplementedError

    @timer
    def is_latest(self):
        if self.model_path.exists():
            return True
        return False

    @timer
    def save(self):
        with open(str(self.model_path), mode='wb') as f:
            pickle.dump(self.model, f)

    @timer
    def load(self):
        with open(str(self.model_path), mode='rb') as f:
            self.model = pickle.load(f)

    @timer
    def get_model(self):
        return self.model
