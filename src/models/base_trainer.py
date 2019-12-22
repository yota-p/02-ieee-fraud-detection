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
    c = None
    name = ''
    model = None
    model_path = None

    def __init__(self, config):
        self.c = config
        self.name = self.c.model.TYPE.lower()
        self.set_model()
        self.model_path = ROOTDIR / f'models/{self.name}_model.pkl'

    @timer
    def run(self, train, force_calculate=False):
        # Skip training & load if output is 'latest'
        if not force_calculate and self._is_latest():
            logger.debug(f'Skip training {self.name}')
            self._load()
            return self.model

        self.train(train)
        self._save()
        logger.debug(f'Saved {self.name}_model')
        return self.model

    @abstractmethod
    def train(self, train):
        raise NotImplementedError

    @abstractmethod
    def set_model(self):
        '''
        Set self.model for modeltype
        '''
        raise NotImplementedError

    @timer
    def _is_latest(self):
        '''
        Check if this output is latest
        TODO: Compare source & input file date
        '''
        if self.model_path.exists():
            return True
        return False

    @timer
    def _save(self):
        with open(str(self.model_path), mode='wb') as f:
            pickle.dump(self.model, f)

    @timer
    def _load(self):
        with open(str(self.model_path), mode='rb') as f:
            self.model = pickle.load(f)
