from abc import ABCMeta, abstractmethod
import pickle

import sys
from pathlib import Path
ROOTDIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOTDIR / 'src'))

from utils.mylog import timer
from logging import getLogger
logger = getLogger('main')


class BaseModelAPI(metaclass=ABCMeta):
    '''
    Abstract base class for modelapi.
    Every concrete modelapi class must inherit this class and
    implement predict() method.
    Call precit() to predict by given model, returns prediction.
    '''
    name = ''
    model = None
    model_path = None
    prediction = None

    def __init__(self):
        self.name = self.model.TYPE.lower()
        self.model_path = ROOTDIR / f'models/{self.name}_model.pkl'

    @timer
    def run(self, test, force_calculate=False):
        '''
        # Skip training & load if output is 'latest'
        if not force_calculate and self._is_latest():
            logger.debug(f'Skip predicting {self.name}')
            self._load()
            return self.prediction
        '''

        self.predict(test)
        # self._save()
        # logger.debug(f'Saved {self.name}_prediction')
        return self.prediction

    @abstractmethod
    def predict(self, test):
        raise NotImplementedError

    '''
    @timer
    def _is_latest(self):
        if self.model_path.exists():
            return True
        return False
    '''

    '''
    @timer
    def _save(self):
        self.model.to_pickle(str(self.model_path))
    '''

    @timer
    def _load(self):
        with open(str(self.model_path), mode='rb') as f:
            self.model = pickle.load(f)
