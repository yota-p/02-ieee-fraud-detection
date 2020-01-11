# Reference: https://github.com/amaotone/spica/blob/master/spica/features/base.py
import re
from abc import ABCMeta, abstractmethod
import pandas as pd
from logging import getLogger
import sys
from pathlib import Path

ROOTDIR = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOTDIR / 'data/feature'
logger = getLogger('main')

sys.path.insert(0, str(ROOTDIR))
from util.mylog import timer


class Feature(metaclass=ABCMeta):
    '''
    Abstract base class for feature processor.
    Every concrete feature class must inherit this class and
    implement _calculate() method.
    Call create_feature() to calc, save, return feature as train, test.
    '''
    prefix = ''
    suffix = ''

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip('_')
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = FEATURE_DIR / f'{self.name}_train.pkl'
        self.test_path = FEATURE_DIR / f'{self.name}_test.pkl'

    @timer
    def create_feature(self, force_calculate=False):
        '''
        Calculate features from given raw data.
        '''
        # Skip calculate & load if output is 'latest'
        if not force_calculate and self._is_latest():
            logger.debug(f'Skip calculating {self.__class__.__name__}')
            self._load()
            logger.debug(f'{self.name}_train.shape: {self.train.shape}')
            logger.debug(f'{self.name}_test.shape : {self.test.shape}')
            return self.train, self.test

        # Calculate & save feature
        self._calculate()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.train.columns = prefix + self.train.columns + suffix
        self.test.columns = prefix + self.test.columns + suffix
        self._save()
        logger.debug(f'Created {self.name}_train.shape: {self.train.shape}')
        logger.debug(f'Created {self.name}_test.shape : {self.test.shape}')
        return self.train, self.test

    @abstractmethod
    def _calculate(self, train_raw, test_raw):
        raise NotImplementedError

    @timer
    def _is_latest(self):
        '''
        Check if this output is latest
        TODO: Compare source & input file date
        '''
        if self.train_path.exists() and self.test_path.exists():
            return True
        return False

    @timer
    def _save(self):
        self.train.to_pickle(str(self.train_path))
        self.test.to_pickle(str(self.test_path))

    @timer
    def _load(self):
        self.train = pd.read_pickle(str(self.train_path))
        self.test = pd.read_pickle(str(self.test_path))
