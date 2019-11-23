# Reference: https://github.com/amaotone/spica/blob/master/spica/features/base.py
import argparse
import inspect
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
import pandas as pd
from mylog import timer
from logging import getLogger
main_logger = getLogger('main')


class Feature(metaclass=ABCMeta):
    '''
    Abstract base class for feature processor.
    Every feature processor class must inherit this class and
    implement _calculate() method.
    Call run() to
    '''
    prefix = ''
    suffix = ''
    dir = '/home/yh/git/02-ieee-fraud-detection/data/processed'

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip('_')
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.pkl'
        self.test_path = Path(self.dir) / f'{self.name}_test.pkl'

    @timer
    def create_feature(self):
        if self._isLatest():
            self._load()
            return self
        self._calculate()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.train.columns = prefix + self.train.columns + suffix
        self.test.columns = prefix + self.test.columns + suffix
        self._save()
        return self

    @timer
    def get_train_test(self):
        return self.train, self.test

    @abstractmethod
    def _calculate(self):
        raise NotImplementedError

    @timer
    def _save(self):
        self.train.to_pickle(str(self.train_path))
        self.test.to_pickle(str(self.test_path))
        return self

    @timer
    def _load(self):
        self.train = pd.read_pickle(str(self.train_path))
        self.test = pd.read_pickle(str(self.test_path))

    @timer
    def _isLatest(self):
        '''
        Check if this dataset exists
        TODO: Compare source & input file date
        '''
        if self.train_path.exists() and self.test_path.exists():
            main_logger.debug(f'Skipped {self.__class__.__name__}')
            return True
        else:
            return False


@timer
def get_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


@timer
def get_features(namespace):
    for k, v in ({k: v for k, v in namespace.items()}).items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


@timer
def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run()._save()
