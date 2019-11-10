# Not used
import os
from configurator import config as c
from save_log import timer
from abc import ABCMeta, abstractmethod
import pandas as pd


class Dataset(metaclass=ABCMeta):
    def __init__(self):
        # Get file from storage?
        pass

    @abstractmethod
    def load(self, label):
        df = pd.DataFrame()
        return df

    def isLatest(self):
        '''
        Check input, output, source and compare timestamps.
        If timestamp of output > input > source, return True.
        Else, return False.
        '''
        return False
