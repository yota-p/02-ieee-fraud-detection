import pandas as pd
import gc
from pathlib import Path
import sys
from feature_factory import FeatureFactory
from utils.mylog import timer
from logging import getLogger
logger = getLogger('main')

ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOTDIR / 'src'))


class Transformer:
    '''
    Create indicated features on config.features.
    '''

    def __init__(self, config):
        self.c = config

    @timer
    def run(self):
        '''
        Create features and return datas for training
        Input: None
        Return: X_train, y_train, X_test
        '''
        # Get key columns
        # TODO: refactor this into read_raw
        factory = FeatureFactory()
        raw = factory.create('raw')
        train_raw, test_raw = raw.create_feature()
        train = train_raw[['TransactionID']]
        test = test_raw[['TransactionID']]

        # For column: create features
        for namespace in self.c.features:
            feature = factory.create(namespace)
            train_feature, test_feature = feature.create_feature()

            # check if row # match before merge
            if not len(train.index) == len(train_feature.index):
                raise TypeError(f'Unable to merge: length of train and feature_train does not match.')
            if not len(test.index) == len(test_feature.index):
                raise TypeError(f'Unable to merge: length of test and feature_test does not match.')

            # Merge created feature to all
            logger.debug(f'Merging created feature {namespace} to transformed datas')

            logger.debug(f'Merge in1: train {train.shape}, in2: train_feature to merge: {train_feature.shape}')
            train = pd.merge(train, train_feature, how='left', on='TransactionID')
            logger.debug(f'Merge out: train {train.shape}')

            logger.debug(f'Merge in2: test {test.shape}, in2: test_feature to merge: {test_feature.shape}')
            test = pd.merge(test, test_feature, how='left', on='TransactionID')
            logger.debug(f'Merge out: train {test.shape}')
            del feature, train_feature, test_feature

        train = train.sort_values(by=['TransactionDT'])
        test = test.sort_values(by=['TransactionDT'])

        # save processed data
        train_path = ROOTDIR / 'data/processed' / f'features_train.pkl'
        test_path = ROOTDIR / 'data/processed' / f'features_test.pkl'
        train.to_pickle(str(train_path))
        test.to_pickle(str(test_path))

        # split data into feature and target
        X_train = train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
        y_train = train[['isFraud']]
        X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)
        pks = test[["TransactionDT", 'TransactionID']]  # TODO: remove this variable
        logger.debug(f'X_train.shape: {X_train.shape}')
        logger.debug(f'y_train.shape: {y_train.shape}')
        logger.debug(f'X_test.shape:  {X_test.shape}')

        del train, test
        gc.collect()

        return X_train, y_train, X_test, pks
