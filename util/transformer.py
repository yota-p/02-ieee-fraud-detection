import pandas as pd
from pathlib import Path
from feature.feature_factory import FeatureFactory
from util.mylog import timer
from logging import getLogger
logger = getLogger('main')


class Transformer:

    @classmethod
    @timer
    def run(cls,
            VERSION,
            features,
            DEBUG_SMALL_DATA,  # use 1% of data if True
            ROOTDIR,
            out_train_path,
            out_test_path,
            **kwargs
            ):
        '''
        Create features and return datas for training
        '''
        # check if output exists
        if isLatest([out_train_path, out_test_path]):
            train = pd.read_pickle(str(out_train_path))
            test = pd.read_pickle(str(out_test_path))
            if DEBUG_SMALL_DATA:
                train = train.sample(frac=0.01, random_state=42)
                test = test.sample(frac=0.01, random_state=42)
            logger.debug(f'Loaded train.shape: {train.shape}')
            logger.debug(f'Loaded test.shape:  {test.shape}')
            return train, test

        # Get key columns
        # TODO: refactor this into read_raw
        factory = FeatureFactory()
        raw = factory.create('raw')
        train_raw, test_raw = raw.create_feature()
        train = train_raw[['TransactionID']]
        test = test_raw[['TransactionID']]

        # For column: create features
        for namespace in features:
            feature = factory.create(namespace)
            train_feature, test_feature = feature.create_feature()

            # check if row # match before merge
            if not len(train.index) == len(train_feature.index):
                raise TypeError(f'Unable to merge: length of train and feature_train does not match.')
            if not len(test.index) == len(test_feature.index):
                raise TypeError(f'Unable to merge: length of test and feature_test does not match.')

            train = pd.merge(train, train_feature, how='left', on='TransactionID')
            test = pd.merge(test, test_feature, how='left', on='TransactionID')
            del feature, train_feature, test_feature

        train = train.sort_values(by=['TransactionDT'])
        test = test.sort_values(by=['TransactionDT'])

        # save processed data
        train.to_pickle(str(out_train_path))
        test.to_pickle(str(out_test_path))

        logger.debug(f'train.shape: {train.shape}')
        logger.debug(f'test.shape:  {test.shape}')

        return train, test


@timer
def isLatest(pathlist):
    for path in pathlist:
        if not path.exists():
            logger.debug(f'{path} does not exist')
            return False
        else:
            logger.debug(f'{path} exists')
    logger.debug('All files existed. Skip transforming.')
    return True