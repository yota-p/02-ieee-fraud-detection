import pandas as pd
from feature_base import Feature
from mylog import timer
from configure import Config as c
import os
from logging import getLogger
logger = getLogger('main')
from reduce_mem_usage import reduce_mem_usage


class Raw(Feature):

    @timer
    def _calculate(self):
        raw_dir = c.storage.DATADIR / 'raw'

        logger.debug('Downloading & unarchiving raw data')
        os.system(f'kaggle competitions download -c {c.project.ID} -p {raw_dir}')
        os.system(f'unzip "{raw_dir}/*.zip" -d {raw_dir}')
        os.system(f'rm -f {raw_dir}/*.zip')

        logger.debug('Loading raw data')
        train_identity = pd.read_csv(f'{raw_dir}/train_identity.csv')
        test_identity = pd.read_csv(f'{raw_dir}/test_identity.csv')
        train_transaction = pd.read_csv(f'{raw_dir}/train_transaction.csv')
        test_transaction = pd.read_csv(f'{raw_dir}/test_transaction.csv')

        # join train_transaction and identity
        logger.debug('Merging raw data')
        self.train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
        self.test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

        # FOR DEBUG: less data
        if c.runtime.DEBUG:
            logger.info('Debug mode. Using 1% of raw data')
            self.train = self.train.sample(frac=0.01, random_state=42)
            self.test = self.test.sample(frac=0.01, random_state=42)

        self.train = reduce_mem_usage(self.train)
        self.test = reduce_mem_usage(self.test)

        # no longer needed
        del train_identity, train_transaction, test_identity, test_transaction
