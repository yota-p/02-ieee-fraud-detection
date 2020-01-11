import argparse
import pandas as pd
import os
from pathlib import Path
import sys
from logging import getLogger, Formatter, StreamHandler, DEBUG

logger = getLogger('main')
ROOTDIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOTDIR / 'data/raw'
DEBUG_MODE = False  # for small size data
PROJECTID = 'ieee-fraud-detection'

sys.path.insert(0, str(ROOTDIR / 'src'))
from features.feature_base import Feature
from utils.mylog import timer
from utils.reduce_mem_usage import reduce_mem_usage


class Raw(Feature):

    @timer
    def _calculate(self):

        if not self.is_downloaded():
            logger.debug('Downloading & unarchiving raw data')
            os.system(f'kaggle competitions download -c {PROJECTID} -p {RAW_DIR}')
            os.system(f'unzip "{RAW_DIR}/*.zip" -d {RAW_DIR}')
            os.system(f'rm -f {RAW_DIR}/*.zip')

        logger.debug('Loading raw csv')
        train_identity = pd.read_csv(f'{RAW_DIR}/train_identity.csv')
        test_identity = pd.read_csv(f'{RAW_DIR}/test_identity.csv')
        train_transaction = pd.read_csv(f'{RAW_DIR}/train_transaction.csv')
        test_transaction = pd.read_csv(f'{RAW_DIR}/test_transaction.csv')

        # Fix the mismatch of column names between train and test
        columns = {}
        for i in range(1, 39):
            no = str(i).zfill(2)
            columns[f'id-{no}'] = f'id_{no}'
        test_identity = test_identity.rename(columns=columns)

        # join train_transaction and identity
        logger.debug('Merging raw data')
        self.train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
        self.test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
        logger.debug(f'Merged into train.shape: {self.train.shape}')
        logger.debug(f'Merged into test.shape : {self.test.shape}')

        # FOR DEBUG: less data
        if DEBUG_MODE:
            logger.info('Debug mode. Using 1% of raw data')
            self.train = self.train.sample(frac=0.01, random_state=42)
            self.test = self.test.sample(frac=0.01, random_state=42)

        self.train = reduce_mem_usage(self.train)
        self.test = reduce_mem_usage(self.test)

        # no longer needed
        del train_identity, train_transaction, test_identity, test_transaction

    @timer
    def is_downloaded(self):
        raw_paths = [RAW_DIR / 'sample_submission.csv',
                     RAW_DIR / 'test_identity.csv',
                     RAW_DIR / 'test_transaction.csv',
                     RAW_DIR / 'train_identity.csv',
                     RAW_DIR / 'train_transaction.csv']
        for path in raw_paths:
            if not path.exists():
                return False
        return True


if __name__ == "__main__":
    formatter = Formatter('[%(asctime)s] %(levelname)-8s >> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = getLogger('main')
    logger.setLevel(DEBUG)
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--force',
                           action='store_true',
                           help='Force re-calculation')
    argparser.add_argument('-d', '--debug',
                           action='store_true',
                           help='Force re-calculation')
    option = argparser.parse_args()
    DEBUG_MODE = option.debug

    f = Raw()
    f.create_feature(force_calculate=option.force)
