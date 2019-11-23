import pandas as pd
from feature_base import Feature
from mylog import timer
from configure import Config as c
import os


class Raw(Feature):

    @timer
    def _calculate(self):
        DATAPATH = '/home/yh/git/02-ieee-fraud-detection/data/'
        raw_dir = DATAPATH + 'raw'

        os.system(f'kaggle competitions download -c {c.project.ID} -p {raw_dir}')
        os.system(f'unzip "{raw_dir}/*.zip" -d {raw_dir}')
        os.system(f'rm -f {raw_dir}/*.zip')

        train_identity = pd.read_csv(f'{raw_dir}/train_identity.csv')
        test_identity = pd.read_csv(f'{raw_dir}/test_identity.csv')
        train_transaction = pd.read_csv(f'{raw_dir}/train_transaction.csv')
        test_transaction = pd.read_csv(f'{raw_dir}/test_transaction.csv')

        # join train_transaction and identity
        self.train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
        self.test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

        # FOR DEBUG: less data
        self.train = self.train.sample(frac=0.01, random_state=0)
        self.test = self.test.sample(frac=0.01, random_state=0)

        # no longer needed
        del train_identity, train_transaction, test_identity, test_transaction
