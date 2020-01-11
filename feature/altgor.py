# Reference: https://www.kaggle.com/artgor/eda-and-models
from pathlib import Path
import sys
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

ROOTDIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOTDIR))
from feature.feature_base import Feature
from util.mylog import timer


class Altgor(Feature):

    @timer
    def calculate(self):
        train_path = ROOTDIR / 'data/feature/raw_train.pkl'
        test_path = ROOTDIR / 'data/feature/raw_test.pkl'
        train = pd.read_pickle(train_path)
        test = pd.read_pickle(test_path)

        # Feature engineering
        train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / \
            train.groupby(['card1'])['TransactionAmt'].transform('mean')
        train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / \
            train.groupby(['card4'])['TransactionAmt'].transform('mean')
        train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / \
            train.groupby(['card1'])['TransactionAmt'].transform('std')
        train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / \
            train.groupby(['card4'])['TransactionAmt'].transform('std')

        test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / \
            test.groupby(['card1'])['TransactionAmt'].transform('mean')
        test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / \
            test.groupby(['card4'])['TransactionAmt'].transform('mean')
        test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / \
            test.groupby(['card1'])['TransactionAmt'].transform('std')
        test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / \
            test.groupby(['card4'])['TransactionAmt'].transform('std')

        train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
        train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
        train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
        train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

        test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
        test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
        test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
        test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

        train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
        train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
        train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
        train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

        test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
        test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
        test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
        test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

        train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
        train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
        train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
        train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')

        test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
        test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')
        test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
        test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')

        train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']
              ] = train['P_emaildomain'].str.split('.', expand=True)
        train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
              ] = train['R_emaildomain'].str.split('.', expand=True)
        test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']
             ] = test['P_emaildomain'].str.split('.', expand=True)
        test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
             ] = test['R_emaildomain'].str.split('.', expand=True)

        # Remove null
        many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
        many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

        # Remove large dispersion
        big_top_value_cols = [col for col in train.columns if train[col].value_counts(
            dropna=False, normalize=True).values[0] > 0.9]
        big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(
            dropna=False, normalize=True).values[0] > 0.9]

        one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
        one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

        cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols
                                + big_top_value_cols_test + one_value_cols + one_value_cols_test))
        cols_to_drop.remove('isFraud')

        train = train.drop(cols_to_drop, axis=1)
        test = test.drop(cols_to_drop, axis=1)

        # Category encoding
        cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19',
                    'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
                    'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
                    'DeviceType', 'DeviceInfo', 'ProductCD',
                    'P_emaildomain', 'R_emaildomain',
                    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                    'addr1', 'addr2',
                    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                    'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3',
                    'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
        for col in cat_cols:
            if col in train.columns:
                le = LabelEncoder()
                le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
                train[col] = le.transform(list(train[col].astype(str).values))
                test[col] = le.transform(list(test[col].astype(str).values))

        train = train.replace([np.inf, -np.inf], np.nan)
        test = test.replace([np.inf, -np.inf], np.nan)

        self.train = train
        self.test = test
