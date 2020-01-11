# Reference: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
from logging import getLogger, Formatter, StreamHandler, DEBUG
import gc
import datetime

logger = getLogger('main')
ROOTDIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOTDIR / 'data/raw'
DEBUG_MODE = False

sys.path.insert(0, str(ROOTDIR))
from feature.feature_base import Feature
from util.mylog import timer


# FREQUENCY ENCODING FOR TWO DFs
@timer
def encode_FE(df1, df2, cols, verbose=False):
    for col in cols:
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        if verbose:
            print(nm, ', ', end='')


# LABEL ENCODING FOR TWO DFs
@timer
def encode_LE(col, train, test, verbose=False):
    df_comb = pd.concat([train[col], test[col]], axis=0)
    df_comb, _ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max() > 32000:
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')
    else:
        train[nm] = df_comb[:len(train)].astype('int16')
        test[nm] = df_comb[len(train):].astype('int16')
    del df_comb
    gc.collect()
    if verbose:
        print(nm, ', ', end='')


# GROUP AGGREGATION MEAN AND STD
# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
@timer
def encode_AG(main_columns, uids, aggregations, train_df, test_df,
              fillna=True, usena=False, verbose=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                if usena:
                    temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg(
                    [agg_type]).reset_index().rename(columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name] = test_df[col].map(temp_df).astype('float32')

                if fillna:
                    train_df[new_col_name].fillna(-1, inplace=True)
                    test_df[new_col_name].fillna(-1, inplace=True)

                if verbose:
                    print("'"+new_col_name+"'", ', ', end='')


# COMBINE FEATURES
@timer
def encode_CB(col1, col2, df1, df2, verbose=False):
    nm = col1+'_'+col2
    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str)
    encode_LE(nm, df1, df2, verbose=False)
    if verbose:
        print(nm, ', ', end='')


# GROUP AGGREGATION NUNIQUE
@timer
def encode_AG2(main_columns, uids, train_df, test_df, verbose=False):
    for main_column in main_columns:
        for col in uids:
            comb = pd.concat([train_df[[col]+[main_column]], test_df[[col]+[main_column]]], axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
            if verbose:
                print(col+'_'+main_column+'_ct, ', end='')


class Magic(Feature):

    @timer
    def calcu(self):
        '''
        Load useful features
        '''
        # COLUMNS WITH STRINGS
        str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5',
                    'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
                    'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

        # FIRST 53 COLUMNS
        cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',
                'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
                'M5', 'M6', 'M7', 'M8', 'M9']

        # V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
        # https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
        v = [1, 3, 4, 6, 8, 11]
        v += [13, 14, 17, 20, 23, 26, 27, 30]
        v += [36, 37, 40, 41, 44, 47, 48]
        v += [54, 56, 59, 62, 65, 67, 68, 70]
        v += [76, 78, 80, 82, 86, 88, 89, 91]

        # v += [96, 98, 99, 104] #relates to groups, no NAN
        v += [107, 108, 111, 115, 117, 120, 121, 123]  # maybe group, no NAN
        v += [124, 127, 129, 130, 136]  # relates to groups, no NAN

        # LOTS OF NAN BELOW
        v += [138, 139, 142, 147, 156, 162]  # b1
        v += [165, 160, 166]  # b1
        v += [178, 176, 173, 182]  # b2
        v += [187, 203, 205, 207, 215]  # b2
        v += [169, 171, 175, 180, 185, 188, 198, 210, 209]  # b2
        v += [218, 223, 224, 226, 228, 229, 235]  # b3
        v += [240, 258, 257, 253, 252, 260, 261]  # b3
        v += [264, 266, 267, 274, 277]  # b3
        v += [220, 221, 234, 238, 250, 271]  # b3

        v += [294, 284, 285, 286, 291, 297]  # relates to grous, no NAN
        v += [303, 305, 307, 309, 310, 320]  # relates to groups, no NAN
        v += [281, 283, 289, 296, 301, 314]  # relates to groups, no NAN
        # v += [332, 325, 335, 338] # b4 lots NAN

        cols += ['V'+str(x) for x in v]
        dtypes = {}
        for c in cols+['id_0'+str(x) for x in range(1, 10)]+['id_'+str(x) for x in range(10, 34)]:
            dtypes[c] = 'float32'
        for c in str_type:
            dtypes[c] = 'category'

        train = pd.read_csv(RAW_DIR / 'train_transaction.csv',
                            index_col='TransactionID', dtype=dtypes, usecols=cols+['isFraud'])
        train_id = pd.read_csv(RAW_DIR / 'train_identity.csv',
                               index_col='TransactionID', dtype=dtypes)
        train = train.merge(train_id, how='left', left_index=True, right_index=True)
        # LOAD TEST
        test = pd.read_csv(RAW_DIR / 'test_transaction.csv',
                           index_col='TransactionID', dtype=dtypes, usecols=cols)
        test_id = pd.read_csv(RAW_DIR / 'test_identity.csv',
                              index_col='TransactionID', dtype=dtypes)
        test = test.merge(test_id, how='left', left_index=True, right_index=True)
        del train_id, test_id

        # Fix the mismatch of column names between train and test
        columns = {}
        for i in range(1, 39):
            no = str(i).zfill(2)
            columns[f'id-{no}'] = f'id_{no}'
        test = test.rename(columns=columns)

        # FOR DEBUG: less data
        if DEBUG_MODE:
            logger.info('Debug mode. Using 1% of raw data')
            train = train.sample(frac=0.01, random_state=42)
            test = test.sample(frac=0.01, random_state=42)

        # train = reduce_mem_usage(train)
        # test = reduce_mem_usage(test)

        logger.debug('Loaded & merged train, test from raw csv')
        logger.debug(f'train.shape: {train.shape}')
        logger.debug(f'test.shape : {test.shape}')

        '''
        Preprocessing
        '''
        # Normalize D columns
        for i in range(1, 16):
            if i in [1, 2, 3, 5, 9]:
                continue
            train['D'+str(i)] = train['D'+str(i)] - train.TransactionDT/np.float32(24*60*60)
            test['D'+str(i)] = test['D'+str(i)] - test.TransactionDT/np.float32(24*60*60)

        # LABEL ENCODE AND MEMORY REDUCE
        cols_to_label_encode = train.columns.drop('isFraud')
        for i, f in enumerate(cols_to_label_encode):
            # FACTORIZE CATEGORICAL VARIABLES
            if (np.str(train[f].dtype) == 'category') | (train[f].dtype == 'object'):
                df_comb = pd.concat([train[f], test[f]], axis=0)
                df_comb, _ = df_comb.factorize(sort=True)
                if df_comb.max() > 32000:
                    print(f, 'needs int32')
                train[f] = df_comb[:len(train)].astype('int16')
                test[f] = df_comb[len(train):].astype('int16')
            # SHIFT ALL NUMERICS POSITIVE. SET NAN to -1
            elif f not in ['TransactionAmt', 'TransactionDT']:
                mn = np.min((train[f].min(), test[f].min()))
                train[f] -= np.float32(mn)
                test[f] -= np.float32(mn)
                train[f].fillna(-1, inplace=True)
                test[f].fillna(-1, inplace=True)

        '''
        Feature Engineering
        We will now engineer features. All of these features where chosen because each increases local validation.
        The procedure for engineering features is as follows. First you think of an idea and create a new feature.
        Then you add it to your model and evaluate whether local validation AUC increases or decreases.
        If AUC increases keep the feature, otherwise discard the feature.
        '''
        # TRANSACTION AMT CENTS
        train['cents'] = (train['TransactionAmt'] - np.floor(train['TransactionAmt'])).astype('float32')
        test['cents'] = (test['TransactionAmt'] - np.floor(test['TransactionAmt'])).astype('float32')
        print('cents, ', end='')
        # FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
        encode_FE(train, test, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
        # COMBINE COLUMNS CARD1+ADDR1, CARD1+ADDR1+P_EMAILDOMAIN
        encode_CB('card1', 'addr1', train, test)
        encode_CB('card1_addr1', 'P_emaildomain', train, test)
        # FREQUENCY ENOCDE
        encode_FE(train, test, ['card1_addr1', 'card1_addr1_P_emaildomain'])
        # GROUP AGGREGATE
        encode_AG(['TransactionAmt', 'D9', 'D11'],
                  ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'],
                  train_df=train, test_df=test, usena=True)

        # CELL In[14]:
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
        train['DT_M'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        train['DT_M'] = (train['DT_M'].dt.year-2017)*12 + train['DT_M'].dt.month

        test['DT_M'] = test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        test['DT_M'] = (test['DT_M'].dt.year-2017)*12 + test['DT_M'].dt.month

        '''
        The Magic Feature - UID
        We will now create and use the MAGIC FEATURES.
        First we create a UID which will help our model find clients (credit cards).
        This UID isn't perfect. Many UID values contain 2 or more clients inside.
        However our model will detect this and by adding more splits with its trees,
        it will split these UIDs and find the single clients (credit cards).
        '''
        train['day'] = train.TransactionDT / (24*60*60)
        train['uid'] = train.card1_addr1.astype(str)+'_'+np.floor(train.day-train.D1).astype(str)

        test['day'] = test.TransactionDT / (24*60*60)
        test['uid'] = test.card1_addr1.astype(str)+'_'+np.floor(test.day-test.D1).astype(str)

        '''
        Group Aggregation Features
        For our model to use the new UID, we need to make lots of aggregated group features.
        We will add 47 new features! The pictures in the introduction to this notebook explain why this works.
        Note that after aggregation, we remove UID from our model. We don't use UID directly.
        '''
        # FREQUENCY ENCODE UID
        encode_FE(train, test, ['uid'])
        # AGGREGATE
        encode_AG(['TransactionAmt', 'D4', 'D9', 'D10', 'D15'], ['uid'], ['mean', 'std'],
                  train, test, fillna=True, usena=True)
        # AGGREGATE
        encode_AG(['C'+str(x) for x in range(1, 15) if x != 3], ['uid'], ['mean'],
                  train, test, fillna=True, usena=True)
        # AGGREGATE # TODO: FIX this! Doesn't work because it's not numeric
        encode_AG(['M'+str(x) for x in range(1, 10)], ['uid'], ['mean'],
                  train, test, fillna=True, usena=True)
        # AGGREGATE
        encode_AG2(['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents'], ['uid'], train, test)
        # AGGREGATE
        encode_AG(['C14'], ['uid'], ['std'], train, test, fillna=True, usena=True)
        # AGGREGATE
        encode_AG2(['C13', 'V314'], ['uid'], train, test)
        # AGGREATE
        encode_AG2(['V127', 'V136', 'V309', 'V307', 'V320'], ['uid'], train, test)
        # NEW FEATURE
        train['outsider15'] = (np.abs(train.D1-train.D15) > 3).astype('int8')
        test['outsider15'] = (np.abs(test.D1-test.D15) > 3).astype('int8')

        logger.debug('Feature engineering finished')
        logger.debug(f'train.shape: {train.shape}')
        logger.debug(f'test.shape : {test.shape}')

        '''
        Feature Selection - Time Consistency
        We added 28 new feature above. We have already removed 219 V Columns from correlation analysis done here.
        So we currently have 242 features now. We will now check each of our 242 for "time consistency".
        We will build 242 models. Each model will be trained on the first month of the training data and
        will only use one feature. We will then predict the last month of the training data. We want both
        training AUC and validation AUC to be above AUC = 0.5. It turns out that 19 features fail this test
        so we will remove them. Additionally we will remove 7 D columns that are mostly NAN. More techniques
        for feature selection are listed here
        '''
        cols_to_drop = []
        # Why drop these???
        cols_to_drop += ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']
        cols_to_drop += ['DT_M', 'day', 'uid']  # leave 'TransactionDT'
        # These features FAILED TIME CONSISTENCY TEST
        cols_to_drop += ['C3', 'M5', 'id_08', 'id_33']
        cols_to_drop += ['card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']
        cols_to_drop += ['id_'+str(x) for x in range(22, 28)]

        # Select needed features only
        train = train.drop(cols_to_drop, axis=1)
        test = test.drop(cols_to_drop, axis=1)

        logger.debug(f'Now using the following {len(train.columns)} feature')
        logger.info(str(np.array(train.columns)))

        self.train = train
        self.test = test
        del train, test


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

    f = Magic()
    f.create_feature(force_calculate=option.force)
