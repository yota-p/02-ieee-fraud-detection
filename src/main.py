from altgorfuncs import *
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
# %matplotlib inline
# from tqdm import tqdm_notebook
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import NuSVR, SVR
# from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, \
    GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import linear_model
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import eli5
import shap
from IPython.display import HTML
import json
import altair as alt

import networkx as nx

from visualization import altairhelper

from data import preprocess

import logger
log = logger.Logger('test')

X, X_test = preprocess.main()

# LGBM 0.9397
n_fold = 5
folds = TimeSeriesSplit(n_splits=n_fold)
folds = KFold(n_splits=5)
'''
params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          #'categorical_feature': cat_cols
         }
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='auc', plot_feature_importance=True,
                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='usual', n_jobs=-1)

sub['isFraud'] = result_dict_lgb['prediction']
sub.to_csv('submission.csv', index=False)

sub.head()

pd.DataFrame(result_dict_lgb['oof']).to_csv('lgb_oof.csv', ind
'''

# XGB
xgb_params = {'eta': 0.04,
              'max_depth': 5,
              'subsample': 0.85,
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': True,
              'nthread': -1,
              'tree_method': 'hist'} # mod from 'gpu_hist'
result_dict_xgb = train_model_classification(X=X, X_test=X_test, y=y, params=xgb_params, folds=folds, model_type='xgb', eval_metric='auc', plot_feature_importance=False,
                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000, averaging='rank')

test = test.sort_values('TransactionDT')
test['prediction'] = result_dict_xgb['prediction']
sub['isFraud'] = pd.merge(sub, test, on='TransactionID')['prediction']
sub.to_csv('submission_xgb.csv', index=False)

# Blending
'''
test = test.sort_values('TransactionDT')
test['prediction'] = result_dict_lgb['prediction'] + result_dict_xgb['prediction']
sub['isFraud'] = pd.merge(sub, test, on='TransactionID')['prediction']
sub.to_csv('blend.csv', index=False)
'''
