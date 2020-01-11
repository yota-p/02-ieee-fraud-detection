import pathlib
from logging import DEBUG, INFO
import numpy as np

ROOTDIR = pathlib.Path(__file__).parents[2].resolve()
VERSION = '0008'

runtime = {
    'ROOTDIR': ROOTDIR,
    'VERSION': VERSION,
    'RANDOM_SEED': 42,
    'DESCRIPTION': 'magic_xgb',
    'RUN_TRAIN': True,
    'RUN_PRED': True,
    'out_sub_path': ROOTDIR / 'data/submission' / f'{VERSION}_submission.csv'
    }


transformer = {
    'ROOTDIR': ROOTDIR,
    'VERSION': VERSION,
    'features': ['magic'],
    'DEBUG_SMALL_DATA': True,
    'out_train_path': ROOTDIR / 'data/feature' / f'train_{VERSION}.pkl',
    'out_test_path': ROOTDIR / 'data/feature' / f'test_{VERSION}.pkl'
    }


model = {
    'TYPE': 'xgb',
    'params': {'n_estimators': 10000,
               'max_depth': 5,
               'learning_rate': 0.01,
               'subsample': 0.9,
               'colsample_bytree': 0.9,
               'missing': np.nan,
               'eval_metric': 'auc',
               'nthread': 4,
               'tree_method': 'hist',
               # 'tree_method': 'gpu_hist'
               'early_stopping_rounds': 10
               },
    }


trainer = {
    'model': model,
    'n_splits': 2
    }


log = {
    'VERSION': VERSION,
    'LOGDIR': ROOTDIR / 'log',
    'main_log_path': ROOTDIR / 'log/main' / f'{VERSION}.log',
    'train_log_path': ROOTDIR / 'log/train' / f'{VERSION}.tsv',
    'FILE_HANDLER_LEVEL': DEBUG,
    'STREAM_HANDLER_LEVEL': DEBUG,
    'SLACK_HANDLER_LEVEL': INFO,
    'slackauth': {
        'HOST': 'slack.com',
        'URL': '/api/chat.postMessage',
        'CHANNEL': 'ieee-fraud-detection',
        'NO_SEND_MESSAGE': False,
        'TOKEN_PATH': pathlib.Path().home().resolve() / '.slack_token'
        }
    }
