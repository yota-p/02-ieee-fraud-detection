import pathlib
from logging import DEBUG, INFO
import numpy as np

VERSION = '0008'
ROOTDIR = pathlib.Path(__file__).parents[1]

config = {
    'runtime': {
        'ROOTDIR': ROOTDIR,
        'VERSION': VERSION,
        'RANDOM_SEED': 42,
        'DESCRIPTION': 'xgb-magic',
        'RUN_TRAIN': True,
        'RUN_PRED': True,
        'out_sub_path': ROOTDIR / 'data/submission' / f'submission_{VERSION}.csv'
        },

    'transformer': {
        'ROOTDIR': ROOTDIR,
        'VERSION': VERSION,
        'features': ['magic'],
        'USE_SMALL_DATA': True,
        'out_train_path': ROOTDIR / 'data/feature' / f'transformed_{VERSION}_train.pkl',
        'out_test_path': ROOTDIR / 'data/feature' / f'transformed_{VERSION}_test.pkl'
        },

    'model': {
        'TYPE': 'xgb',
        'params': {'num_boost_round': 10000,
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
        'dir': ROOTDIR / 'data/model'
        },

    'trainer': {
        'n_splits': 5,
        },

    'log': {
        'VERSION': VERSION,
        'main_log_path': ROOTDIR / 'log' / f'main_{VERSION}.log',
        'train_log_path': ROOTDIR / 'log' / f'train_{VERSION}.tsv',
        'FILE_HANDLER_LEVEL': DEBUG,
        'STREAM_HANDLER_LEVEL': DEBUG,
        'SLACK_HANDLER_LEVEL': INFO,
        'slackauth': {
            'HOST': 'slack.com',
            'URL': '/api/chat.postMessage',
            'CHANNEL': 'ieee-fraud-detection',
            'NO_SEND_MESSAGE': False,
            'TOKEN_PATH': pathlib.Path().home() / '.slack_token'
            }
        }
}
