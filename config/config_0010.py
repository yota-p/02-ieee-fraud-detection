import pathlib
from logging import DEBUG, INFO
import numpy as np

VERSION = '0010'
ROOTDIR = pathlib.Path(__file__).parents[1]

config = {
    'runtime': {
        'ROOTDIR': ROOTDIR,
        'VERSION': VERSION,
        'RANDOM_SEED': 42,
        'DESCRIPTION': 'xgb-nroman',
        'RUN_TRAIN': True,
        'RUN_PRED': True,
        'USE_SMALL_DATA': False,
        },

    'transformer': {
        'ROOTDIR': ROOTDIR,
        'VERSION': VERSION,
        'features': ['nroman']
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
                   }
        },

    'trainer': {
        'n_splits': 5,
        },

    'log': {
        'VERSION': VERSION,
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
