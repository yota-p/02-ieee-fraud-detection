import pathlib
import numpy as np

VERSION = '0010'
ROOTDIR = pathlib.Path(__file__).parents[1]

config = {
    'runtime': {
        'ROOTDIR': ROOTDIR,
        'VERSION': VERSION,
        'RANDOM_SEED': 42,
        'USE_SMALL_DATA': False,
        },

    'features': ['magic'],

    'model': {
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        'TYPE': 'xgb',
        'params': {
                   # General
                   'booster': 'gbtree',
                   'nthread': 4,
                   # Tree booster
                   'eta': 0.02,  # learning_rate
                   'gamma': 0,  # min_split_loss
                   'max_depth': 12,
                   'min_child_weight': 1,
                   'subsample': 0.8,
                   'colsample_bytree': 0.4,
                   'tree_method': 'hist',
                   # Training
                   'eval_metric': 'auc',
                   'missing': np.nan
                   }
        },

    'trainer': {
        'n_splits': 5,
        'num_boost_round': 5000,
        'early_stopping_rounds': 100
        },

    'slackauth': {
        'HOST': 'slack.com',
        'URL': '/api/chat.postMessage',
        'CHANNEL': 'ieee-fraud-detection',
        'NO_SEND_MESSAGE': False,
        'TOKEN_PATH': pathlib.Path().home() / '.slack_token'
        }
}
