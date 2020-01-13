import pathlib
from logging import DEBUG, INFO

VERSION = '0007'
ROOTDIR = pathlib.Path(__file__).parents[1]

config = {
    'runtime': {
        'ROOTDIR': ROOTDIR,
        'VERSION': VERSION,
        'RANDOM_SEED': 42,
        'DESCRIPTION': 'lgb-nroman',
        'RUN_TRAIN': True,
        'RUN_PRED': True,
        'out_sub_path': ROOTDIR / 'data/submission' / f'submission_{VERSION}.csv'
        },

    'transformer': {
        'ROOTDIR': ROOTDIR,
        'VERSION': VERSION,
        'features': ['nroman'],
        'USE_SMALL_DATA': True,
        'out_train_path': ROOTDIR / 'data/feature' / f'transformed_{VERSION}_train.pkl',
        'out_test_path': ROOTDIR / 'data/feature' / f'transformed_{VERSION}_test.pkl'
        },

    'model': {
        'TYPE': 'lgb',
        'params': {'boosting_type': 'gbdt',
                   'num_leaves': 491,
                   'max_depth': -1,
                   'learning_rate': 0.006883242363721497,
                   'num_boost_round': 1000,
                   'objective': 'binary',
                   'min_child_weight': 0.03454472573214212,
                   'reg_alpha': 0.3899927210061127,
                   'reg_lambda': 0.6485237330340494,
                   'random_state': 47,
                   'feature_fraction': 0.3797454081646243,
                   'bagging_fraction': 0.4181193142567742,
                   'min_data_in_leaf': 106,
                   'bagging_seed': 11,
                   'metric': 'auc',
                   'verbosity': -1,
                   'max_bin': 255,
                   'early_stopping_rounds': 30
                   },
        'model_dir': ROOTDIR / 'data/model',
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
