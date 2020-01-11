import pathlib
from logging import DEBUG, INFO

ROOTDIR = pathlib.Path(__file__).parents[2].resolve()
VERSION = '0007'

runtime = {
    'ROOTDIR': ROOTDIR,
    'VERSION': VERSION,
    'RANDOM_SEED': 42,
    'DESCRIPTION': 'nroman_lgb',
    'RUN_TRAIN': True,
    'RUN_PRED': True,
    'out_sub_path': ROOTDIR / 'data/submission' / f'{VERSION}_submission.csv'
    }


transformer = {
    'ROOTDIR': ROOTDIR,
    'VERSION': VERSION,
    'features': ['nroman'],
    'DEBUG_SMALL_DATA': True,
    'out_train_path': ROOTDIR / 'data/feature' / f'train_{VERSION}.pkl',
    'out_test_path': ROOTDIR / 'data/feature' / f'test_{VERSION}.pkl'
    }


model = {
    'TYPE': 'lgb',
    'params': {'boosting_type': 'gbdt',
               'num_leaves': 491,
               'max_depth': -1,
               'learning_rate': 0.006883242363721497,
               'n_estimators': 10000,
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
               'early_stopping_rounds': 500
               }
    }


trainer = {
    'model': model,
    'n_splits': 5
    }


log = {
    'VERSION': VERSION,
    'LOGDIR': ROOTDIR / 'log',
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
        'TOKEN_PATH': pathlib.Path().home().resolve() / '.slack_token'
        }
    }
