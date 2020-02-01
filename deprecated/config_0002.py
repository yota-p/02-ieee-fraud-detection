import pathlib
from logging import DEBUG, INFO

ROOTDIR = pathlib.Path(__file__).parents[1].resolve()


class RuntimeConfig:
    VERSION = None
    DEBUG = False
    N_JOBS = -1
    RANDOM_SEED = 42


class LogConfig:
    ROOTDIR = pathlib.Path()
    VERSION = None
    slackauth = None

    FILE_HANDLER_LEVEL = DEBUG
    STREAM_HANDLER_LEVEL = DEBUG
    SLACK_HANDLER_LEVEL = INFO
    LOGFILE = VERSION
    LOGDIR = ROOTDIR / 'log'

    @classmethod
    def set_params(cls, VERSION, ROOTDIR, slackauth):
        cls.VERSION = VERSION
        cls.ROOTDIR = ROOTDIR
        cls.slackauth = slackauth


class SlackAuth:
    HOST = 'slack.com'
    URL = '/api/chat.postMessage'
    CHANNEL = 'ieee-fraud-detection'
    NO_SEND_MESSAGE = False
    TOKEN_PATH = pathlib.Path().home() / '.slack_token'

    @classmethod
    def set_params(cls, ROOTDIR):
        cls.ROOTDIR = ROOTDIR


class ExperimentConfig:
    RUN_TRAIN = True
    RUN_PRED = True


class TransformerConfig:
    features = ['magic']
    # features = ['nroman']


class ModelConfig:
    TYPE = 'xgb'
    params = None

    '''
    if TYPE == 'lgb':
        params = {'num_leaves': 491,
                  'min_child_weight': 0.03454472573214212,
                  'feature_fraction': 0.3797454081646243,
                  'bagging_fraction': 0.4181193142567742,
                  'min_data_in_leaf': 106,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': 0.006883242363721497,
                  'boosting_type': 'gbdt',
                  'bagging_seed': 11,
                  'metric': 'auc',
                  'verbosity': -1,
                  'reg_alpha': 0.3899927210061127,
                  'reg_lambda': 0.6485237330340494,
                  'random_state': 47,
                  }
    '''

    if TYPE == 'xgb':
        params = {'n_estimators': 2000,
                  'max_depth': 12,
                  'learning_rate': 0.02,
                  'subsample': 0.8,
                  'colsample_bytree': 0.4,
                  'missing': -1,
                  'eval_metric': 'auc',
                  'nthread': 4,
                  'tree_method': 'hist'
                  # 'tree_method': 'gpu_hist'
                  }

    if TYPE is None:
        raise Exception(f'ModelConfig.params for {TYPE} is not defined')


class TrainerConfig:
    model = None
    early_stopping_rounds = 100

    @classmethod
    def set_params(cls, model):
        cls.model = model


class ModelAPIConfig:
    model = None

    @classmethod
    def set_params(cls, model):
        cls.model = model


class Config:
    runtime = RuntimeConfig()
    SlackAuth().set_params(ROOTDIR=ROOTDIR)
    _slackauth = SlackAuth()
    LogConfig().set_params(VERSION=runtime.VERSION,
                           ROOTDIR=ROOTDIR,
                           slackauth=_slackauth)
    log = LogConfig()
    experiment = ExperimentConfig()
    transformer = TransformerConfig()
    _model = ModelConfig()
    TrainerConfig().set_params(model=_model)
    trainer = TrainerConfig()
    ModelAPIConfig.set_params(model=_model)
    modelapi = ModelAPIConfig()
