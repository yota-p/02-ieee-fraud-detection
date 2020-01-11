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
    VERSION = None
    features = ['nroman']
    DEBUG_SMALL_DATA = False  # use 1% of data if True


class ModelConfig:
    TYPE = 'lgb'

    if TYPE == 'lgb':
        params = {'boosting_type': 'gbdt',
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
    else:
        raise Exception(f'Model config for {TYPE} is not defined')


class TrainerConfig:
    model = None
    n_splits = 5

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
