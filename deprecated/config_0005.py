import pathlib
from logging import DEBUG, INFO
import numpy as np

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
    DEBUG_SMALL_DATA = False  # use 1% of data if True


class ModelConfig:
    TYPE = 'xgb'
    params = None

    if TYPE == 'xgb':
        params = {'n_estimators': 2000,
                  'max_depth': 12,
                  'learning_rate': 0.02,
                  'subsample': 0.8,
                  'colsample_bytree': 0.4,
                  'missing': np.nan,
                  'eval_metric': 'auc',
                  'nthread': 4,
                  'tree_method': 'hist'
                  # 'tree_method': 'gpu_hist'
                  }
        early_stopping_rounds = 100

    if TYPE is None:
        raise Exception(f'ModelConfig.params for {TYPE} is not defined')


class TrainerConfig:
    model = None

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
