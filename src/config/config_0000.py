import pathlib
from logging import DEBUG, INFO


class ProjectConfig:
    NO = '02'
    ID = 'ieee-fraud-detection'


class RuntimeConfig:
    VERSION = '0000'
    DEBUG = False
    N_JOBS = -1
    RANDOM_SEED = 42


class StorageConfig:
    '''
    Specify by pathlib.Path() object
    '''
    # Directory
    ROOTDIR = pathlib.Path(__file__).parents[2].resolve()
    SRCDIR = ROOTDIR / 'src'
    DATADIR = ROOTDIR / 'data'

    # Files
    RAW = ['sample_submission.csv.zip',
           'test_identity.csv.zip',
           'test_transaction.csv.zip',
           'train_identity.csv.zip',
           'train_transaction.csv.zip']


class SlackAuth:
    HOST = 'slack.com'
    URL = '/api/chat.postMessage'
    CHANNEL = f'{ProjectConfig.NO}-{ProjectConfig.ID}'
    TOKEN_FILE = ".slack_token"
    TOKEN_PATH = StorageConfig().ROOTDIR / TOKEN_FILE
    NO_SEND_MESSAGE = False


class LogConfig:
    slackauth = SlackAuth()
    FILE_HANDLER_LEVEL = DEBUG
    STREAM_HANDLER_LEVEL = DEBUG
    SLACK_HANDLER_LEVEL = INFO
    FILENAME = RuntimeConfig().VERSION
    LOGDIR = StorageConfig().ROOTDIR / 'log'


class ExperimentConfig:
    '''
    Default model is Train only
    '''
    RUN_TRAIN = True
    RUN_PRED = False


class TransformerConfig:
    features = ['raw',
                'altgor']


class ModelConfig:
    TYPE = 'lgb'


class TrainerConfig:
    model = ModelConfig()


class ModelAPIConfig:
    model = ModelConfig()


class Config:
    project = ProjectConfig()
    runtime = RuntimeConfig()
    storage = StorageConfig()
    log = LogConfig()
    experiment = ExperimentConfig()
    transformer = TransformerConfig()
    trainer = TrainerConfig()
    modelapi = ModelAPIConfig()
