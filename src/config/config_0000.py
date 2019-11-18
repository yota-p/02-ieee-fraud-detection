import pathlib


class ProjectConfig:
    NO = '02'
    ID = 'ieee-fraud-detection'


class RuntimeConfig:
    VERSION = '0000'
    NO_SEND_MESSAGE = True  # args.nomsg
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
    LOGDIR = ROOTDIR / 'log'

    # Files
    RAW = ['sample_submission.csv.zip',
           'test_identity.csv.zip',
           'test_transaction.csv.zip',
           'train_identity.csv.zip',
           'train_transaction.csv.zip']


class LogConfig:
    class SlackAuth:
        CHANNEL = f'{ProjectConfig.NO}-{ProjectConfig.ID}'
        TOKEN_FILE = ".slack_token"
        TOKEN_PATH = StorageConfig().ROOTDIR / TOKEN_FILE
        SEND_MSG = True

    slackauth = SlackAuth()


class ExperimentConfig:
    '''
    Default model is Train only
    '''
    RUN_TRAIN = True
    RUN_PRED = False


class TransformerConfig:
    # Feature processor in order of execution
    # {'module':'Class'}
    PIPE = [{'altgor': 'Altgor'},
            {'raw': 'Raw'}
            ]


class ModelConfig:
    # Corresponds to directory name in src/models
    TYPE = 'lightgbm'


class TrainerConfig:
    pass


class ModelAPIConfig:
    pass


class Config:
    project = ProjectConfig()
    runtime = RuntimeConfig()
    storage = StorageConfig()
    log = LogConfig()
    experiment = ExperimentConfig()
    transformer = TransformerConfig()
    model = ModelConfig()
    trainer = TrainerConfig()
    modelapi = ModelAPIConfig()
