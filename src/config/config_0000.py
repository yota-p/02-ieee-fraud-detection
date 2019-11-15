import pathlib
import sys
from src.utils import seeder
from get_option import get_option


class EnvironmentConfig:
    ROOTPATH = pathlib.Path(__file__).parents[2].resolve()
    SRCPATH = ROOTPATH / 'src'
    DATAPATH = ROOTPATH / 'data'
    LOGPATH = ROOTPATH / 'log'
    SEED = 42

    def __init__(self):
        # Add PATH
        sys.path.append(str(self.SRCPATH))
        # Random seed
        seeder.seed_everything(self.SEED)


class RuntimeConfig:
    '''
    Get option from command line arguments
    '''
    args = get_option()
    VERSION = args.version
    NO_SEND_MESSAGE = True  # args.nomsg
    DEBUG = args.dbg
    PREDICT = args.pred
    PREDICT_ONLY = args.predOnly
    TRAIN_ONE_ROUND = args.trainOneRound
    DASK_MODE = args.dask
    N_JOBS = args.nJobs


class ProjectConfig:
    NO = '02'
    ID = 'ieee-fraud-detection'


class LogConfig:
    class SlackAuth:
        CHANNEL = f'{ProjectConfig.NO}-{ProjectConfig.ID}'
        TOKEN_FILE = ".slack_token"
        TOKEN_PATH = EnvironmentConfig().ROOTPATH / TOKEN_FILE

    slackauth = SlackAuth()


class DatasetConfig:
    RAW = ['sample_submission.csv.zip',
           'test_identity.csv.zip',
           'test_transaction.csv.zip',
           'train_identity.csv.zip',
           'train_transaction.csv.zip']


class TransformerConfig:
    # Feature processor in order of execution
    # {'module':'Class'}
    PIPE = [{'altgor': 'Altgor'},
            {'raw': 'Raw'}
            ]
    pass


class ModelConfig:
    # Corresponds to directory name in src/models
    TYPE = 'lightgbm'


class TrainerConfig:
    pass


class ModelAPIConfig:
    pass


class Config:
    environment = EnvironmentConfig()
    runtime = RuntimeConfig()
    project = ProjectConfig()
    log = LogConfig()
    dataset = DatasetConfig()
    transformer = TransformerConfig()
    trainer = TrainerConfig()
    modelapi = ModelAPIConfig()
    model = ModelConfig()
