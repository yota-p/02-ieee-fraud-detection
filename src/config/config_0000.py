import pathlib
import sys
from src.utils import seeder
from get_option import get_option


class EnvironmentConfig:
    SRCPATH = pathlib.Path(__file__).parents[1].resolve()
    ROOTPATH = pathlib.Path(__file__).parents[2].resolve()

    def __init__(self):
        # Add PATH
        sys.path.append(str(self.SRCPATH))


class RuntimeConfig:
    VERSION = get_option().version
    DEBUG = True
    SEED = 42
    NO_SEND_MESSAGE = get_option().NoSendMessage

    def __init__(self):
        # Random seed
        seeder.seed_everything(self.SEED)


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
