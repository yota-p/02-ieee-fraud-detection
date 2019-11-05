import pathlib
import sys
from src.utils import seeder
from get_option import get_option


class EnvironmentConfig:
    SEED = 42

    def __init__(self):
        # Add PATH
        srcpath = pathlib.Path(__file__).parents[1].resolve()
        # Random seed
        sys.path.append(str(srcpath))
        seeder.seed_everything(self.SEED)


class ProjectConfig:
    ROOT = pathlib.Path(__file__).parents[2].resolve()
    NO = '02'
    ID = 'ieee-fraud-detection'


class RuntimeConfig:
    VERSION = get_option().version
    DEBUG = True


class SlackAuth:
    CHANNEL = f'{ProjectConfig.NO}-{ProjectConfig.ID}'
    TOKEN_FILE = ".slack_token"
    TOKEN_PATH = ProjectConfig().ROOT / TOKEN_FILE


class LogConfig:
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
    pass


class TrainerConfig:
    MODEL = 'lgbm'


class ModelAPIConfig:
    MODEL = 'lgbm'


class ExperimentConfig:
    transformer = TransformerConfig()
    trainer = TrainerConfig()
    modelapi = ModelAPIConfig()


class Config:
    environment = EnvironmentConfig()
    project = ProjectConfig()
    runtime = RuntimeConfig()
    log = LogConfig()
    dataset = DatasetConfig()
    experiment = ExperimentConfig()
