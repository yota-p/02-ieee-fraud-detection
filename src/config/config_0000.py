from get_option import get_option
import pathlib
import sys


class Config:
    def __init__(self):
        self.environment = EnvironmentConfig()
        self.runtime = RuntimeConfig()
        self.log = LogConfig()
        self.dataset = DatasetConfig()
        self.experiment = ExperimentConfig()


class EnvironmentConfig:
    def __init__(self):
        srcpath = pathlib.Path(__file__).parents[1].resolve()
        sys.path.append(str(srcpath))
        self.root = pathlib.Path(__file__).parents[2].resolve()


class RuntimeConfig:
    def __init__(self):
        self.VERSION = get_option().version
        self.DEBUG = True


class LogConfig:
    def __init__(self):
        pass


class DatasetConfig:
    def __init__(self):
        pass


class ExperimentConfig():
    def __init__(self):
        self.transformer = TransformerConfig()
        self.trainer = TrainerConfig()
        self.modelapi = ModelAPIConfig()


class TransformerConfig:
    def __init__(self):
        pass


class TrainerConfig:
    def __init__(self):
        self.model = ModelConfig()


class ModelAPIConfig:
    def __init__(self):
        self.model = ModelConfig()


class ModelConfig:
    def __init__(self):
        pass
