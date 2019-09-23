import ConfigBase


class Config:
    def __init__(self, **args):
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.training = TrainerConfig()


class ResourceConfig:
    def __init__(self, **args):
        self.param1 = 1


class ModelConfig:
    def __init__(self, **args):
        self.param1 = 2


class TrainerConfig:
    def __init__(self, **args):
        self.param1 = 3
