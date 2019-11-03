from get_option import get_option


class Config:
    def __init__(self, **args):
        self.VERSION = get_option.version
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.training = TrainerConfig()

    VERSION = '0000'


class ResourceConfig:
    def __init__(self, **args):
        self.param1 = 1


class ModelConfig:
    def __init__(self, **args):
        self.param1 = 2


class TrainerConfig:
    def __init__(self, **args):
        self.param1 = 3
