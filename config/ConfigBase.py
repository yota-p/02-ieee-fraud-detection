class ConfigBase:
    def __init__(self, **args):
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.training = TrainerConfig()


class ResourceConfig:
    def __init__(self, **args):
        self.param1 = 1


class ModelConfig:
    def __init__(self, **args):
        self.resnet_layer_num = 10
        self.l2_decay = 0.01


class TrainerConfig:
    def __init__(self, **args):
        self.param1 = 1
