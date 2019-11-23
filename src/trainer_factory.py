from mylog import timer
from models.lightgbm.trainer import Trainer


class TrainerFactory:

    @timer
    def create(self, trainer_config):
        if trainer_config.model.TYPE == 'lightgbm':
            return Trainer(trainer_config)
        else:
            return None
