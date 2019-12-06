from mylog import timer
from models import lightgbm
from models import lightgbm2


class TrainerFactory:

    @timer
    def create(self, trainer_config):
        if trainer_config.model.TYPE == 'lightgbm':
            return lightgbm.trainer.Trainer(trainer_config)
        elif trainer_config.model.TYPE == 'lightgbm2':
            return lightgbm2.trainer.Trainer(trainer_config)
        else:
            raise ValueError('{trainer_config.model.TYPE} does not exist in factory menu')
