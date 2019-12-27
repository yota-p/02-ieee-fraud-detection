from utils.mylog import timer
from models.lgb_skl_trainer import LGB_Skl_Trainer
from models.lgb_trainer import LGB_Trainer
from models.xgb_trainer import XGB_Trainer


class TrainerFactory:

    @timer
    def create(self, trainer_config):
        if trainer_config.model.TYPE == 'lgb_skl':
            return LGB_Skl_Trainer(trainer_config)
        elif trainer_config.model.TYPE == 'lgb':
            return LGB_Trainer(trainer_config)
        elif trainer_config.model.TYPE == 'xgb':
            return XGB_Trainer(trainer_config)
        else:
            raise ValueError('{trainer_config.model.TYPE} does not exist in factory menu')
