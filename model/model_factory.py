from logging import getLogger
from model.lightgbm_model import LightGBM
from model.xgboost_model import XGBoost
from util.mylog import timer

logger = getLogger('main')


class ModelFactory:

    @timer
    def create(self, config: dict):
        logger.debug(f'Creating model {config.type}')
        if config.type == 'lgb':
            return LightGBM(config)
        elif config.type == 'xgb':
            return XGBoost(config)
        else:
            raise ValueError(f'Model {config.type} is not in factory menu')
