from logging import getLogger
from model.lightgbm_model import LightGBM
from model.xgboost_model import XGBoost
from model.catboost_model import CatBoost
from util.mylog import timer

logger = getLogger('main')


class ModelFactory:
    @timer
    def create(self, config: dict):
        logger.debug(f'Creating model {config.type}')
        if config.type == 'lightgbm':
            return LightGBM(config)
        elif config.type == 'xgboost':
            return XGBoost(config)
        elif config.type == 'catboost':
            return CatBoost(config)
        else:
            raise ValueError(f'Model {config.type} is not in factory menu')
