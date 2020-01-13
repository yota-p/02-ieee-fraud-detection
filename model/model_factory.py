from logging import getLogger
from model.lightgbm_model import LightGBM
from model.xgboost_model import XGBoost
from util.mylog import timer

logger = getLogger('main')


class ModelFactory:

    @timer
    def create(self, config: dict):
        logger.info(f'Creating model {config.TYPE}')
        if config.TYPE == 'lgb':
            return LightGBM(config)
        elif config.TYPE == 'xgb':
            return XGBoost(config)
        else:
            raise ValueError(f'Model {config.TYPE} is not in factory menu')
