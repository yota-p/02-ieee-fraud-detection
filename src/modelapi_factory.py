from utils.mylog import timer
from models.lgb_modelapi import LGB_ModelAPI
from models.xgb_modelapi import XGB_ModelAPI


class ModelAPIFactory:

    @timer
    def create(self, modelapi_config):
        if modelapi_config.model.TYPE == 'lgb':
            return LGB_ModelAPI(modelapi_config)
        elif modelapi_config.model.TYPE == 'xgb':
            return XGB_ModelAPI(modelapi_config)
        else:
            raise ValueError('{modelapi_config.model.TYPE} does not exist in factory menu')
