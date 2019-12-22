from utils.mylog import timer
from models import lightgbm
from models import lightgbm2
from models import xgb


class ModelAPIFactory:

    @timer
    def create(self, modelapi_config):
        if modelapi_config.model.TYPE == 'lightgbm':
            return lightgbm.modelapi.ModelAPI(modelapi_config)
        if modelapi_config.model.TYPE == 'lightgbm2':
            return lightgbm2.modelapi.ModelAPI(modelapi_config)
        if modelapi_config.model.TYPE == 'xgb':
            return xgb.modelapi.ModelAPI(modelapi_config)
        else:
            raise ValueError('{modelapi_config.model.TYPE} does not exist in factory menu')
