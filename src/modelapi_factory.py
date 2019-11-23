from mylog import timer
from models.lightgbm.modelapi import ModelAPI


class ModelAPIFactory:

    @timer
    def create(self, modelapi_config):
        if modelapi_config.model.TYPE == 'lightgbm':
            return ModelAPI(modelapi_config)
        else:
            return None
