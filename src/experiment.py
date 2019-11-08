from dataset import Raw
from transformer import Transformer
from save_log import stop_watch
from configurator import config as c
# dynamical import of models
from importlib import import_module
type = c.model.TYPE
Model = getattr(import_module(f'models.{type}.model'), 'Model')
Trainer = getattr(import_module(f'models.{type}.trainer'), 'Trainer')
ModelAPI = getattr(import_module(f'models.{type}.modelapi'), 'ModelAPI')


class Experiment:

    @stop_watch
    def run(self):
        raw = Raw()
        df_train = raw.load('train')
        df_test = raw.load('test')

        transformer = Transformer()
        df_train, df_test = transformer.transform(df_train, df_test)

        trainer = Trainer()
        trainer.train()

        modelapi = ModelAPI()
        pred = modelapi.predict(df_test)

        return
