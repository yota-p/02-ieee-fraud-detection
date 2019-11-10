from transformer import Transformer
from save_log import timer
from configurator import config as c
# dynamical import of models
from features.raw import Raw
from importlib import import_module
type = c.model.TYPE
Model = getattr(import_module(f'models.{type}.model'), 'Model')
Trainer = getattr(import_module(f'models.{type}.trainer'), 'Trainer')
ModelAPI = getattr(import_module(f'models.{type}.modelapi'), 'ModelAPI')


class Experiment:

    @timer
    def run(self):
        # get_raw_data()

        df_train, df_test = Raw().run().get_train_test()

        transformer = Transformer()
        df_train, df_test = transformer.transform(df_train, df_test)

        trainer = Trainer()
        trainer.train()

        modelapi = ModelAPI()
        pred = modelapi.predict(df_test)

        return
