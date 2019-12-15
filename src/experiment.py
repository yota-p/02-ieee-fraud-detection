import pandas as pd

from utils.mylog import timer
from transformer import Transformer
from trainer_factory import TrainerFactory
from modelapi_factory import ModelAPIFactory


class Experiment:

    def __init__(self, config):
        self.c = config

    @timer
    def run(self):
        transformer = Transformer(self.c.transformer)
        X_train, y_train, X_test, pks = transformer.run()  # TODO: remove pks
        del transformer

        if self.c.experiment.RUN_TRAIN:
            trainer = TrainerFactory().create(self.c.trainer)
            trained_model = trainer.train(X_train, y_train)
            del trainer

        if self.c.experiment.RUN_PRED:
            modelapi = ModelAPIFactory().create(self.c.modelapi)
            modelapi.set_model(trained_model)
            sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
            sub['TransactionID'] = pks['TransactionID']
            sub['isFraud'] = modelapi.predict(X_test)
            sub.to_csv(self.c.storage.DATADIR / 'processed' / '0001_submission.csv', index=False)
            del modelapi
