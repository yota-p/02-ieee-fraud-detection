import pandas as pd
from pathlib import Path

from utils.mylog import timer
from transformer import Transformer
from trainer_factory import TrainerFactory
from modelapi_factory import ModelAPIFactory

ROOTDIR = Path(__file__).resolve().parents[1]


class Experiment:
    '''
    Pass data on memory as long as possible.
    '''

    def __init__(self, config):
        self.c = config

    @timer
    def run(self):
        transformer = Transformer(self.c.transformer)
        train, test = transformer.run()
        del transformer

        if self.c.experiment.RUN_TRAIN:
            trainer = TrainerFactory().create(self.c.trainer)
            trained_model = trainer.run(train)
            del train, trainer

        if self.c.experiment.RUN_PRED:
            modelapi = ModelAPIFactory().create(self.c.modelapi)
            modelapi.set_model(trained_model)

            sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
            sub['TransactionID'] = test['TransactionID']

            sub['isFraud'] = modelapi.run(test)  # returns y
            sub.to_csv(ROOTDIR / 'data/submission/submission.csv', index=False)  # TODO: set VERSION!
            del modelapi
