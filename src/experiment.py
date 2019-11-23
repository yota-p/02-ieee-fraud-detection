from mylog import timer
from transformer import Transformer
from trainer_factory import TrainerFactory
from modelapi_factory import ModelAPIFactory


class Experiment:

    def __init__(self, config):
        self.c = config

    @timer
    def run(self):
        transformer = Transformer(self.c.transformer)
        X_train, y_train, X_test = transformer.run()

        if self.c.experiment.RUN_TRAIN:
            trainer = TrainerFactory().create(self.c.trainer)
            trainer.train(X_train, y_train)
            trained_model = trainer.get_model()

        if self.c.experiment.RUN_PRED:
            modelapi = ModelAPIFactory().create(self.c.modelapi)
            modelapi.set_model(trained_model)
            y_test = modelapi.predict(X_test)
