from dataset import raw
from transformer import Transformer
from trainer import Trainer


class Experiment:

    def run(self):
        df_train = raw.load('train')
        df_test = raw.load('test')

        transformer = Transformer()
        df_train, df_test = transformer.transform(df_train, df_test)

        trainer = Trainer()
        trainer.train()
        return
