from save_log import timer
import gc
from features.altgor import Altgor
from importlib import import_module
from configurator import config as c
for mod, cls in c.transformer.PIPE:
    import_module(f'features.{mod}')


class Transformer:

    @timer
    def transform(self, train, test):
        for mod, cls in c.transformer.PIPE:
            train, test = mod.cls().run().get_train_test()
        # TODO: choose method dynamically
        #train, test = Altgor().run().get_train_test()

        # Split data into feature and target
        X_train = train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
        y_train = train['isFraud']
        X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

        gc.collect()

        return X_train, y_train, X_test
