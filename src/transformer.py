from mylog import timer
import gc
from feature_factory import FeatureFactory


class Transformer:
    '''
    Input: None
    Output: X_train, y_train, X_test
    '''

    def __init__(self, config):
        self.c = config

    @timer
    def run(self):
        train, test = None, None
        factory = FeatureFactory()

        # create features
        for namespace in self.c.features:
            feature = factory.create(namespace)
            feature.create_feature()
            train, test = feature.get_train_test()

        # split data into feature and target
        X_train = train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
        y_train = train['isFraud']
        X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

        gc.collect()

        return X_train, y_train, X_test
