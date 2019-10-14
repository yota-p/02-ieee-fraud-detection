import pandas as pd
from config import project
from utils import pickleWrapper as pw
from config import mode
from save_log import stop_watch


@stop_watch('pickle_raw_data()')
def pickle_raw_data():
    dir = project.rootdir + 'data/raw/'

    train_identity = pd.read_csv(f'{dir}train_identity.csv')
    train_transaction = pd.read_csv(f'{dir}train_transaction.csv')
    test_identity = pd.read_csv(f'{dir}test_identity.csv')
    test_transaction = pd.read_csv(f'{dir}test_transaction.csv')

    if mode.DEBUG:
        # random 10% for developing
        pw.dump(train_identity.sample(frac=0.10, random_state=0), f'{dir}train_identity.pickle', 'wb')
        pw.dump(train_transaction.sample(frac=0.10, random_state=0), f'{dir}train_transaction.pickle', 'wb')
    else:
        # full data for production
        pw.dump(train_identity,    f'{dir}train_identity.pickle', 'wb')
        pw.dump(train_transaction, f'{dir}train_transaction.pickle', 'wb')

    pw.dump(test_identity,     f'{dir}test_identity.pickle', 'wb')
    pw.dump(test_transaction,  f'{dir}test_transaction.pickle', 'wb')


if __name__ == '__main__':
    pickle_raw_data()
