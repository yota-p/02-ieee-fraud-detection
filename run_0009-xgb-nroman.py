import gc
import warnings
import traceback
from logging import getLogger
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from util.easydict import EasyDict
from util.option import parse_option
from util.seeder import seed_everything
from util.mylog import create_logger, timer, blocktimer
from util.transformer import Transformer
from model.model_factory import ModelFactory

# config
from config.config_0009 import config


@timer
def main(c):

    with blocktimer('Preprocess'):
        train, test = Transformer.run(**c.transformer.__dict__)
        X_train, y_train, X_test = split_X_y(train, test)
        test = test.sort_values('TransactionDT')

    with blocktimer('Tune & Train'):
        modelfactory = ModelFactory()

        # tune the model params
        model = modelfactory.create(c.model)
        optimal_c_model = tune_gbdt_params(model, X_train, y_train, c.trainer.n_splits)

        # train with best params, full data
        model = modelfactory.create(optimal_c_model)
        model = model.train(X_train, y_train)

        # save results
        model.save(c.model.dir / f'model_{c.runtime.VERSION}_{c.model.TYPE}.pkl')

        importance = pd.DataFrame(model.feature_importance,
                                  index=X_train.columns,
                                  columns=['importance'])

        importance_path = c.runtime.ROOTDIR / 'feature/importance' / f'importance_{c.runtime.VERSION}.csv'
        importance.to_csv(importance_path)
        logger.info(f'Saved {str(importance_path)}')

    with blocktimer('Predict'):
        sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
        sub['TransactionID'] = test['TransactionID']

        y_test = model.predict(X_test)

        sub['isFraud'] = y_test
        sub.to_csv(c.runtime.out_sub_path, index=False)
        logger.debug(f'Saved {c.runtime.out_sub_path}')


def split_X_y(train, test):
    X_train = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_train = train.sort_values('TransactionDT')['isFraud']
    X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

    return X_train, y_train, X_test


@timer
def tune_gbdt_params(model, X, y, n_splits) -> dict:
    '''
    Tune parameter num_boost_round
    '''
    # start tuning train log
    create_logger('train', **c.log)
    logger_train = getLogger('train')
    logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'val_auc'))

    # aucs = list()

    # split data into train, validation
    folds = TimeSeriesSplit(n_splits=n_splits)
    for i, (idx_train, idx_val) in enumerate(folds.split(X, y)):
        fold = i + 1
        with blocktimer(f'Training on Fold {fold}'):
            X_train = X.iloc[idx_train]
            y_train = y.iloc[idx_train]
            X_val = X.iloc[idx_val]
            y_val = y.iloc[idx_val]

            # train
            model = model.train_and_validate(X_train, y_train, X_val, y_val, logger_train, fold)
            model.save(c.model.dir / f'model_{c.runtime.VERSION}_{c.model.TYPE}_fold{fold}.pkl')

            # record result
            # aucs.append(model.val_auc)
            # logger.info(f'train_auc: {model.train_auc} val_auc: {model.val_auc}')

    # logger.info(f'Mean AUC: {np.mean(aucs)}')

    # make optimal config from result
    optimal_c_model = model.config
    if model.best_iteration is not None:
        optimal_c_model.params['num_boost_round'] = model.best_iteration
    else:
        logger.warn('Did not meet early stopping. Try larger num_boost_rounds.')
    # no need after optimized num_boost_round
    del optimal_c_model.params['early_stopping_rounds']
    return optimal_c_model


if __name__ == "__main__":
    gc.enable()
    warnings.filterwarnings('ignore')

    # read config & apply option
    c = EasyDict(config)
    opt = parse_option()
    c.transformer.USE_SMALL_DATA = opt.small
    c.log.slackauth.NO_SEND_MESSAGE = opt.nomsg

    seed_everything(c.runtime.RANDOM_SEED)

    create_logger('main', **c.log)
    logger = getLogger('main')
    logger.info(f':thinking_face: Starting experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}')

    try:
        main(c)
        logger.info(f':sunglasses: Finished experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}')
    except Exception:
        logger.critical(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        logger.critical(f':skull: Stopped experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}')
