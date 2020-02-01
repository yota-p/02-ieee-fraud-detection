import gc
import warnings
import traceback
from logging import getLogger
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
# import lightgbm as lgb

from util.option import parse_option
from util.seeder import seed_everything
from util.mylog import create_logger, timer, blocktimer
from util.transformer import Transformer
from model.model_factory import ModelFactory
# from model.base_trainer import BaseTrainer

from config.config_0007 import config
from util.easydict import EasyDict

warnings.filterwarnings('ignore')


@timer
def main(c_runtime, c_transformer, c_model, c_trainer, c_log):

    with blocktimer('Preprocess'):
        train, test = Transformer.run(**c_transformer.__dict__)
        X_train, y_train, X_test = split_X_y(train, test)

    with blocktimer('Tune & Train'):
        modelfactory = ModelFactory()

        # tune the model params
        model = modelfactory.create(c_model)
        optimal_c_model = tune_gbdt_params(model, X_train, y_train, c_trainer.n_splits)

        # train with best params, full data
        model = modelfactory.create(optimal_c_model)
        model = model.train(X_train, y_train)

    with blocktimer('Predict'):
        sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
        sub['TransactionID'] = test['TransactionID']

        y_test = model.predict(X_test)

        sub['isFraud'] = y_test
        sub.to_csv(c_runtime.out_sub_path, index=False)
        logger.info(f'Saved {c_runtime.out_sub_path}')


def split_X_y(train, test):
    train.reset_index(inplace=True)
    train.set_index('TransactionID', drop=False, inplace=True)
    cols = train.columns.drop(['isFraud', 'TransactionDT', 'TransactionID'])
    X_train = train[cols]
    y_train = train['isFraud']

    test.reset_index(inplace=True)
    test.set_index('TransactionID', drop=False, inplace=True)
    X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

    return X_train, y_train, X_test


@timer
def tune_gbdt_params(model, X, y, n_splits) -> dict:
    '''
    Tune parameters by training
    '''

    # start tuning train log
    create_logger('train', **c.log)
    logger_train = getLogger('train')
    logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'val_auc'))

    aucs = list()
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns

    # split data into train, validation
    folds = TimeSeriesSplit(n_splits=n_splits)
    for i, (idx_train, idx_val) in enumerate(folds.split(X, y)):
        fold = i + 1
        with blocktimer(f'Fold {fold}'):
            # prepare
            logger.info(f'Training on fold {fold}')
            X_train = X.iloc[idx_train]
            y_train = y.iloc[idx_train]
            X_val = X.iloc[idx_val]
            y_val = y.iloc[idx_val]

            # train
            model = model.train_and_validate(X_train, y_train, X_val, y_val, logger_train, fold)

            # record result
            feature_importances[f'fold_{fold}'] = model.feature_importance
            aucs.append(model.validation_auc)
            # TODO: save models at each steps
            logger.debug(f'Fold {fold} finished')

    logger.info('Training has finished.')
    logger.debug(f'Mean AUC: {np.mean(aucs)}')
    # TODO: save feature importance and other

    # make optimal config from result
    optimal_c_model = model.config
    if model.best_iteration is not None:
        # new param
        optimal_c_model.params['num_boost_round'] = model.best_iteration
    else:
        logger.warn('Did not meet early stopping. Try larger num_boost_rounds.')
    # no need after optimized num_boost_round
    del optimal_c_model.params['early_stopping_rounds']
    return optimal_c_model


if __name__ == "__main__":
    gc.enable()

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
        main(c_runtime=c.runtime,
             c_transformer=c.transformer,
             c_model=c.model,
             c_trainer=c.trainer,
             c_log=c.log
             )
        logger.info(f':sunglasses: Finished experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}')
    except Exception:
        logger.critical(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        logger.critical(f':skull: Stopped experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}')
