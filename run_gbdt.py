import gc
import warnings
import traceback
from logging import getLogger
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from util.easydict import EasyDict
from util.option import parse_option
from util.seeder import seed_everything
from util.mylog import create_logger, timer, blocktimer
from util.transformer import Transformer
from model.model_factory import ModelFactory

# config
from config.config_0010 import config


@timer
def main(c):
    dsize = '.small' if c.runtime.USE_SMALL_DATA is True else ''
    with blocktimer('Preprocess'):
        out_transformed_train_path = c.runtime.ROOTDIR / 'data/feature' / \
            f'transformed_{c.runtime.VERSION}_train{dsize}.pkl'
        out_transformed_test_path = c.runtime.ROOTDIR / 'data/feature' / \
            f'transformed_{c.runtime.VERSION}_test{dsize}.pkl'
        train, test = Transformer.run(ROOTDIR=c.transformer.ROOTDIR,
                                      VERSION=c.transformer.VERSION,
                                      features=c.transformer.features,
                                      USE_SMALL_DATA=c.runtime.USE_SMALL_DATA,
                                      transformed_train_path=out_transformed_train_path,
                                      transformed_test_path=out_transformed_test_path,
                                      )
        X_train, y_train, X_test = split_X_y(train, test)
        test = test.sort_values('TransactionDT')

    with blocktimer('Tune & Train'):
        modelfactory = ModelFactory()

        # tune the model params
        model = modelfactory.create(c.model)
        optimal_c_model = tune_gbdt_params(model, X_train, y_train, c.trainer.n_splits, dsize)

        # train with best params, full data
        model = modelfactory.create(optimal_c_model)
        model = model.train(X_train, y_train)

        # save results
        out_model_dir = c.runtime.ROOTDIR / 'data/model' / f'model_{c.runtime.VERSION}_{c.model.TYPE}{dsize}.pkl'
        model.save(out_model_dir)

        importance = pd.DataFrame(model.feature_importance,
                                  index=X_train.columns,
                                  columns=['importance'])

        importance_path = c.runtime.ROOTDIR / 'feature/importance' / f'importance_{c.runtime.VERSION}{dsize}.csv'
        importance.to_csv(importance_path)
        logger.info(f'Saved {str(importance_path)}')

    with blocktimer('Predict'):
        sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
        sub['TransactionID'] = test['TransactionID']

        y_test = model.predict(X_test)

        sub['isFraud'] = y_test
        out_sub_path = c.runtime.ROOTDIR / 'data/submission' / f'submission_{c.runtime.VERSION}{dsize}.csv'
        sub.to_csv(out_sub_path, index=False)
        logger.debug(f'Saved {out_sub_path}')


def split_X_y(train, test):
    X_train = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_train = train.sort_values('TransactionDT')['isFraud']
    X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

    return X_train, y_train, X_test


@timer
def tune_gbdt_params(model, X, y, n_splits, dsize) -> dict:
    '''
    Tune parameter num_boost_round
    '''
    # start tuning train log
    train_log_path = c.runtime.ROOTDIR / 'log' / f'train_{c.runtime.VERSION}{dsize}.tsv'
    create_logger('train',
                  VERSION=c.runtime.VERSION,
                  log_path=train_log_path,
                  FILE_HANDLER_LEVEL=c.log.FILE_HANDLER_LEVEL,
                  STREAM_HANDLER_LEVEL=c.log.STREAM_HANDLER_LEVEL,
                  SLACK_HANDLER_LEVEL=c.log.SLACK_HANDLER_LEVEL,
                  slackauth=c.log.slackauth
                  )
    logger_train = getLogger('train')
    logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'val_auc'))

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
            out_model_fold_dir = c.runtime.ROOTDIR / 'data/model' / \
                f'model_{c.runtime.VERSION}_{c.model.TYPE}_fold{fold}{dsize}.pkl'
            model.save(out_model_fold_dir)

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
    c.runtime.USE_SMALL_DATA = opt.small
    c.log.slackauth.NO_SEND_MESSAGE = opt.nomsg

    seed_everything(c.runtime.RANDOM_SEED)

    dsize = '.small' if c.runtime.USE_SMALL_DATA is True else ''
    main_log_path = c.runtime.ROOTDIR / 'log' / f'main_{c.runtime.VERSION}{dsize}.log'
    create_logger('main',
                  VERSION=c.runtime.VERSION,
                  log_path=main_log_path,
                  FILE_HANDLER_LEVEL=c.log.FILE_HANDLER_LEVEL,
                  STREAM_HANDLER_LEVEL=c.log.STREAM_HANDLER_LEVEL,
                  SLACK_HANDLER_LEVEL=c.log.SLACK_HANDLER_LEVEL,
                  slackauth=c.log.slackauth
                  )
    logger = getLogger('main')
    logger.info(f':thinking_face: Starting experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}{dsize}')
    logger.info(f'Options indicated: {opt}')

    try:
        main(c)
        logger.info(f':sunglasses: Finished experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}{dsize}')
    except Exception:
        logger.critical(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        logger.critical(f':skull: Stopped experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}{dsize}')
