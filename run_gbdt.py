import gc
import warnings
import traceback
from logging import getLogger, INFO, DEBUG
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

from util.easydict import EasyDict
from util.option import parse_option
from util.seeder import seed_everything
from util.mylog import create_logger, timer, blocktimer
from util.transformer import Transformer
from model.model_factory import ModelFactory

# config
from config.config_0012 import config
# from config.config_0012 import config
# from config.config_0013 import config


@timer
def main(c):
    dsize = '.small' if c.runtime.use_small_data is True else ''
    with blocktimer('Preprocess'):
        out_transformed_train_path = Path(f'data/feature/transformed_{c.runtime.VERSION}_train{dsize}.pkl')
        out_transformed_test_path = Path(f'data/feature/transformed_{c.runtime.VERSION}_test{dsize}.pkl')
        train, test = Transformer.run(c.features,
                                      use_small_data=c.runtime.use_small_data,
                                      transformed_train_path=out_transformed_train_path,
                                      transformed_test_path=out_transformed_test_path
                                      )
        X_train, y_train, X_test = split_X_y(train, test)
        test = test.sort_values('TransactionDT')

    with blocktimer('Tune & Train'):
        modelfactory = ModelFactory()

        # tune the model params
        model = modelfactory.create(c.model)
        best_iteration = optimize_num_boost_round(model, X_train, y_train, c.trainer.n_splits, dsize)

        # train with best params, full data
        model = modelfactory.create(c.model)
        model = model.train(X_train, y_train, num_boost_round=best_iteration)

        # save results
        out_model_dir = Path(f'data/model/model_{c.runtime.VERSION}_{c.model.type}{dsize}.pkl')
        model.save(out_model_dir)

        importance = pd.DataFrame(model.feature_importance,
                                  index=X_train.columns,
                                  columns=['importance'])
        importance_path = Path(f'feature/importance/importance_{c.runtime.VERSION}{dsize}.csv')
        importance.to_csv(importance_path)
        logger.info(f'Saved {str(importance_path)}')

    with blocktimer('Predict'):
        sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
        sub['TransactionID'] = test['TransactionID']

        y_test = model.predict(X_test)

        sub['isFraud'] = y_test
        out_sub_path = Path(f'data/submission/submission_{c.runtime.VERSION}{dsize}.csv')
        sub.to_csv(out_sub_path, index=False)
        logger.debug(f'Saved {out_sub_path}')


def split_X_y(train, test):
    X_train = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_train = train.sort_values('TransactionDT')['isFraud']
    X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

    return X_train, y_train, X_test


@timer
def optimize_num_boost_round(model, X, y, n_splits, dsize) -> dict:
    '''
    Tune parameter num_boost_round
    '''
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
            model = model.train(X_train, y_train,
                                X_val, y_val,
                                num_boost_round=c.trainer.num_boost_round,
                                early_stopping_rounds=c.trainer.early_stopping_rounds,
                                fold=fold)
            out_model_fold_dir = Path(f'data/model/model_{c.runtime.VERSION}_{c.model.type}_fold{fold}{dsize}.pkl')
            model.save(out_model_fold_dir)

    # make optimal config from result
    if model.best_iteration is not None:
        logger.info(f'Early stopping. Best iteration is: {model.best_iteration}')
        return model.best_iteration
    else:
        logger.warn('Did not meet early stopping. Try larger num_boost_rounds.')
        return c.train.num_boost_round


if __name__ == "__main__":
    gc.enable()
    warnings.filterwarnings('ignore')

    # read config & apply option
    c = EasyDict(config)
    opt = parse_option()
    c.runtime = {}
    c.runtime.VERSION = opt.version
    c.runtime.use_small_data = opt.small
    c.runtime.no_send_message = opt.nomsg
    c.runtime.random_seed = opt.seed

    seed_everything(c.runtime.random_seed)

    dsize = '.small' if c.runtime.use_small_data is True else ''
    main_log_path = Path(f'log/main_{c.runtime.VERSION}{dsize}.log')
    train_log_path = Path(f'log/train_{c.runtime.VERSION}{dsize}.tsv')
    create_logger('main',
                  VERSION=c.runtime.VERSION,
                  log_path=main_log_path,
                  FILE_HANDLER_LEVEL=DEBUG,
                  STREAM_HANDLER_LEVEL=DEBUG,
                  SLACK_HANDLER_LEVEL=INFO,
                  slackauth=c.slackauth
                  )
    create_logger('train',
                  VERSION=c.runtime.VERSION,
                  log_path=train_log_path,
                  FILE_HANDLER_LEVEL=DEBUG,
                  STREAM_HANDLER_LEVEL=DEBUG,
                  SLACK_HANDLER_LEVEL=INFO,
                  slackauth=c.slackauth
                  )
    logger = getLogger('main')
    logger.info(f':thinking_face: Starting experiment {c.runtime.VERSION}_{c.model.type}_{c.features}{dsize}')
    logger.info(f'Options indicated: {opt}')

    try:
        main(c)
        logger.info(f':sunglasses: Finished experiment {c.runtime.VERSION}{dsize}')
    except Exception:
        logger.critical(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        logger.critical(f':skull: Stopped experiment {c.runtime.VERSION}{dsize}')
