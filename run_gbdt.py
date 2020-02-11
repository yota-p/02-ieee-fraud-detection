import gc
import warnings
import traceback
import json
from logging import getLogger, INFO
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

from util.easydict import EasyDict
from util.option import parse_option
from util.seeder import seed_everything
from util.mylog import create_logger, timer, blocktimer
from util.transformer import Transformer
from model.model_factory import ModelFactory


@timer(INFO)
def main(c):
    dsize = '.small' if c.runtime.use_small_data is True else ''
    paths = EasyDict()
    scores = EasyDict()

    with blocktimer('Preprocess', level=INFO):
        paths.out_train_path = f'data/feature/transformed_{c.runtime.version}_train{dsize}.pkl'
        paths.out_test_path = f'data/feature/transformed_{c.runtime.version}_test{dsize}.pkl'

        train, test = Transformer.run(c.features,
                                      c.runtime.use_small_data,
                                      paths.out_train_path,
                                      paths.out_test_path
                                      )
        X_train, y_train, X_test = split_X_y(train, test)
        test = test.sort_values('TransactionDT')

    with blocktimer('Optimize', level=INFO):
        modelfactory = ModelFactory()

        # tune the model params
        model = modelfactory.create(c.model)
        best_iteration = optimize_num_boost_round(
            model,
            X_train,
            y_train,
            c.train.n_splits,
            dsize,
            paths,
            scores)

    with blocktimer('Train', level=INFO):
        # train with best params, full data
        model = modelfactory.create(c.model)
        model = model.train(X_train, y_train, num_boost_round=best_iteration)
        importance = pd.DataFrame(model.feature_importance,
                                  index=X_train.columns,
                                  columns=['importance'])

        # save results
        out_model_dir = f'data/model/model_{c.runtime.version}_{c.model.type}{dsize}.pkl'
        model.save(out_model_dir)
        paths.update({'out_model_dir': out_model_dir})

        importance_path = f'feature/importance/importance_{c.runtime.version}{dsize}.csv'
        importance.to_csv(importance_path)

    with blocktimer('Predict', level=INFO):
        sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
        sub['TransactionID'] = test['TransactionID']
        y_test = model.predict(X_test)
        sub['isFraud'] = y_test

        out_sub_path = f'data/submission/submission_{c.runtime.version}{dsize}.csv'
        sub.to_csv(out_sub_path, index=False)
        paths.update({'out_sub_path': out_sub_path})

    result = EasyDict()
    result.update(c)
    result.scores = scores
    result.paths = paths
    return result


def split_X_y(train, test):
    X_train = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_train = train.sort_values('TransactionDT')['isFraud']
    X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

    return X_train, y_train, X_test


@timer
def optimize_num_boost_round(model, X, y, n_splits, dsize, paths, scores) -> dict:
    '''
    Tune parameter num_boost_round
    '''
    logger_train = getLogger('train')
    logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'val_auc'))

    # split data into train, validation
    folds = TimeSeriesSplit(n_splits=n_splits)
    for i, (idx_train, idx_val) in enumerate(folds.split(X, y)):
        fold = i + 1
        with blocktimer(f'Training on Fold {fold}', level=INFO):
            X_train = X.iloc[idx_train]
            y_train = y.iloc[idx_train]
            X_val = X.iloc[idx_val]
            y_val = y.iloc[idx_val]

            # train
            model = model.train(X_train, y_train,
                                X_val, y_val,
                                num_boost_round=c.train.num_boost_round,
                                early_stopping_rounds=c.train.early_stopping_rounds,
                                fold=fold)
            model_fold_path = f'data/model/model_{c.runtime.version}_{c.model.type}_fold{fold}{dsize}.pkl'
            paths.update({f'model_fold_{fold}_path': model_fold_path})
            model.save(paths[f'model_fold_{fold}_path'])

    # make optimal config from result
    if model.best_iteration > 0:
        logger.info(f'Early stopping. Best iteration is: {model.best_iteration}')
        scores.best_iteration = model.best_iteration
        return model.best_iteration
    else:
        logger.warn('Did not meet early stopping. Try larger num_boost_rounds.')
        scores.best_iteration = None
        return c.train.num_boost_round


if __name__ == "__main__":
    gc.enable()
    warnings.filterwarnings('ignore')

    # slack config
    slackauth = EasyDict(json.load(open('./slackauth.json', 'r')))
    slackauth.token_path = Path().home() / slackauth.token_file

    # get option
    opt = parse_option()
    c = json.load(open(f'config/config_{opt.version}.json'))
    c = EasyDict(c)
    c.runtime = {}
    c.runtime.version = opt.version
    c.runtime.use_small_data = opt.small
    c.runtime.no_send_message = opt.nomsg
    c.runtime.random_seed = opt.seed

    seed_everything(c.runtime.random_seed)

    dsize = '.small' if c.runtime.use_small_data is True else ''
    main_log_path = f'log/main_{c.runtime.version}{dsize}.log'
    train_log_path = f'log/train_{c.runtime.version}{dsize}.tsv'
    create_logger('main',
                  version=c.runtime.version,
                  log_path=main_log_path,
                  slackauth=slackauth,
                  no_send_message=c.runtime.no_send_message
                  )
    create_logger('train',
                  version=c.runtime.version,
                  log_path=train_log_path,
                  slackauth=slackauth,
                  no_send_message=c.runtime.no_send_message
                  )
    logger = getLogger('main')
    logger.info(f':thinking_face: Starting experiment {c.runtime.version}_{c.model.type}{dsize}')
    logger.info(f'Options indicated: {opt}')

    try:
        result = main(c)
        result.paths.main_log_path = main_log_path
        result.paths.train_log_path = train_log_path
        result.paths.result = f'config/result_{c.runtime.version}{dsize}.json'
        result.executed = True
        logger.info(f':sunglasses: Finished experiment {c.runtime.version}{dsize}')
    except Exception:
        result.executed = False
        logger.critical(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        logger.critical(f':skull: Stopped experiment {c.runtime.version}{dsize}')
    finally:
        json.dump(result, open(result.paths.result, 'w'), indent=4)
