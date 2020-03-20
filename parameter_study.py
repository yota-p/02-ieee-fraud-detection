import gc
import warnings
import traceback
import json
from logging import getLogger, INFO
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
import optuna

from util.easydict import EasyDict
from util.option import parse_option
from util.seeder import seed_everything
from util.mylog import create_logger, timer, blocktimer
from model.model_factory import ModelFactory


@timer(INFO)
def main(c):
    dsize = '.small' if c.runtime.use_small_data is True else ''
    paths = EasyDict()
    scores = EasyDict()
    result = EasyDict()
    result.update(c)
    modelfactory = ModelFactory()

    with blocktimer('Preprocess', level=INFO):
        paths.in_train_path = f'data/feature/{c.features[0]}_train.pkl'
        paths.in_test_path = f'data/feature/{c.features[0]}_test.pkl'
        train = pd.read_pickle(paths.in_train_path)
        test = pd.read_pickle(paths.in_test_path)
        logger.debug(f'Loaded feature {c.features[0]}')

        if c.runtime.use_small_data:
            frac = 0.001
            train = train.sample(frac=frac, random_state=42)
            test = test.sample(frac=frac, random_state=42)
        logger.debug(f'train.shape: {train.shape}, test.shape: {test.shape}')

        # Split into X, y
        X_train = train.drop('isFraud', axis=1)
        X_test = test
        y_train = train['isFraud'].copy(deep=True)
        del train, test

    with blocktimer('Optimize num_boost_round', level=INFO):
        if c.train.optimize_num_boost_round is True:
            # tune the model params
            model = modelfactory.create(c.model)
            best_iteration = optimize_num_boost_round(
                model,
                X_train[c.cols],
                y_train,
                c.train.n_splits,
                dsize,
                paths,
                scores)
        else:
            logger.debug('Skip optimization')
            best_iteration = c.train.num_boost_round

    with blocktimer('Optimize model params', level=INFO):
        if c.train.optimize_model_params is True:
            # define objective for optuna
            def objectives(trial):
                max_depth = trial.suggest_int('max_depth', 3, 12)
                params = {
                    'boosting_type': 'gbdt',
                    # num_leaves should be smaller than approximately 2^max_depth*0.75
                    'num_leaves': 2 ** max_depth * 3 // 4,
                    'max_depth': max_depth,
                    'learning_rate': 0.05,
                    'objective': 'binary',
                    'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e0),  # 0.03454472573214212,
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 1e0),  # 0.3899927210061127,
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 1e0),  # 0.6485237330340494,
                    'random_state': 42,
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),  # 106,
                    'metric': 'auc',
                    'max_bin': 255
                    }
                c.model.params = params

                # Train by 6-fold CV
                oof = np.zeros(len(X_train))
                preds = np.zeros(len(X_test))

                skf = GroupKFold(n_splits=6)
                for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=X_train['DT_M'])):
                    fold = i+1
                    month = X_train.iloc[idxV]['DT_M'].iloc[0]
                    model_fold_path = f'data/model/model_{c.runtime.version}_{c.model.type}_opt_fold{fold}{dsize}.pkl'
                    model = modelfactory.create(c.model)
                    logger.info(f'Fold {fold} withholding month {month}')
                    logger.info(f'rows of train= {len(idxT)}, rows of holdout= {len(idxV)}')

                    model = model.train(X_train[c.cols].iloc[idxT], y_train.iloc[idxT],
                                        X_train[c.cols].iloc[idxV], y_train.iloc[idxV],
                                        num_boost_round=best_iteration,
                                        early_stopping_rounds=c.train.early_stopping_rounds,
                                        # categorical_features=categorical_features,
                                        fold=i+1)

                    oof[idxV] = model.predict(X_train[c.cols].iloc[idxV])
                    preds += model.predict(X_test[c.cols])/skf.n_splits

                    paths.update({f'model_fold_{fold}_path': model_fold_path})
                    model.save(paths[f'model_fold_{fold}_path'])
                    del model
                score = roc_auc_score(y_train, oof)
                logger.info(f'Fold {fold} OOF cv= {score}')
                return score

            # run optimization
            opt = optuna.create_study(direction='maximize',
                                      study_name=f'parameter_study_0016{dsize}',
                                      storage=f'sqlite:///data/optimization/parameter_study_0016{dsize}.db',
                                      load_if_exists=True)
            opt.optimize(objectives, n_trials=20)
            trial = opt.best_trial
            logger.debug(f'Best trial: {trial.value}')
            logger.debug(f'Best params: {trial.params}')
            scores.best_trial = trial.value
            result.optimize = {}
            result.optimize.best_params = trial.params
        else:
            logger.debug('Skip optimization')

    with blocktimer('Train', level=INFO):
        if c.train.train_model:
            logger.debug(f'Now using the following {len(c.cols)} features.')
            logger.debug(f'{np.array(c.cols)}')

            oof = np.zeros(len(X_train))
            preds = np.zeros(len(X_test))

            skf = GroupKFold(n_splits=6)
            for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=X_train['DT_M'])):
                month = X_train.iloc[idxV]['DT_M'].iloc[0]
                logger.info(f'Fold {i+1} withholding month {month}')
                logger.info(f'rows of train ={len(idxT)}, rows of holdout ={len(idxV)}')

                '''
                categorical_features = ['ProductCD', 'M4',
                                        'card1', 'card2', 'card3', 'card5', 'card6',
                                        'addr1', 'addr2', 'dist1', 'dist2',
                                        'P_emaildomain', 'R_emaildomain',
                                        ]
                '''

                model = modelfactory.create(c.model)
                model = model.train(X_train[c.cols].iloc[idxT], y_train.iloc[idxT],
                                    X_train[c.cols].iloc[idxV], y_train.iloc[idxV],
                                    num_boost_round=best_iteration,
                                    early_stopping_rounds=c.train.early_stopping_rounds,
                                    # categorical_features=categorical_features,
                                    fold=i+1)

                oof[idxV] = model.predict(X_train[c.cols].iloc[idxV])
                preds += model.predict(X_test[c.cols])/skf.n_splits
                del model
            logger.info(f'OOF cv= {roc_auc_score(y_train, oof)}')
            paths.importance_path = f'feature/importance/importance_{c.runtime.version}{dsize}.csv'
            # model.save(paths.out_model_dir)
            '''
            importance = pd.DataFrame(model.feature_importance,
                                      index=X_train.columns,
                                      columns=['importance'])
            importance.to_csv(paths.importance_path)
            '''

    with blocktimer('Predict', level=INFO):
        if c.train.predict:
            sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
            sub['TransactionID'] = X_test.reset_index()['TransactionID']
            sub['isFraud'] = preds

            paths.out_sub_path = f'data/submission/submission_{c.runtime.version}{dsize}.csv'
            sub.to_csv(paths.out_sub_path, index=False)

    result.scores = scores
    result.paths = paths
    return result


@timer
def optimize_num_boost_round(model, X, y, n_splits, dsize, paths, scores) -> dict:
    '''
    Estimate best num_boost_round by early stopping
    '''
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


if __name__ == '__main__':
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
    logger_train = getLogger('train')
    logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'val_auc'))

    try:
        result = main(c)
        result.paths.main_log_path = main_log_path
        result.paths.train_log_path = train_log_path
        result.paths.result = f'config/result_{c.runtime.version}{dsize}.json'
        json.dump(result, open(result.paths.result, 'w'), indent=4)
        logger.info(f':sunglasses: Finished experiment {c.runtime.version}{dsize}')
    except Exception:
        logger.critical(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        logger.critical(f':skull: Stopped experiment {c.runtime.version}{dsize}')
