'''
Script for tuning model hyper parameters.
'''
import gc
import warnings
import traceback
import json
from logging import getLogger, INFO
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
import optuna
import mlflow
from functools import partial

from util.easydict import EasyDict
from util.option import parse_option
from util.seeder import seed_everything
from util.mylog import create_logger, timer, blocktimer
from model.model_factory import ModelFactory

experiment_type = 'tune_hyper_params'


def objective(trial, X_train, y_train, X_test, cols, c):
    '''
    Define objectives for optuna
    '''
    modelfactory = ModelFactory()
    if c.model.type == 'lightgbm':
        max_depth = trial.suggest_int('max_depth', 3, 12)
        params_to_tune = {
            # num_leaves should be smaller than approximately 2^max_depth*0.75
            'num_leaves': 2 ** max_depth * 3 // 4,
            'max_depth': max_depth,
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 1e0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 1e0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0, 1),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0, 1)
            }
    elif c.model.type == 'xgboost':
        params_to_tune = {
            'min_split_loss': trial.suggest_loguniform('min_split_loss', 1e-3, 1e0),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e0),
            'subsample': trial.suggest_uniform('subsample', 0, 1),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.0, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1e0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1e0)
            }
    elif c.model.type == 'catboost':
        max_depth = trial.suggest_int('max_depth', 3, 12)
        params_to_tune = {
            # num_leaves should be smaller than approximately 2^max_depth*0.75
            # 'num_leaves': 2 ** max_depth * 3 // 4,
            'max_depth': max_depth,
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 1e0)
            }

    # apply suggested params
    params = c.model.params.copy()
    params.update(params_to_tune)

    # Train by 6-fold CV
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))
    skf = GroupKFold(n_splits=6)
    for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=X_train['DT_M'])):
        fold = i + 1
        month = X_train.iloc[idxV]['DT_M'].iloc[0]
        model_fold_path = f'data/model/model_{c.runtime.version}_opt_fold{fold}{c.runtime.dsize}.pkl'
        model = modelfactory.create(c.model)
        logger.info(f'Fold {fold} withholding month {month}')
        logger.info(f'rows of train= {len(idxT)}, rows of holdout= {len(idxV)}')

        model = model.train(X_train[cols].iloc[idxT], y_train.iloc[idxT],
                            X_train[cols].iloc[idxV], y_train.iloc[idxV],
                            params=params,
                            num_boost_round=c.train.num_boost_round,
                            early_stopping_rounds=c.train.early_stopping_rounds,
                            fold=i+1)

        oof[idxV] = model.predict(X_train[cols].iloc[idxV])
        preds += model.predict(X_test[cols]) / skf.n_splits

        r.paths.update({f'model_fold_{fold}_path': model_fold_path})
        model.save(r.paths[f'model_fold_{fold}_path'])
        del model

    score = roc_auc_score(y_train, oof)
    logger.info(f'Fold {fold} OOF cv= {score}')
    mlflow.log_metric('oof_cv_score', score, step=trial.number)
    return score


@timer(INFO)
def main(c, r):
    r.scores = {}

    with blocktimer('Preprocess', level=INFO):
        # unpack feature set list. set[i]={name: cols}
        for name, col_list in c.feature.set.items():
            in_train_path = f'data/feature/{name}_train.pkl'
            in_test_path = f'data/feature/{name}_test.pkl'
            cols = col_list
            train = pd.read_pickle(in_train_path)
            test = pd.read_pickle(in_test_path)
            logger.debug(f'Loaded feature {name}')

        if c.runtime.use_small_data:
            frac = 0.001
            train = train.sample(frac=frac, random_state=42)
            test = test.sample(frac=frac, random_state=42)

        logger.debug(f'train.shape: {train.shape}, test.shape: {test.shape}')

        # Split into X, y
        X_train = train.drop(c.feature.target, axis=1)
        y_train = train[c.feature.target].copy(deep=True)
        X_test = test
        del train, test

    with blocktimer('Tune hyper params', level=INFO):
        '''
        Run optimization
        '''
        mlflow.log_param('type', c.model.type)
        mlflow.log_param('num_boost_round', c.train.num_boost_round)
        mlflow.log_param('early_stopping_rounds', c.train.early_stopping_rounds)

        f = partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, cols=cols, c=c)
        opt = optuna.create_study(
            direction='maximize',
            study_name=f'{experiment_type}_{c.runtime.version}{c.runtime.dsize}',
            storage=f'sqlite:///data/optimization/{experiment_type}_{c.runtime.version}{c.runtime.dsize}.db',
            load_if_exists=True)
        opt.optimize(f, n_trials=c.optimize.n_trials)
        trial = opt.best_trial

        r.optimize = {}
        r.scores.best_trial = trial.number
        r.scores.best_score = trial.value
        r.optimize.best_params = trial.params
        tuned_params = c.model.params.copy()
        tuned_params.update(trial.params)
        r.model.tuned_params = tuned_params

        logger.debug(f'Best trial: {trial.number}')
        logger.debug(f'Best score: {trial.value}')
        logger.debug(f'Best params: {trial.params}')

        mlflow.log_metric('best_trial', trial.number)
        mlflow.log_metric('best_score', trial.value)
        mlflow.log_params(trial.params)

        return r


if __name__ == '__main__':
    gc.enable()
    warnings.filterwarnings('ignore')

    # get option
    opt = parse_option()
    # c is for config
    c = json.load(open(f'config/config_{opt.version}.json'))
    c = EasyDict(c)
    c.runtime = {}
    c.runtime.version = opt.version
    c.runtime.use_small_data = opt.small
    c.runtime.no_send_message = opt.nomsg
    c.runtime.random_seed = opt.seed
    c.runtime.dsize = '.small' if c.runtime.use_small_data is True else ''

    # dict to save results
    r = EasyDict()
    r.update(c)
    r.paths = {}
    r.paths.result = f'config/result_{c.runtime.version}{c.runtime.dsize}.json'

    seed_everything(c.runtime.random_seed)

    # check config.experiment_type
    if c.experiment_type != experiment_type:
        raise ValueError(f'experiment_type in config: {c.experiment_type} does not match this script')

    # start logging
    r.paths.main_log_path = f'log/main_{c.runtime.version}{c.runtime.dsize}.log'
    r.paths.train_log_path = f'log/train_{c.runtime.version}{c.runtime.dsize}.tsv'
    slackauth = EasyDict(json.load(open('./slackauth.json', 'r')))
    slackauth.token_path = Path().home() / slackauth.token_file
    create_logger('main',
                  version=c.runtime.version,
                  log_path=r.paths.main_log_path,
                  slackauth=slackauth,
                  no_send_message=c.runtime.no_send_message
                  )
    create_logger('train',
                  version=c.runtime.version,
                  log_path=r.paths.train_log_path,
                  slackauth=slackauth,
                  no_send_message=c.runtime.no_send_message
                  )
    logger = getLogger('main')
    logger.info(f':thinking_face: Starting experiment {c.runtime.version}_{c.model.type}{c.runtime.dsize}')
    logger.info(f'Options indicated: {opt}')
    '''
    logger_train = getLogger('train')
    logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'val_auc'))
    '''

    # start mlflow
    mlflow.set_experiment(f'tune_hyper_params{c.runtime.dsize}')
    mlflow.start_run(run_name=c.runtime.version)

    try:
        r = main(c, r)
        json.dump(r, open(r.paths.result, 'w'), indent=4)
        logger.info(f':sunglasses: Finished experiment {c.runtime.version}{c.runtime.dsize}')
        for k, v in r.paths.items():
            mlflow.log_artifact(v)

    except Exception:
        logger.critical(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        logger.critical(f':skull: Stopped experiment {c.runtime.version}{c.runtime.dsize}')
