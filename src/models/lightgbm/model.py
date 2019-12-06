# https://www.kaggle.com/artgor/eda-and-models
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from logging import getLogger
from mylog import timer
logger_main = getLogger('main')
logger_train = getLogger('train')


class Model:
    def __init__(self, config):
        self.c = config

    @timer
    def train_model_classification(self,
                                   X, X_test, y,
                                   params,
                                   folds,
                                   model_type='lgb',
                                   eval_metric='auc',
                                   columns=None,
                                   plot_feature_importance=False,
                                   model=None,
                                   verbose=10000,
                                   early_stopping_rounds=200,
                                   n_estimators=50000,
                                   splits=None,
                                   n_folds=3,
                                   averaging='usual',
                                   n_jobs=-1):

        columns = X.columns if columns is None else columns
        n_splits = folds.n_splits if splits is None else n_folds
        X_test = X_test[columns]

        # to set up scoring parameters
        metrics_dict = {'auc': {'lgb_metric_name': self.eval_auc,
                                'catboost_metric_name': 'AUC',
                                'sklearn_scoring_function': metrics.roc_auc_score},
                        }

        result_dict = {}
        if averaging == 'usual':
            # out-of-fold predictions on train data
            oof = np.zeros((len(X), 1))

            # averaged predictions on train data
            prediction = np.zeros((len(X_test), 1))

        elif averaging == 'rank':
            # out-of-fold predictions on train data
            oof = np.zeros((len(X), 1))

            # averaged predictions on train data
            prediction = np.zeros((len(X_test), 1))

        # list of scores on folds
        scores = []
        feature_importance = pd.DataFrame()

        # split and train on folds
        for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
            print(f'Fold {fold_n + 1} started at {time.ctime()}')
            if type(X) == np.ndarray:
                X_train, X_valid = X[columns][train_index], X[columns][valid_index]
                y_train, y_valid = y[train_index], y[valid_index]
            else:
                X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
                y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            if model_type == 'lgb':
                model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs=n_jobs)
                model.fit(X_train, y_train,
                          eval_set=[(X_train, y_train), (X_valid, y_valid)
                                    ], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds)

                y_pred_valid = model.predict_proba(X_valid)[:, 1]
                y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

            if averaging == 'usual':

                oof[valid_index] = y_pred_valid.reshape(-1, 1)
                scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

                prediction += y_pred.reshape(-1, 1)

            elif averaging == 'rank':

                oof[valid_index] = y_pred_valid.reshape(-1, 1)
                scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

                prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)

            if model_type == 'lgb' and plot_feature_importance:
                # feature importance
                fold_importance = pd.DataFrame()
                fold_importance["feature"] = columns
                fold_importance["importance"] = model.feature_importances_
                fold_importance["fold"] = fold_n + 1
                feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        prediction /= n_splits

        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

        result_dict['oof'] = oof
        result_dict['prediction'] = prediction
        result_dict['scores'] = scores

        if model_type == 'lgb':
            if plot_feature_importance:
                feature_importance["importance"] /= n_splits
                cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                    by="importance", ascending=False)[:50].index

                result_dict['feature_importance'] = feature_importance
                result_dict['top_columns'] = cols

        return result_dict
