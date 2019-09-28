import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from config.tempconfig import LOCAL_TEST, TARGET, SEED


# TODO: move to config
lgb_params = {
                    'objective': 'binary',
                    'boosting_type': 'gbdt',
                    'metric': 'auc',
                    'n_jobs': -1,
                    'learning_rate': 0.01,
                    'num_leaves': 2**8,
                    'max_depth': -1,
                    'tree_learner': 'serial',
                    'colsample_bytree': 0.5,
                    'subsample_freq': 1,
                    'subsample': 0.7,
                    'n_estimators': 800,
                    'max_bin': 255,
                    'verbose': -1,
                    'seed': SEED,
                    'early_stopping_rounds': 100,
                }


def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):

    folds = GroupKFold(n_splits=NFOLDS)

    X, y = tr_df[features_columns], tr_df[target]
    P, P_y = tt_df[features_columns], tt_df[target]
    split_groups = tr_df['DT_M']

    tt_df = tt_df[['TransactionID', target]]
    predictions = np.zeros(len(tt_df))
    oof = np.zeros(len(tr_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
        print('Fold:', fold_)
        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

        print(len(tr_x), len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets=[tr_data, vl_data],
            verbose_eval=200,
        )

        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS

        oof_preds = estimator.predict(vl_x)
        oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X.columns)),
                                       columns=['Value', 'Feature'])
            print(feature_imp)

        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()

    tt_df['prediction'] = predictions
    print('OOF AUC:', metrics.roc_auc_score(y, oof))
    if LOCAL_TEST:
        print('Holdout AUC:', metrics.roc_auc_score(tt_df[TARGET], tt_df['prediction']))

    return tt_df
