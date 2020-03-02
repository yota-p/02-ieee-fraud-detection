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
from model.model_factory import ModelFactory


@timer(INFO)
def main(c):
    dsize = '.small' if c.runtime.use_small_data is True else ''
    paths = EasyDict()
    scores = EasyDict()
    modelfactory = ModelFactory()

    with blocktimer('Preprocess', level=INFO):
        paths.out_train_path = f'data/feature/transformed_{c.runtime.version}_train{dsize}.pkl'
        paths.out_test_path = f'data/feature/transformed_{c.runtime.version}_test{dsize}.pkl'

        # read data
        train = pd.read_pickle(f'data/feature/{c.features[0]}_train.pkl')
        test = pd.read_pickle(f'data/feature/{c.features[0]}_test.pkl')

        if c.runtime.use_small_data:
            frac = 0.001
            train = train.sample(frac=frac, random_state=42)
            test = test.sample(frac=frac, random_state=42)
        logger.debug(f'Loaded feature {c.features[0]}')
        logger.debug(f'train.shape: {train.shape}, test.shape: {test.shape}')

        # separate into X, y
        X_train = train.drop('isFraud', axis=1)
        y_train = train['isFraud'].copy()
        X_test = test
        del train, test

    with blocktimer('Train', level=INFO):
        # col
        cols = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card5',
                'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain',
                'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5',
                'D10', 'D11', 'D15', 'M1', 'M2', 'M3', 'M4', 'M6', 'M7', 'M8',
                'M9', 'V1', 'V3', 'V4', 'V6', 'V8', 'V11', 'V13', 'V14', 'V17',
                'V20', 'V23', 'V26', 'V27', 'V30', 'V36', 'V37', 'V40', 'V41',
                'V44', 'V47', 'V48', 'V54', 'V56', 'V59', 'V62', 'V65', 'V67',
                'V68', 'V70', 'V76', 'V78', 'V80', 'V82', 'V86', 'V88', 'V89',
                'V91', 'V107', 'V108', 'V111', 'V115', 'V117', 'V120', 'V121',
                'V123', 'V124', 'V127', 'V129', 'V130', 'V136', 'V138', 'V139',
                'V142', 'V147', 'V156', 'V160', 'V162', 'V165', 'V166', 'V169',
                'V171', 'V173', 'V175', 'V176', 'V178', 'V180', 'V182', 'V185',
                'V187', 'V188', 'V198', 'V203', 'V205', 'V207', 'V209', 'V210',
                'V215', 'V218', 'V220', 'V221', 'V223', 'V224', 'V226', 'V228',
                'V229', 'V234', 'V235', 'V238', 'V240', 'V250', 'V252', 'V253',
                'V257', 'V258', 'V260', 'V261', 'V264', 'V266', 'V267', 'V271',
                'V274', 'V277', 'V281', 'V283', 'V284', 'V285', 'V286', 'V289',
                'V291', 'V294', 'V296', 'V297', 'V301', 'V303', 'V305', 'V307',
                'V309', 'V310', 'V314', 'V320', 'id_01', 'id_02', 'id_03', 'id_04',
                'id_05', 'id_06', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13',
                'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28',
                'id_29', 'id_31', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
                'DeviceInfo', 'cents', 'addr1_FE', 'card1_FE', 'card2_FE',
                'card3_FE', 'P_emaildomain_FE', 'card1_addr1',
                'card1_addr1_P_emaildomain', 'card1_addr1_FE',
                'card1_addr1_P_emaildomain_FE', 'TransactionAmt_card1_mean',
                'TransactionAmt_card1_std', 'TransactionAmt_card1_addr1_mean',
                'TransactionAmt_card1_addr1_std',
                'TransactionAmt_card1_addr1_P_emaildomain_mean',
                'TransactionAmt_card1_addr1_P_emaildomain_std', 'D9_card1_mean',
                'D9_card1_std', 'D9_card1_addr1_mean', 'D9_card1_addr1_std',
                'D9_card1_addr1_P_emaildomain_mean',
                'D9_card1_addr1_P_emaildomain_std', 'D11_card1_mean',
                'D11_card1_std', 'D11_card1_addr1_mean', 'D11_card1_addr1_std',
                'D11_card1_addr1_P_emaildomain_mean',
                'D11_card1_addr1_P_emaildomain_std', 'DT_M', 'uid_FE',
                'TransactionAmt_uid_mean', 'TransactionAmt_uid_std', 'D4_uid_mean',
                'D4_uid_std', 'D9_uid_mean', 'D9_uid_std', 'D10_uid_mean',
                'D10_uid_std', 'D15_uid_mean', 'D15_uid_std', 'C1_uid_mean',
                'C2_uid_mean', 'C4_uid_mean', 'C5_uid_mean', 'C6_uid_mean',
                'C7_uid_mean', 'C8_uid_mean', 'C9_uid_mean', 'C10_uid_mean',
                'C11_uid_mean', 'C12_uid_mean', 'C13_uid_mean', 'C14_uid_mean',
                'M1_uid_mean', 'M2_uid_mean', 'M3_uid_mean', 'M4_uid_mean',
                'M5_uid_mean', 'M6_uid_mean', 'M7_uid_mean', 'M8_uid_mean',
                'M9_uid_mean', 'uid_P_emaildomain_ct', 'uid_dist1_ct',
                'uid_DT_M_ct', 'uid_id_02_ct', 'uid_cents_ct', 'C14_uid_std',
                'uid_C13_ct', 'uid_V314_ct', 'uid_V127_ct', 'uid_V136_ct',
                'uid_V309_ct', 'uid_V307_ct', 'uid_V320_ct', 'outsider15']

        # CHRIS - TRAIN 75% PREDICT 25%
        idxT = X_train.index[:3*len(X_train)//4]
        idxV = X_train.index[3*len(X_train)//4:]

        # Predict test.csv
        import numpy as np
        import xgboost as xgb
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score
        oof = np.zeros(len(X_train))
        preds = np.zeros(len(X_test))

        skf = GroupKFold(n_splits=6)
        for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=X_train['DT_M'])):
            month = X_train.iloc[idxV]['DT_M'].iloc[0]
            logger.info(f'Fold {i} withholding month {month}')
            logger.info(f'rows of train ={len(idxT)}, rows of holdout ={len(idxV)}')
            clf = xgb.XGBClassifier(
                n_estimators=5000,
                max_depth=12,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.4,
                missing=-1,
                eval_metric='auc',
                # USE CPU
                # nthread=4,
                tree_method='hist'
                # USE GPU
                # tree_method='gpu_hist'
            )
            h = clf.fit(X_train[cols].iloc[idxT], y_train.iloc[idxT],
                        eval_set=[(X_train[cols].iloc[idxV], y_train.iloc[idxV])],
                        verbose=100, early_stopping_rounds=200)

            oof[idxV] += clf.predict_proba(X_train[cols].iloc[idxV])[:, 1]
            preds += clf.predict_proba(X_test[cols])[:, 1]/skf.n_splits
            del h, clf
        logger.info('#'*20)
        logger.info(f'XGB96 OOF CV= {roc_auc_score(y_train, oof)}')

        # Kaggle Submission File
        '''
        sample_submission = pd.read_csv('data/raw/sample_submission.csv')
        sample_submission.isFraud = preds
        sample_submission.to_csv('sub_xgb_96.csv', index=False)
        '''

        sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
        sub.TransactionID = X_test.reset_index()['TransactionID']
        sub.isFraud = preds
        # sub['TransactionID'] = X_test.reset_index()['TransactionID']

        paths.out_sub_path = f'data/submission/submission_{c.runtime.version}{dsize}.csv'
        sub.to_csv(paths.out_sub_path, index=False)

        '''
        model = modelfactory.create(c.model)
        model = model.train(X_train.loc[idxT, :], y_train[idxT],
                            X_train.loc[idxV, :], y_train[idxV],
                            num_boost_round=c.num_boost_round)
        importance = pd.DataFrame(model.feature_importance,
                                  index=X_train.columns,
                                  columns=['importance'])

        # save results
        paths.out_model_dir = f'data/model/model_{c.runtime.version}_{c.model.type}{dsize}.pkl'
        paths.importance_path = f'feature/importance/importance_{c.runtime.version}{dsize}.csv'
        model.save(paths.out_model_dir)
        importance.to_csv(paths.importance_path)
        '''

    '''
    with blocktimer('Predict', level=INFO):
        sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
        # sub['TransactionID'] = test['TransactionID']
        # sub['TransactionID'] = test.reset_index()['TransactionID']
        y_test = model.predict(X_test)
        sub['isFraud'] = y_test
        sub['TransactionID'] = X_test.reset_index()['TransactionID']

        paths.out_sub_path = f'data/submission/submission_{c.runtime.version}{dsize}.csv'
        sub.to_csv(paths.out_sub_path, index=False)
    '''

    result = EasyDict()
    result.update(c)
    result.scores = scores
    result.paths = paths
    return result


def split_X_y(train, test):
    '''
    X_train = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_train = train.sort_values('TransactionDT')['isFraud']
    X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
    '''
    X_train = train.drop('isFraud', axis=1)
    y_train = train['isFraud']
    X_test = test

    return X_train, y_train, X_test


@timer
def optimize_num_boost_round(model, X, y, n_splits, dsize, paths, scores) -> dict:
    '''
    Tune parameter num_boost_round
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
