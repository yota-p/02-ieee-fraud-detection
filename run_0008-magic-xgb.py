import gc
import warnings
import traceback
from logging import getLogger, DEBUG
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
# import lightgbm as lgb
import xgboost as xgb

from util.configure import Config
from util.seeder import seed_everything
from util.mylog import create_logger, timer, blocktimer
from util.transformer import Transformer
from model.base_trainer import BaseTrainer


warnings.filterwarnings('ignore')
config_mod = 'config.config_0008'


@timer
def main(c_runtime, c_transformer, c_model, c_trainer, c_log):

    with blocktimer('Preprocess'):
        train, test = Transformer.run(**c_transformer.__dict__)
        X_train, y_train, X_test = split_X_y(train, test)

    with blocktimer('Train'):
        # trainer = LGB_Trainer(c_trainer)
        trainer = XGB_Trainer(c_trainer)
        trainer.run(X_train, y_train)
        model = trainer.get_model()

    with blocktimer('Predict'):
        sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
        sub['TransactionID'] = test['TransactionID']
        y_test = model.predict_proba(X_test)[:, 1]
        sub['isFraud'] = y_test
        sub.to_csv(c_runtime.out_sub_path, index=False)
        logger.info(f'Saved {c_runtime.out_sub_path}')


@timer
def split_X_y(train, test):
    logger = getLogger('main')
    logger.debug(f'input train.shape: {train.shape}')
    logger.debug(f'input test.shape : {test.shape}')
    train.reset_index(inplace=True)
    train.set_index('TransactionID', drop=False, inplace=True)
    cols = train.columns.drop(['isFraud', 'TransactionDT', 'TransactionID'])
    X_train = train[cols]
    y_train = train['isFraud']

    test.reset_index(inplace=True)
    test.set_index('TransactionID', drop=False, inplace=True)
    X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)

    logger.debug(f'output X_train.shape: {X_train.shape}')
    logger.debug(f'output y_train.shape: {y_train.shape}')
    logger.debug(f'output X_test.shape : {X_test.shape}')

    return X_train, y_train, X_test


# LGB
# https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
# XGB
# https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600
# class LGB_Trainer(BaseTrainer):
class XGB_Trainer(BaseTrainer):
    @timer
    def train(self, X, y):
        # start train log
        create_logger('train', **c.log)
        logger_train = getLogger('train')
        logger_train.debug('{}\t{}\t{}\t{}'.format('fold', 'iteration', 'train_auc', 'eval_auc'))

        # clf = lgb.LGBMClassifier(**self.c.model.params)
        clf = xgb.XGBClassifier(**self.c.model.params)
        aucs = list()
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = X.columns

        # K-fold CV
        folds = TimeSeriesSplit(n_splits=self.c.n_splits)
        for fold, (idx_train, idx_val) in enumerate(folds.split(X, y)):

            with blocktimer(f'Training on fold {fold + 1}'):
                # prepare inputs for train
                X_train = X.iloc[idx_train]
                y_train = y.iloc[idx_train]
                X_val = X.iloc[idx_val]
                y_val = y.iloc[idx_val]
                callbacks = [self.log_evaluation(logger_train, period=10, fold=(fold+1))]

                # train
                clf.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_val, y_val)],
                        verbose=1000,
                        callbacks=callbacks)

                # record result
                feature_importances[f'fold_{fold + 1}'] = clf.feature_importances_
                # LGB
                # aucs.append(clf.best_score_['valid_1']['auc'])
                # XGB
                aucs.append(clf.best_score['valid_1']['auc'])

        logger.info('Training has finished.')
        logger.debug(f'Mean AUC: {np.mean(aucs)}')
        # TODO: save feature importance and other

        # TODO: remember or save models for each steps
        with blocktimer(f'Re-train with best parameter'):
            best_params = self.c.model.params
            del best_params['early_stopping_rounds']
            # LGB
            # best_iteration = clf.best_iteration_
            # XGB
            best_iteration = clf.best_iteration
            if best_iteration is not None:
                best_params['n_estimators'] = best_iteration
            else:
                logger.warn('Training did not converge. Try larger n_estimators.')
            clf = xgb.XGBClassifier(**best_params)
            clf.fit(X, y)

        self.model = clf

    '''
    # LGB
    def log_evaluation(self, logger, period=1, show_stdv=True, level=DEBUG, fold=1):
        def _callback(env):
            if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
                train_auc = env.evaluation_result_list[0][2]
                eval_auc = env.evaluation_result_list[1][2]
                logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}\t{train_auc:.6f}\t{eval_auc:.6f}')
        _callback.order = 10
        return _callback
    '''

    # XGB
    def log_evaluation(self, logger, period=1, show_stdv=True, level=DEBUG, fold=1):
        def _callback(env):
            if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
                # XGBClassifier.evaluation_result_list contains values as below
                # env.evaluation_result_list = [('validation_0-auc', 0.882833), ('validation_1-auc', 0.827249)]
                train_auc = env.evaluation_result_list[0][1]
                eval_auc = env.evaluation_result_list[1][1]
                logger.log(level, f'{fold:0>3}\t{env.iteration+1:0>6}\t{train_auc:.6f}\t{eval_auc:.6f}')
        _callback.order = 10
        return _callback


if __name__ == "__main__":
    gc.enable()
    c = Config()
    c.import_config_module(config_mod)
    c.apply_option()
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
