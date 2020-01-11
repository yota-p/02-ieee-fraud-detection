import gc
import warnings
import traceback
import pandas as pd
from logging import getLogger

from util.configure import Config
from util.seeder import seed_everything
from util.mylog import timer, create_logger
from util.transformer import Transformer
from model.lgb_trainer import LGB_Trainer
from model.lgb_modelapi import LGB_ModelAPI

warnings.filterwarnings('ignore')
config_mod = 'config.config_0006_nroman_lgb'


class Experiment:
    def __init__(self, config):
        self.c = config

    @timer
    def run(self):
        train, test = Transformer.run(**self.c.transformer.__dict__)

        if self.c.runtime.RUN_TRAIN:
            trainer = LGB_Trainer(self.c.trainer)
            trained_model = trainer.run(train)
            del train, trainer

        if self.c.runtime.RUN_PRED:
            modelapi = LGB_ModelAPI(self.c.modelapi)
            modelapi.set_model(trained_model)

            sub = pd.DataFrame(columns=['TransactionID', 'isFraud'])
            sub['TransactionID'] = test['TransactionID']

            sub['isFraud'] = modelapi.run(test)  # returns y
            sub.to_csv(self.c.runtime.ROOTDIR / f'data/submission/{self.c.runtime.VERSION}_submission.csv',
                       index=False)  # TODO: set VERSION!
            del modelapi


if __name__ == "__main__":
    gc.enable()
    c = Config()
    c.import_config_module(config_mod)
    c.apply_option()
    seed_everything(c.runtime.RANDOM_SEED)
    create_logger('main', c.log)
    create_logger('train', c.log)
    logger = getLogger('main')
    logger.info(f':thinking_face: Starting experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}')

    try:
        experiment = Experiment(c)
        experiment.run()
        logger.info(f':sunglasses: Finished experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}')
    except Exception:
        logger.critical(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        logger.critical(f':skull: Stopped experiment {c.runtime.VERSION}_{c.runtime.DESCRIPTION}')
