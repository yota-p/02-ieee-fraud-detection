import gc
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import traceback
import pathlib
from src.utils import seeder

from configure import Config
from mylog import get_logger, create_main_logger, create_train_logger, send_message, timer
from experiment import Experiment


@timer
def main():
    Experiment().run()


if __name__ == "__main__":
    c = Config(config_dir=pathlib.Path('src/config'))
    gc.enable()
    seeder.seed_everything(c.runtime.RANDOM_SEED)
    create_main_logger(c.runtime.VERSION, c.storage.LOGDIR)
    create_train_logger(c.runtime.VERSION)
    send_message(f':thinking_face: ============ {datetime.now():%Y-%m-%d %H:%M:%S} ============ :thinking_face:')
    try:
        main()
        send_message(f':sunglasses: ============ {datetime.now():%Y-%m-%d %H:%M:%S} ============ :sunglasses:')
    except Exception as e:
        logger = get_logger('main')
        logger.exception(f'{e}')
        send_message(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        send_message(f':skull: ============ {datetime.now():%Y-%m-%d %H:%M:%S} ============ :skull:')
