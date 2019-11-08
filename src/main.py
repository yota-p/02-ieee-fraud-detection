import gc
import warnings
from datetime import datetime
from save_log import create_main_logger, create_train_logger, send_message, stop_watch, get_main_logger
warnings.filterwarnings('ignore')
from experiment import Experiment
from configurator import config as c
import traceback


@stop_watch
def main():
    Experiment().run()


if __name__ == "__main__":
    gc.enable()
    create_main_logger(c.runtime.VERSION)
    create_train_logger(c.runtime.VERSION)
    send_message(f':thinking_face: ============ {datetime.now():%Y-%m-%d %H:%M:%S} ============ :thinking_face:')
    try:
        main()
        send_message(f':sunglasses: ============ {datetime.now():%Y-%m-%d %H:%M:%S} ============ :sunglasses:')
    except Exception as e:
        logger = get_main_logger(c.runtime.VERSION)
        logger.exception(f'{e}')
        send_message(f':smiling_imp: Exception occured \n {traceback.format_exc()}')
        send_message(f':skull: ============ {datetime.now():%Y-%m-%d %H:%M:%S} ============ :skull:')
