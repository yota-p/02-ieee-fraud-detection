import gc
import warnings
from datetime import datetime
from save_log import create_main_logger, create_train_logger, send_message, stop_watch
warnings.filterwarnings('ignore')
from experiment import Experiment
from configurator import config as c


@stop_watch('main()')
def main():
    send_message(f':thinking_face: ============= {str(datetime.now())} ============= :thinking_face:')
    Experiment().run()


# Environment configuration
if __name__ == "__main__":
    gc.enable()
    create_main_logger(c.runtime.VERSION)
    create_train_logger(c.runtime.VERSION)
    main()
