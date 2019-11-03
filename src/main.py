import gc
import warnings
from datetime import datetime
from save_log import create_main_logger, create_train_logger, send_message, stop_watch
warnings.filterwarnings('ignore')
# from pipeline import Pipeline
from experiment import Experiment
from configure import Configger


@stop_watch('main()')
def main(config):
    send_message(f':thinking_face: ============= {str(datetime.now())} ============= :thinking_face:')
    experiment = Experiment(config)
    experiment.run()


# Environment configuration
if __name__ == "__main__":
    gc.enable()
    config = Configger.get_config()
    create_main_logger(config.VERSION)
    create_train_logger(config.VERSION)
    main(config)
