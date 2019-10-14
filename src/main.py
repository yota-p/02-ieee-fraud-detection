import gc
import warnings
from datetime import datetime
from logging import getLogger
from get_option import get_option
from save_log import create_main_logger, create_train_logger, get_version, send_message, stop_watch
from exceptions import DuplicateVersionException, IrregularArgumentException, IrregularCalcBackException
warnings.filterwarnings('ignore')
from pipeline import Pipeline


@stop_watch('main()')
def main(args):
    send_message(":thinking_face: ============= {} ============= :thinking_face:".format(str(datetime.now())))
    pipeline = Pipeline(version)
    pipeline.run()


if __name__ == "__main__":
    gc.enable()
    version = get_version()
    create_main_logger(version)
    create_train_logger(version)
    try:
        main(get_option())
    except DuplicateVersionException:
        send_message(":stop: Duplicate Version Exception Occurred.")
        getLogger(version).exception("Duplicate Version Exception Occurred.")
    except IrregularArgumentException:
        send_message(":stop: Irregular Argument for Feature Extraction.")
        getLogger(version).exception("Irregular Argument for Feature Extraction.")
    except IrregularCalcBackException:
        send_message(":stop: Irregular Dataframe back.")
        getLogger(version).exception("Irregular Dataframe back.")
    except Exception:
        send_message(":stop: Unexpected Exception Occurred.")
        getLogger(version).exception("Unexpected Exception Occurred.")
