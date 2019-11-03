from functools import wraps
from get_option import get_option
import time
from pathlib import Path
import slack
from slack_path import SlackAuth
from logging import getLogger, Formatter, FileHandler, StreamHandler
from logging import INFO, DEBUG
from configure import Configger


def create_main_logger(version, params=None):
    main_formatter = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")
    main_extension = ".log"
    __create_logger("main", version, main_formatter, main_extension, params)


def create_train_logger(version, params=None):
    train_formatter = Formatter()
    train_extension = ".tsv"
    __create_logger("train", version, train_formatter, train_extension, params)


def __create_logger(post_fix, version, formatter, extension, params):
    log_path = Path(__file__).parents[1] / "log" / post_fix
    Path.mkdir(log_path, exist_ok=True, parents=True)

    log_file = Path(log_path / (version + extension)).resolve()
    __init_logfile(params, log_file)

    if post_fix == "train":
        logger_ = get_training_logger(version)
    else:
        logger_ = getLogger(version)
    logger_.setLevel(DEBUG)

    file_handler = FileHandler(log_file)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(formatter)

    logger_.addHandler(file_handler)
    logger_.addHandler(stream_handler)


def get_training_logger(version):
    return getLogger(version + "train")


def __init_logfile(params, log_file):
    with open(log_file, "w") as f:
        if params is not None:
            f.write(params)
        else:
            pass


def stop_watch(*dargs, **dkargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            method_name = dargs[0]
            start = time.time()
            log = "[START]  {}".format(method_name)
            getLogger(Configger.get_config().runtime.VERSION).info(log)
            send_message(log)

            result = func(*args, **kargs)
            elapsed_time = int(time.time() - start)
            minits, sec = divmod(elapsed_time, 60)
            hour, minits = divmod(minits, 60)

            log = "[FINISH] {}: [elapsed_time] >> {:0>2}:{:0>2}:{:0>2}".format(method_name, hour, minits, sec)
            getLogger(Configger.get_config().runtime.VERSION).info(log)
            send_message(log)
            return result
        return wrapper
    return decorator


def send_message(text):
    if get_option().NoSendMessage:
        return
    token = read_token(SlackAuth.TOKEN_PATH)
    if token is None:
        return
    client = slack.WebClient(token)
    text = "[{}]: {}".format(Configger.get_config().runtime.VERSION, text)
    client.chat_postMessage(
        channel=SlackAuth.CHANNEL,
        text=text
    )
    del client


def read_token(token_path):
    token = ""
    if token_path.exists() is not True:
        return None
    with open(token_path, 'r') as f:
        token = f.read()
    token = token.replace('\n', '')
    return token
