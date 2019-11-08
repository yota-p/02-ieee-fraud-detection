from functools import wraps
import time
from pathlib import Path
import slack
from logging import getLogger, Formatter, FileHandler, StreamHandler
from logging import INFO, DEBUG
from configurator import config as c


def create_main_logger(version, params=None):
    main_formatter = Formatter('[%(levelname)s] %(asctime)s >>\t%(message)s')
    main_extension = '.log'
    __create_logger('main', version, main_formatter, main_extension, params)


def create_train_logger(version, params=None):
    train_formatter = Formatter()
    train_extension = '.tsv'
    __create_logger('train', version, train_formatter, train_extension, params)


def __create_logger(post_fix, version, formatter, extension, params):
    log_path = Path(__file__).parents[1] / 'log' / post_fix
    Path.mkdir(log_path, exist_ok=True, parents=True)

    log_file = Path(log_path / (version + extension)).resolve()
    __init_logfile(params, log_file)

    if post_fix == 'train':
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


def get_main_logger(version):
    return getLogger(version)


def get_training_logger(version):
    return getLogger(version + 'train')


def __init_logfile(params, log_file):
    with open(log_file, 'w') as f:
        if params is not None:
            f.write(params)
        else:
            pass


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        logger = getLogger(c.runtime.VERSION)
        method_name = func.__qualname__
        start = time.time()
        text = f'[START]  {method_name}'
        logger.info(text)
        send_message(text)

        result = func(*args, **kargs)
        elapsed_time = int(time.time() - start)
        minutes, sec = divmod(elapsed_time, 60)
        hour, minutes = divmod(minutes, 60)

        log = f'[FINISH] {method_name}: [elapsed_time] >> {hour:0>2}:{minutes:0>2}:{sec:0>2}'
        logger.info(log)
        send_message(log)
        return result
    return wrapper


def send_message(text):
    if c.runtime.NO_SEND_MESSAGE:
        return
    token = read_token(c.log.slackauth.TOKEN_PATH)
    if token is None:
        return
    client = slack.WebClient(token)
    text = f'[{c.runtime.VERSION}]: {text}'
    client.chat_postMessage(
        channel=c.log.slackauth.CHANNEL,
        text=text
    )
    del client


def read_token(token_path):
    token = ''
    if not token_path.exists():
        return None
    with open(token_path, 'r') as f:
        token = f.read()
    token = token.replace('\n', '')
    return token
