from functools import wraps
import time
from pathlib import Path
import slack
from logging import getLogger, Formatter, FileHandler, StreamHandler
from logging import INFO, DEBUG
from configurator import Config
c = Config()


def get_logger(type):
    return getLogger(type)


def create_main_logger(version, dir):
    formatter = Formatter('[%(levelname)s] %(asctime)s >>\t%(message)s')
    extension = '.log'
    __create_logger('main', version, dir, formatter, extension)


def create_train_logger(version, dir):
    formatter = Formatter()
    extension = '.tsv'
    __create_logger('train', version, dir, formatter, extension)


def __create_logger(post_fix, version, dir, formatter, extension):
    Path.mkdir(dir, exist_ok=True, parents=True)

    log_file = Path(dir / (version + extension)).resolve()

    if post_fix == 'train':
        logger_ = getLogger('train')
    elif post_fix == 'main':
        logger_ = getLogger('main')
    else:
        raise Exception

    logger_.setLevel(DEBUG)

    file_handler = FileHandler(log_file)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(formatter)

    logger_.addHandler(file_handler)
    logger_.addHandler(stream_handler)


def timer(func):
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
    token = __read_token(c.log.slackauth.TOKEN_PATH)
    if token is None:
        return
    client = slack.WebClient(token)
    text = f'[{c.runtime.VERSION}]: {text}'
    client.chat_postMessage(
        channel=c.log.slackauth.CHANNEL,
        text=text
    )
    del client


def __read_token(token_path):
    token = ''
    if not token_path.exists():
        return None
    with open(token_path, 'r') as f:
        token = f.read()
    token = token.replace('\n', '')
    return token
