from functools import wraps
import time
from contextlib import contextmanager
from logging import getLogger, Formatter, FileHandler, StreamHandler, DEBUG, INFO
from logging.handlers import HTTPHandler
from pathlib import Path


class SlackHandler(HTTPHandler):
    def __init__(self, host, url, token, channel, username):
        super().__init__(host, url, method='POST', secure=True)
        self._token = token
        self._channel = channel
        self._username = username

    def mapLogRecord(self, record):
        ret = {
            "token": self._token,
            "channel": self._channel,
            "text": self.format(record),
            "username": self._username
        }
        return ret

    def setFormatter(self, formatter):
        self.formatter = formatter

    def setLevel(self, level):
        self.level = level


def create_logger(name,
                  version,
                  log_path,
                  slackauth,
                  no_send_message=False,
                  stream_handler_level=DEBUG,
                  file_handler_level=DEBUG,
                  slack_handler_level=INFO
                  ):
    '''
    This is a method to initialize logger for specified name.
    To initialize logger:
        create_logger('fizz', ...)
    To get logger:
        from logging import getLogger
        logger = getLogger('fizz')
    '''
    if name == 'main':
        formatter = Formatter(f'[{version}] %(asctime)s %(levelname)-5s > %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger_ = getLogger(name)
    elif name == 'train':
        formatter = Formatter()
        logger_ = getLogger(name)
    else:
        raise ValueError(f'Logger name {name} is not in menu.')

    logger_.setLevel(DEBUG)

    # for file
    file_handler = FileHandler(log_path, '+w')
    file_handler.setLevel(file_handler_level)
    file_handler.setFormatter(formatter)
    logger_.addHandler(file_handler)

    # for stream
    stream_handler = StreamHandler()
    stream_handler.setLevel(stream_handler_level)
    stream_handler.setFormatter(formatter)
    logger_.addHandler(stream_handler)

    # for slack
    if not no_send_message:
        token = __read_token(slackauth.token_path)
        slack_handler = SlackHandler(
            host=slackauth.host,
            url=slackauth.url,
            channel=slackauth.channel,
            token=token,
            username='LogBot')
        slack_handler.setLevel(slack_handler_level)
        slack_handler.setFormatter(formatter)
        logger_.addHandler(slack_handler)


def __read_token(token_path):
    token = None
    if not Path(token_path).exists():
        return None
    with open(token_path, 'r') as f:
        token = f.read()
    token = token.replace('\n', '')
    return token


def dynamic_args(func0):
    '''
    Decorator to allow defining decorators with & without arguments
    https://qiita.com/koyopro/items/8ce097b07605ee487ab2
    '''
    def wrapper(*args, **kwargs):
        if len(args) != 0 and callable(args[0]):
            # if func passed as first arg: treat as decorator without args
            func = args[0]
            return wraps(func)(func0(func))
        else:
            # treat as decorator with args
            def _wrapper(func):
                return wraps(func)(func0(func, *args, **kwargs))
            return _wrapper
    return wrapper


@dynamic_args
def timer(f, level=DEBUG):
    def _wrapper(*args, **kwargs):
        logger = getLogger('main')
        start = time.time()
        func_name = f.__qualname__
        logger.log(level, f'Start {func_name}')

        result = f(*args, **kwargs)

        elapsed_time = int(time.time() - start)
        minutes, sec = divmod(elapsed_time, 60)
        hour, minutes = divmod(minutes, 60)

        logger.log(level, f'End   {func_name}: [elapsed] >> {hour:0>2}:{minutes:0>2}:{sec:0>2}')
        return result
    return _wrapper


@contextmanager
def blocktimer(block_name, level=DEBUG):
    logger = getLogger('main')
    start = time.time()
    logger.log(level, f'Start {block_name}')

    yield

    elapsed_time = int(time.time() - start)
    minutes, sec = divmod(elapsed_time, 60)
    hour, minutes = divmod(minutes, 60)
    logger.log(level, f'End   {block_name}: [elapsed] >> {hour:0>2}:{minutes:0>2}:{sec:0>2}')
