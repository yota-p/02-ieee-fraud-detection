from save_log import timer, create_train_logger
from config import config as c


class Trainer:

    def __init__(self):
        create_train_logger(c.runtime.VERSION)

    @timer
    def train(self):
        return None
