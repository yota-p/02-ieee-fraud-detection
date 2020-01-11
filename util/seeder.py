import random
import os
import numpy as np


def seed_everything(seed=0):
    '''
    # Seeder
    : seed to make all processes deterministic     # type: int
    '''

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
