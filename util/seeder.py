import os
import random
import numpy as np


def seed_everything(seed):
    '''
    Seed to make all processes deterministic
    '''
    # python hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
