import os
from configurator import config as c
from save_log import stop_watch


# Define datas as value object
class Raw:

    @stop_watch
    def __init__(self):
        os.system(f'kaggle competitions download -c {c.project.ID} -p {c.environment.ROOTPATH}data/raw/')
        os.system(f'unzip "{c.environment.ROOTPATH}data/raw/*.zip" -d {c.environment.ROOTPATH}data/raw')
        os.system(f'rm -f {c.environment.ROOTPATH}data/raw/*.zip')

    @stop_watch
    def load(self, dataname):
        return None
