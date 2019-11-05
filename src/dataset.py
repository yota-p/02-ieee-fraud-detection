import os
from configurator import config as c


class raw:
    def __init__():
        os.system(f'kaggle competitions download -c {c.project.id} -p {c.project.rootdir}data/raw/')
        os.system(f'unzip "{c.project.rootdir}data/raw/*.zip" -d {c.project.rootdir}data/raw')
        os.system(f'rm -f {c.project.rootdir}data/raw/*.zip')

    def load(dataname):
        return None

    def done():
        return False
