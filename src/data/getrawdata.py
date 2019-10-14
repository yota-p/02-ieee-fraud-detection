import os
from config import project


def get_raw_data():

    os.system('kaggle competitions download -c ' + project.id + ' -p ' + project.rootdir + 'data/raw/')
    os.system(f'unzip "{project.rootdir}data/raw/*.zip" -d {project.rootdir}data/raw')
    os.system(f'rm -f {project.rootdir}data/raw/*.zip')


if __name__ == '__main__':
    get_raw_data()
