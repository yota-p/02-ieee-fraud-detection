from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
import competition


def download():

    # TODO: Fix this script
    # ISSUE: ieee-fraud-detection not found
    # ISSUE: which api to call?
    # ISSUE: how to specify download directory?
    pass
    # Competition = competition.Competition()
    # api = KaggleApi()
    # api.authenticate()
    # api.competitions_data_download_file(Competition.id, Competition.filename.encode('utf-8'))


if __name__ == '__main__':
    download()
