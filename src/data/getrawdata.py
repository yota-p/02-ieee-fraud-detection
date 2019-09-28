from kaggle.api.kaggle_api_extended import KaggleApi
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
import project


def output():
    dataset = ['sample_submission.csv.zip',
               'test_identity.csv.zip',
               'test_transaction.csv.zip',
               'train_identity.csv.zip',
               'train_transaction.csv.zip']
    return dataset


def run():

    # TODO: Fix this script
    # ISSUE: ieee-fraud-detection not found
    # ISSUE: which api to call?
    # ISSUE: how to specify download directory?
    pass
    # compeinfo = competition.Info()
    # api = KaggleApi()
    # api.authenticate()
    # print(api.competitions_data_list_files('ieee-fraud-detection'))
    # api.competitions_data_download_file('ieee-fraud-detection', 'train_transaction.csv'.encode('utf-8'))
    # for filename in compeinfo.dataset:
    #     api.competitions_data_download_file(compeinfo.id, filename.encode('utf-8'))


if __name__ == '__main__':
    run()
