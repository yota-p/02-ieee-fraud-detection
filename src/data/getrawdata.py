from kaggle.api.kaggle_api_extended import KaggleApi
from config import project


def output():
    dataset = ['sample_submission.csv',
               'test_identity.csv',
               'test_transaction.csv',
               'train_identity.csv',
               'train_transaction.csv']
    dataset = [project.rootdir + 'data/raw/' + file for file in dataset]
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
