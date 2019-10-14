import glob
import luigi
from luigi.util import requires
from save_log import stop_watch
from config import project
from exceptions import DuplicateVersionException
from AbstractTask import AbstractTask
from data.getrawdata import get_raw_data
from data.picklerawdata import pickle_raw_data
from data.preprocess import preprocess
from models.lightgbm import model


class Pipeline:

    def __init__(self, version):
        self.version = version
        self.__check_config(self.version)

    @stop_watch('Pipeline.run()')
    def run(self):
        luigi.run(['TaskTrain', '--workers', '1', '--local-scheduler'])

    def __check_config(self, version):
        version_file_list = glob.glob(project.rootdir + 'src/config/' + version + '*.py')
        if len(version_file_list) > 1:
            raise DuplicateVersionException()
        return version_file_list


class TaskGetRawData(AbstractTask):

    def run(self):
        get_raw_data()

    def output(self):
        out_files = ['sample_submission.csv',
                     'test_identity.csv',
                     'test_transaction.csv',
                     'train_identity.csv',
                     'train_transaction.csv']
        for file in out_files:
            yield luigi.LocalTarget(project.rootdir + 'data/raw/' + file)


@requires(TaskGetRawData)
class TaskPickleRawData(AbstractTask):

    def run(self):
        pickle_raw_data()

    def output(self):
        out_files = ['train_identity.pickle',
                     'train_transaction.pickle',
                     'test_identity.pickle',
                     'train_identity.pickle']
        for file in out_files:
            yield luigi.LocalTarget(project.rootdir + 'data/raw/' + file)


@requires(TaskPickleRawData)
class TaskPreprocess(AbstractTask):

    def run(self):
        preprocess()

    def output(self):
        out_files = ['X.pickle', 'X_test.pickle', 'y.pickle']
        for file in out_files:
            yield luigi.LocalTarget(project.rootdir + 'data/processed/' + file)


@requires(TaskPreprocess)
class TaskTrain(AbstractTask):

    def run(self):
        model.train()

    def output(self):
        out_files = ['submission.csv']
        return luigi.LocalTarget(project.rootdir + 'data/processed/' + out_files[0])


'''
@requires(TaskTrain)
class TaskPredict(luigi.Task):

    def output(self):
        return luigi.LocalTarget("data/TaskPredict.txt")

    def run(self):
        return None


@requires(TaskPredict)
class TaskSubmit(luigi.Task):

    def output(self):
        # dependencies = ["data/TaskSubmit.txt", "data/test.txt"]
        # for filename in dependencies:
        #    yield luigi.LocalTarget(filename)
        return None

    def run(self):
        return None
'''

if __name__ == '__main__':
    Pipeline().run()
