import luigi
from luigi.util import requires
from save_log import stop_watch
from models import train
from tasks.GetRawDataImpl import GetRawDataImpl
from tasks.PreprocessImpl import PreprocessImpl


class Pipeline:

    @stop_watch('Pipeline.run()')
    def run(self):
        luigi.run(['GetRawData', '--workers', '1', '--local-scheduler'])


class GetRawData(GetRawDataImpl):
    pass


@requires(GetRawData)
class Preprocess(PreprocessImpl):
    pass


'''
@requires(TaskPreprocess)
class TaskTrain(luigi.Task):

    def output(self):
        return luigi.LocalTarget("data/TaskLearn.txt")

    def run(self):
        return train.run()


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


@requires(TaskSubmit)
class TaskGoal(luigi.Task):

    def output(self):
        return None

    def run(self):
        return None
'''

if __name__ == '__main__':
    Pipeline().run()
