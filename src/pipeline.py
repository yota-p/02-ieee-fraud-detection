import luigi
from luigi.util import requires
from save_log import stop_watch
from data import getrawdata
from data import preprocess
from models import train
import project


class Pipeline:

    @stop_watch('Pipeline.run()')
    def run(self):
        luigi.run(['TaskPreprocess', '--workers', '1', '--local-scheduler'])


class TaskGetRawData(luigi.Task):

    def output(self):
        return luigi.LocalTarget(getrawdata.output())

    def run(self):
        getrawdata.run()


@requires(TaskGetRawData)
class TaskPreprocess(luigi.Task):

    def output(self):
        return luigi.LocalTarget(preprocess.output())

    def run(self):
        return preprocess.run()


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
        #dependencies = ["data/TaskSubmit.txt", "data/test.txt"]
        #for filename in dependencies:
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


if __name__ == '__main__':
    Pipeline().run()
