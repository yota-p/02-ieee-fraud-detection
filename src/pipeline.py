import luigi
from luigi.util import requires
from save_log import stop_watch
from data import getrawdata


class Pipeline:

    @stop_watch('Pipeline.run()')
    def run(self):
        luigi.run(['TaskGoal', '--workers', '1', '--local-scheduler'])


class TaskStart(luigi.Task):

    def output(self):
        return None

    def run(self):
        return None


@requires(TaskStart)
class TaskGetRawData(luigi.Task):

    def output(self):
        return luigi.LocalTarget("TaskGetRawData.txt")

    def run(self):
        getrawdata.download()


@requires(TaskGetRawData)
class TaskPreprocess(luigi.Task):

    def output(self):
        return luigi.LocalTarget("TaskPreprocess.txt")

    def run(self):
        return None


@requires(TaskPreprocess)
class TaskLearn(luigi.Task):

    def output(self):
        return luigi.LocalTarget("TaskLearn.txt")

    def run(self):
        return None


@requires(TaskLearn)
class TaskPredict(luigi.Task):

    def output(self):
        return luigi.LocalTarget("TaskPredict.txt")

    def run(self):
        return None


@requires(TaskPredict)
class TaskSubmit(luigi.Task):

    def output(self):
        return luigi.LocalTarget("TaskSubmit.txt")

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
