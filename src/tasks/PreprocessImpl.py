import luigi
from tasks import AbstractTask
from data import preprocess


class TaskPreprocessImpl(AbstractTask):

    def output(self):
        return luigi.LocalTarget(preprocess.output())

    def run(self):
        return preprocess.run()
