import luigi
from tasks.AbstractTask import AbstractTask
from data import preprocess


class PreprocessImpl(AbstractTask):

    def output(self):
        return luigi.LocalTarget(preprocess.output())

    def run(self):
        return preprocess.run()
