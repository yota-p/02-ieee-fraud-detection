import luigi
from tasks import AbstractTask
from data import getrawdata


class TaskGetRawDataImpl(AbstractTask.AbstractTask):

    def output(self):
        return luigi.LocalTarget(getrawdata.output())

    def run(self):
        getrawdata.run()
