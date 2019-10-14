import luigi
from tasks.AbstractTask import AbstractTask
from data import getrawdata


class GetRawDataImpl(AbstractTask):

    def output(self):
        return luigi.LocalTarget(getrawdata.output())

    def run(self):
        getrawdata.run()
