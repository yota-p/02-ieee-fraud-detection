import luigi
from abc import ABCMeta, abstractmethod


class AbstractTask(luigi.Task, metaclass=ABCMeta):
    '''
    Every tasks need to inherit this class.
    Since python doesn't have interface,
    abstract class is used to force implementing
    run and output method in every task class.
    If not overrided, this class will cause RuntimeError.
    '''

    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def run(self):
        pass
