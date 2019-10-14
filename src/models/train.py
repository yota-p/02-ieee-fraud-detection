import luigi
from AbstractTask import AbstractTask
from models.lightgbm import model
from config import project


def train():
    # trainer = Trainer(Model())
    # trainer.train()
    model.train()


class Trainer():

    def __init__(self):
        pass


if __name__ == '__main__':
    train()

'''
class TaskTrain(AbstractTask):

    def run(self):
        train()

    def output(self):
        out_files = ['submission.csv']
        return luigi.LocalTarget(project.rootdir + 'data/processed/' + out_files[0])
'''
