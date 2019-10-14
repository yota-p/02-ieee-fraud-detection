from pipeline import Pipeline
from save_log import stop_watch


class Manager:

    @stop_watch("Manager.run()")
    def run(self):
        pipeline = Pipeline()
        pipeline.run()
