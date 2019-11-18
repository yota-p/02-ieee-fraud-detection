# Get Config class dynamically from specified version
import sys
from importlib import import_module
from argparse import ArgumentParser
import threading


# TODO: Config object to be singleton
class Config:
    '''
    Class to contain configurations from file & options
    To use : config = Config(config_dir, config_name, use_option).
    '''
    _instance = None
    _lock = threading.Lock()

    project = None
    runtime = None
    storage = None
    log = None
    experiment = None
    transformer = None
    model = None
    trainer = None
    modelapi = None

    def __new__(cls):
        '''
        Singleton
        '''
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_dir=None, config_name=None, use_option=False):
        '''
        Create Config() object from file and optional arguments.
        - config_dir    : directory of config file (pathlib.Path).
        - config_name   : config module (str). If not specified, set from option.
        - use_option    : if True, override file by option (boolean).
        '''
        config_from_file = self._read_config(config_dir, config_name)
        if use_option:
            option = self._argparse_option()
        self._copy_config(config_from_file)
        self._override_config(option)

    def _read_config(self, config_dir, config_name=None):
        '''
        Get config from file and set to self._config
        '''
        sys.path.append(str(config_dir.resolve()))
        if config_name is None:
            mod = import_module('config_' + self._argparse_option().version)
        else:
            mod = import_module(config_name)
        return mod.Config()

    def _copy_config(self, c):
        '''
        Copy each sub-config attributes from file to Config() object.
        '''
        self.project = c.project
        self.runtime = c.runtime
        self.storage = c.storage
        self.log = c.log
        self.experiment = c.experiment
        self.transformer = c.transformer
        self.model = c.model
        self.trainer = c.trainer
        self.modelapi = c.modelapi

    def _parse_option(self):
        '''
        Get optional variables from command line aruguments
        and returns class with attributes
        --
        References
        https://qiita.com/taashi/items/400871fb13df476f42d2
        https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0
        '''
        argparser = ArgumentParser()

        argparser.add_argument('version',
                               type=str,
                               help='Version ID')

        argparser.add_argument('--nomsg',
                               action='store_true',
                               help='Not sending message')

        argparser.add_argument('--debug',
                               action='store_true',
                               help='Debug mode')

        argparser.add_argument('--pred',
                               action='store_true',
                               help='Executing train and prediction')

        argparser.add_argument('--predOnly',
                               action='store_true',
                               help='Executing prediction only')

        argparser.add_argument('--nJobs',
                               type=int,
                               default=-1,
                               help='n_jobs for preprocess.')

        return argparser.parse_args()

    def _override_config(self, option):
        self.runtime.VERSION = option.version
        self.runtime.DEBUG = option.debug
        self.log.slackauth.NO_MSG = option.nomsg
        if option.pred:
            self.experiment.RUN_TRAIN = True
            self.experiment.RUN_PRED = True
        if option.predOnly:
            self.experiment.RUN_TRAIN = False
            self.experiment.RUN_PRED = True
