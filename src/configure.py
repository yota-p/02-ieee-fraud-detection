# Get Config class dynamically from specified version
import sys
from importlib import import_module
from argparse import ArgumentParser


# TODO: Config object to be singleton
class Config:
    '''
    Class to contain configurations from file & options.
    This is an static class.
    To initialize: call set_parameter()
    To read: from configure import Config as c
    '''
    project = None
    runtime = None
    storage = None
    log = None
    experiment = None
    transformer = None
    trainer = None
    modelapi = None

    @classmethod
    def set_parameter(cls, config_dir=None, config_name=None, use_option=False):
        '''
        Set parameters to Config() object from file and optional arguments.
        - config_dir    : directory of config file (pathlib.Path).
        - config_name   : config module (str). If not specified, set from option.
        - use_option    : if True, override file by option (boolean).
        '''
        config_from_file = cls._read_config(config_dir, config_name)
        cls._copy_config(config_from_file)
        if use_option:
            option = cls._parse_option()
            cls._override_config(option)

    @classmethod
    def _read_config(cls, config_dir, config_name=None):
        '''
        Get config from file and set to class variables
        '''
        sys.path.append(str(config_dir.resolve()))
        if config_name is None:
            mod = import_module('config_' + cls._parse_option().version)
        else:
            mod = import_module(config_name)
        return mod.Config()

    @classmethod
    def _copy_config(cls, c):
        '''
        Copy each sub-config attributes from file to Config() object.
        '''
        cls.project = c.project
        cls.runtime = c.runtime
        cls.storage = c.storage
        cls.log = c.log
        cls.experiment = c.experiment
        cls.transformer = c.transformer
        cls.trainer = c.trainer
        cls.modelapi = c.modelapi

    @classmethod
    def _parse_option(cls):
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

        argparser.add_argument('--debug', action='store_true',
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

    @classmethod
    def _override_config(cls, option):
        cls.runtime.VERSION = option.version
        cls.log.slackauth.NO_SEND_MESSAGE = option.nomsg
        cls.runtime.DEBUG = option.debug
        if option.pred:
            cls.experiment.RUN_TRAIN = True
            cls.experiment.RUN_PRED = True
        if option.predOnly:
            cls.experiment.RUN_TRAIN = False
            cls.experiment.RUN_PRED = True
        cls.runtime.N_JOBS = option.nJobs
