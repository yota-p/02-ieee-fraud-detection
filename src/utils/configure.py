# This script contains some 'black magics'.
# Reads config*.py, which contains several dict
# Generates Config object, which contains read dicts.
# Though dict object in Python is bit complicated to access.
# So dicts will be wrapped by class EasyDict().
from importlib import import_module
from argparse import ArgumentParser


class Config:
    '''
    Get Config class dynamically from specified version
    http://nullbyte.hatenablog.com/entry/2015/12/11/005246
    ---
    Class to contain configurations from file & options.
    This is an static class.
    To initialize: call set_parameter()
    To read: from configure import Config as c
    '''
    _confdict = {}

    def __getattr__(self, k):
        try:
            return self._confdict[k]
        except KeyError as e:
            raise KeyError(f'No such value in config : {k}') from e

    def __setattr__(self, k, v):
        self._confdict[k] = v

    @classmethod
    def load(cls, d):
        cls._confdict = EasyDict(d)

    @classmethod
    def import_config_module(cls, module_path):
        mod = import_module(module_path)

        cls.load({
            k: v for k, v in mod.__dict__.items()
            # private variables naming convension
            if not k.startswith("_")
            # system variables (python spec)
            and not k.startswith("__")
            })

    def apply_option(self):
        option = parse_option()
        self = override_config(self, option)


def parse_option():
    '''
    Get optional variables from command line arguments
    --
    References
    https://qiita.com/taashi/items/400871fb13df476f42d2
    https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0
    '''
    argparser = ArgumentParser()

    argparser.add_argument('--nomsg',
                           action='store_true',
                           help='Not sending message')

    argparser.add_argument('--small',
                           action='store_true',
                           help='Use small data set for debug')

    return argparser.parse_args()


def override_config(config, option):
    config.transformer.DEBUG_SMALL_DATA = option.small
    config.log.slackauth.NO_SEND_MESSAGE = option.nomsg


class EasyDict(dict):
    '''
    Dictionary object wrapper to make it accessible using dot(.)s.
    https://github.com/makinacorpus/easydict
    ---
    ex.
    dic1 = {'k1':v1, 'k2':{'k3':v2}}
    dic2 = EasyDict(dic1)
    # To get v1, v2:
    dic1['k1'] # => v1
    dic1['k2']['k3'] # => v2
    # This can be accessed by:
    dic2.k1 # => v1
    dic2.k2.k3 # => v2
    '''

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)
