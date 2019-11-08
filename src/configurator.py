# Get Config class dynamically from specified version
import sys
import pathlib
from get_option import get_option
from importlib import import_module


class Configurator:

    def get_config(self):
        configpath = pathlib.Path('src/config').resolve()
        sys.path.append(str(configpath))
        mod = import_module('config_' + get_option().version)
        return mod.Config()


config = Configurator().get_config()
