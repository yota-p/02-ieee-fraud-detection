# Get Config class dynamically from specified version
import sys
import pathlib
from get_option import get_option
from importlib import import_module


class Configger:

    def get_config():
        configpath = pathlib.Path('src/config').resolve()
        sys.path.append(str(configpath))
        print(str(configpath))
        mod = import_module('config_' + get_option().version)
        return mod.Config()
