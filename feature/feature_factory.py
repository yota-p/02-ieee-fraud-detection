from feature.raw import Raw
from feature.altgor import Altgor
from feature.nroman import Nroman
from feature.magic import Magic
from util.mylog import timer


class FeatureFactory:

    @timer
    def create(self, featurename):
        if featurename == 'raw':
            return Raw()
        elif featurename == 'altgor':
            return Altgor()
        elif featurename == 'nroman':
            return Nroman()
        elif featurename == 'magic':
            return Magic()
        else:
            raise ValueError(f'Feature {featurename} does not exist in factory menu')
