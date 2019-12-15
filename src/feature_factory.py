from features.raw import Raw
from features.altgor import Altgor
from features.nroman import Nroman
from utils.mylog import timer


class FeatureFactory:

    @timer
    def create(self, featurename):
        if featurename == 'raw':
            return Raw()
        elif featurename == 'altgor':
            return Altgor()
        elif featurename == 'nroman':
            return Nroman()
        else:
            raise ValueError('{featurename} does not exist in factory menu')
