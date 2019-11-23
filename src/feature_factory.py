from features.raw import Raw
from features.altgor import Altgor
from mylog import timer


class FeatureFactory:

    @timer
    def create(self, featurename):
        if featurename == 'raw':
            return Raw()
        elif featurename == 'altgor':
            return Altgor()
        else:
            return None
