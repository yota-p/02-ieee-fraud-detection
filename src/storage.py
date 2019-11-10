class Filenames:
    PATH1 = 'a'
    PATH2 = 'b'
    datas = {
            'dataname1': PATH1,
            'dataname2': PATH2
            }


def getPath(dataname, path):
    return path


def addPath(dataname, path):
    return


class Storage:
    def stage(self):
        '''
        Stage model file of experiment to staging folder
        '''
        pass

    def deploy(self):
        '''
        Deploy staged model to some dir
        '''

    def path(self):
        path = ''
        return path
