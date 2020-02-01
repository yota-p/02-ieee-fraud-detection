from util.mylog import timer


@timer
def is_latest(pathlist):
    for path in pathlist:
        if not path.exists():
            return False
    return True
