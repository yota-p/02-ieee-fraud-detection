import pickle


def dump(obj, filename, mode):

    with open(filename, mode=mode) as f:
        pickle.dump(obj, f)
