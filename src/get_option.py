# Arguments
from argparse import ArgumentParser


def get_option():
    """
    Define options of command lines
    ---
    References
    https://qiita.com/taashi/items/400871fb13df476f42d2
    https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0
    """
    argparser = ArgumentParser()

    argparser.add_argument('version',
                           type=str,
                           help='Version ID')

    argparser.add_argument('--nomsg',
                           action='store_true',
                           help='Not sending message')

    argparser.add_argument('--dbg',
                           action='store_true',
                           help='Debug mode')

    argparser.add_argument('--pred',
                           action='store_true',
                           help='Executing prediction')

    argparser.add_argument('--predOnly',
                           action='store_true',
                           help='Executing ONLY prediction')

    argparser.add_argument('--trainOneRound',
                           action='store_true',
                           help='Training ONLY one round')

    argparser.add_argument('--dask',
                           action='store_true',
                           help='Using dask.dataframe.read_csv')

    argparser.add_argument('--nJobs',
                           type=int,
                           default=-1,
                           help='n_jobs for preprocess.')
    return argparser.parse_args()
