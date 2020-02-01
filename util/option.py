from argparse import ArgumentParser


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
                           default=False,
                           action='store_true',
                           help='Not sending message')

    argparser.add_argument('--small',
                           default=False,
                           action='store_true',
                           help='Use small data set for debug')

    option = argparser.parse_args()

    return option
