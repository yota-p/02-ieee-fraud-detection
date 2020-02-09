from argparse import ArgumentParser


def parse_option():
    argparser = ArgumentParser()

    argparser.add_argument('version',
                           help='Version of config')

    argparser.add_argument('--nomsg',
                           default=False,
                           action='store_true',
                           help='Not sending message')

    argparser.add_argument('--small',
                           default=False,
                           action='store_true',
                           help='Use small data set for debug')

    argparser.add_argument('--seed',
                           default=42,
                           type=int,
                           help='Use small data set for debug')

    args = argparser.parse_args()

    return args
