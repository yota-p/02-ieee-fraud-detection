from pathlib import Path


class SlackAuth():
    CHANNEL = "02-ieee-fraud-detection"
    TOKEN_FILE = ".slack_token"
    TOKEN_PATH = Path(__file__).absolute().parents[1] / TOKEN_FILE
