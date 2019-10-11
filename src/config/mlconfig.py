from src.utils import seeder

SEED = 42
seeder.seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'
# START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
