import os
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SOURCE_DIR, 'data')
RESULT_DIR = os.path.join(SOURCE_DIR, 'result')
CACHE_DIR = os.path.join(SOURCE_DIR, 'cache')
LOG_DIR = os.path.join(SOURCE_DIR, 'log')

FIG_DIR = r"/home/starishlee/Documents/EE228@SJTU/project/fig"

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


mkdir(DATA_DIR)
mkdir(RESULT_DIR)
mkdir(CACHE_DIR)
mkdir(LOG_DIR)

TRAIN_LOGGER = os.path.join(LOG_DIR, 'train.log')

