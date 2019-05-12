import datetime
import pickle
import os


def cache_write(file_path, data):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)


def cache_load(file_path):
    if not os.path.isfile(file_path):
        print('Warning: No such as file to Load')
        return None
    with open(file_path, 'rb') as fp:
        pp = pickle.load(fp)
    return pp
