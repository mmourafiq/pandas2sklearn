import os
import pickle
import pandas as pd


def split_file(file_path, chunksize):
    """
    Splits a file into multiple file with a size equal to chunksize.
    :param file_name (str): the file to split (should be csv file).
    :param chunksize (int): the size of individual parts.
    """
    reader = pd.read_csv(file_path, chunksize=chunksize)
    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    base_file_name = file_name.split('.csv')[0]
    count = 0
    for chunk in reader:
        chunk.to_csv('{}/.parts/{}-{}.csv'.format(dir_name, base_file_name, count))
        count += 1


def dump_clf(clf, clf_name):
    """
    Dumps a classifier with a given name.
    :param clf: scikit learn classifier.
    :param clf_name (str): used to save a `clf_name.dat` file.
    """
    pickle.dump(clf, open('{}.dat'.format(clf_name), 'w'))


def load_clf(clf_name):
    """
    Loads a classifier.
    :param clf_name (str): the classifier name to load.
    :return: loaded classifier.
    """
    return pickle.load(open('{}.dat'.format(clf_name), 'r'))
