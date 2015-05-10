from collections import namedtuple
import pandas as pd
import numpy as np
from sklearn import preprocessing


ALL = 'ALL'
INCLUDE = 'INCLUDE'
EXCLUDE = 'EXCLUDE'


DATA_SET = namedtuple(
    'DataSet',
    'feature_names data target_names target has_target ids has_ids')

DATA_SET_TRANSFORMED = namedtuple(
    'DataSetTransformed',
    'data target_names target has_target ids has_ids')


def clean_id_column(target_column):
    if not isinstance(target_column, (int, str, unicode)):
        return None
    return target_column


def get_columns(data_frame, target_column=None, id_column=None, usage=ALL,
                columns=None, columns_order=None):
    """
    Returns a `data_frame` columns.
    :param data_frame:
    :param target_column: a columns designating a target.
    :param id_column: a column designating an id.
    :param usage (str): should be a value from [ALL, INCLUDE, EXCLUDE].
                        this value only makes sense columns is also set.
                        otherwise, should be used with default value ALL.
    :param columns: if usage is all, this value is not used.
                    if usage is INCLUDE, returns the intersection `data_frame` columns.
                    if usage is EXCLUDE, returns the data_frame columns excluding this.
    :param columns_order: returns the `data_frame` columns with this order.
    :return: `data_frame` columns, excluding `target_column` and `id_column` if given.
             `data_frame` columns, including/excluding the `columns` depending on `usage`.
    """
    target_column = clean_id_column(target_column)
    id_column = clean_id_column(id_column)

    columns_excluded = pd.Index([])
    try:
        columns_included = data_frame[columns_order].columns
    except (TypeError, KeyError, ValueError):
        columns_included = data_frame.columns

    if target_column:
        columns_excluded = pd.Index([target_column])

    if id_column:
        columns_excluded = columns_excluded.union(pd.Index([id_column]))

    if usage == INCLUDE:
        try:
            columns_included = columns_included.intersection(pd.Index(columns))
        except TypeError:
            pass
    elif usage == EXCLUDE:
        try:
            columns_excluded = columns_excluded.union(pd.Index(columns))
        except TypeError:
            pass

    return columns_included.drop(columns_excluded)


def factorize_target(data_frame, target_column):
    """
    :param data_frame:
    :param target_column:
    :return: factorized target as enumerated type or categorical variable.
    """
    factorized_target = pd.factorize(data_frame[target_column])
    target = factorized_target[0]
    target_names = factorized_target[1]
    has_target = True
    return target, target_names, has_target


def prepare_data(data_frame, target_column=None, id_column=None, usage=ALL,
                 columns=None, columns_order=None):
    """
    Prepares the data, by separating feature names, data values, targets and ids.
    :param data_frame:
    :param target_column:
    :param id_column:
    :param usage:
    :param columns:
    :param columns_order:
    :return: `DATA_SET` object.
    """
    target_column = clean_id_column(target_column)
    id_column = clean_id_column(id_column)
    feature_names = get_columns(data_frame, target_column, id_column, usage, columns,
                                columns_order)
    data = data_frame[feature_names].values
    target = np.array([])
    target_names = np.array([])
    has_target = False
    ids = np.array([])
    has_ids = False

    if target_column:
        target, target_names, has_target = factorize_target(data_frame, target_column)

    if id_column:
        has_ids = True
        ids = data_frame[id_column]

    return DATA_SET(feature_names, data, target_names, target, has_target, ids, has_ids)


def select_indices(data_set, indices):
    """
    Selects a subset of indices from `data_set`.
    :param data_set:
    :param indices (list):
    :return: `DATA_SET` object.
    """
    target = data_set.target[indices] if data_set.has_target else data_set.target
    ids = data_set.ids[indices] if data_set.has_ids else data_set.ids
    return DATA_SET(data_set.feature_names, data_set.data[indices],
                    data_set.target_names, target, data_set.has_target, ids, data_set.has_ids)


def get_feature_location(data_set, feature):
    """
    :param data_set:
    :param feature (str):
    :return: feature location.
    """
    return data_set.feature_names.get_loc(feature)


def standard_scale(data, data_mean=None, data_std=None):
    """
    Does a standardization over data.
    Sometimes data do not fit in memory and we need to process chunks of data,
    in this case, `data_mean` and `data_std` are required to be calculated before
    scaling.
    :param data (array):
    :param data_mean (array):
    :param data_std (array):
    :return (array): standardized data.
    """
    data = data.astype('float')
    if (data_mean is None) and (data_std is None):
        # use sklearn default standardScaler
        std_scale = preprocessing.StandardScaler().fit(data)
        return std_scale.transform(data)

    # Custom standardization, since data is probably spread on
    # several parts and stats were collected before
    return (data - data_mean) / data_std


def min_max_scale(data, data_min=None, data_max=None):
    """
    Does a minmax scaling over data.
    Sometimes data do not fit in memory and we need to process chunks of data,
    in this case, `data_min` and `data_max` are required to be calculated before
    scaling.
    :param data (array):
    :param data_min (array):
    :param data_max (array):
    :return (array): scaled data.
    """
    data = data.astype('float')

    if (data_min is None) and (data_max is None):
        # use sklearn default MinMaxScaler
        std_scale = preprocessing.MinMaxScaler().fit(data)
        return std_scale.transform(data)

    # Custom standardization, since data is probably spread on
    # several parts and stats were collected before
    return (data - data_min) / (data_max - data_min)


def deviation_of_mean(data, multiplier=3, mean=None, std=None):
    """
    Remove values larger than `multiplier` * `std`.
    :param data (array): is normal distribution
    :param multiplier (float):
    :param mean (array):
    :param std (array):
    :return (array):
    """
    if not all([mean, std]):
        mean = data.mean()
        std = data.std()

    # truncate data
    data = np.minimum(data, mean + multiplier * std)
    return data


def median_absolute_deviation(data, c=0.6745, multiplier=3, median=None):
    """
    Remove values larger than `multiplier` * `mad`
    :param data (array):
    :param c (float): The normalization constant. Defined as scipy.stats.norm.ppf(3/4.)
    :param multiplier (float):
    :param median (float):
    :return (array):
    """
    if median is None:
        median = data.media

    mad = np.median(np.fabs(data - median)) / c

    # truncate data larger than mad * multiplier
    data = np.minimum(data, median + multiplier * mad)
    return data
