__version__ = '0.0.2'

import numpy as np
import pandas as pd


class DataSet(object):
    """
    The `DataSet` class represents a mapping mechanism from pandas dataframes to
    sklearn `datasets` format.

    Attributes:
      df (pandas.DataFrame):
      fixtures (ndarray): the DataFrame values restricted to the list of feature names
      feature_names (list): list of the column names that represent the feature names
      target_column (str): the name of the target column
      target (ndarray): the vectorized target
      target_names (list): the list of target names
      id_column (str): the name of the id column
      id (ndarray):
    """
    ALL = 'ALL'
    INCLUDE = 'INCLUDE'
    EXCLUDE = 'EXCLUDE'

    def __init__(self, df, target_column=None, id_column=None, usage=ALL, columns=None):
        self.df = df
        self.target_column = None
        self.target = None
        self.target_names = None
        self.id_column = None
        self.feature_names = None

        if target_column:
            self.set_target(target_column, False)
        if id_column:
            self.set_id(id_column, False)
        self.set_feature_names(usage, columns)

    def __getitem__(self, key):
        if isinstance(key, str) and self._clean_column(key):
            return self.df[key].values

        if isinstance(key, (tuple, list)):
            error_keys = [k for k in key if not self._clean_column(k)]
            if len(error_keys) > 0:
                raise KeyError(', '.join(error_keys))
            return self.df[list(key)].values

        if isinstance(key, pd.Index):
            error_keys = [k for k in key.values if not self._clean_column(k)]
            if len(error_keys) > 0:
                raise KeyError(', '.join(error_keys))
            return self.df[key].values

        if isinstance(key, np.ndarray):
            error_keys = [k for k in key if not self._clean_column(k)]
            if len(error_keys) > 0:
                raise KeyError(', '.join(error_keys))
            return self.df[key].values

        raise KeyError(key)

    def get_feature_location(self, feature):
        """
        :param feature (str):
        :return: the feature location.
        """
        return self.feature_names.get_loc(feature)

    def _clean_column(self, column):
        if not isinstance(column, (int, str, unicode)):
            raise ValueError('{} is not a valid column'.format(column))
        return column in self.df.columns

    @property
    def id(self):
        if self.has_id():
            return self.df[self.id_column]

    @property
    def data(self):
        return self.df[self.feature_names].values

    def has_id(self):
        return True if self.id_column else False

    def has_target(self):
        return True if self.target_column else False

    def set_target(self, target_column, update_features=True):
        if self._clean_column(target_column):
            self.target_column = target_column
            self.target, self.target_names = pd.factorize(self.df[target_column])
            if update_features:
                self.set_feature_names()

    def set_id(self, id_column, update_features=True):
        if self._clean_column(id_column):
            self.id_column = id_column
            if update_features:
                self.set_feature_names()

    def set_feature_names(self, usage=ALL, columns=None):
        self.feature_names = self.get_columns(usage, columns)

    def get_columns(self, usage, columns=None):
        """
        Returns a `data_frame.columns`.
        :param usage (str): should be a value from [ALL, INCLUDE, EXCLUDE].
                            this value only makes sense if attr `columns` is also set.
                            otherwise, should be used with default value ALL.
        :param columns: * if `usage` is all, this value is not used.
                        * if `usage` is INCLUDE, the `df` is restricted to the intersection
                          between `columns` and the `df.columns`
                        * if usage is EXCLUDE, returns the `df.columns` excluding these `columns`
        :return: `data_frame` columns, excluding `target_column` and `id_column` if given.
                 `data_frame` columns, including/excluding the `columns` depending on `usage`.
        """
        columns_excluded = pd.Index([])
        columns_included = self.df.columns

        if self.has_target():
            columns_excluded = pd.Index([self.target_column])

        if self.has_id():
            columns_excluded = columns_excluded.union(pd.Index([self.id_column]))

        if usage == self.INCLUDE:
            try:
                columns_included = columns_included.intersection(pd.Index(columns))
            except TypeError:
                pass
        elif usage == self.EXCLUDE:
            try:
                columns_excluded = columns_excluded.union(pd.Index(columns))
            except TypeError:
                pass

        columns_included = columns_included.difference(columns_excluded)
        return columns_included.intersection(self.df.columns)

    def to_dict(self, columns=None):
        """
        Return a python dict from a pandas dataframe, with columns as keys
        :return: dict
        """
        if columns:
            assert len(columns) == len(self.df.columns)
        d = [
            dict([
                (columns[i] if columns else col_name, row[i])
                for i, col_name in enumerate(self.feature_names)
            ])
            for row in self.data
        ]
        return d
