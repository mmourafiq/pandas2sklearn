import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from pandas_sklearn import DataSet

@pytest.fixture
def iris_data():
    return pd.read_csv('tests/fixtures/iris.csv')

def test_dataset_creates_nes_instance(iris_data):
    dataset = DataSet(iris_data)

    # has a df
    assert_frame_equal(dataset.df, iris_data)

    # has no id
    assert not dataset.has_id()

    # has no target
    assert not dataset.has_target()

    # all columns names are now features
    assert len(dataset.feature_names) == 5


def test_dataset_set_target_works(iris_data):
    dataset = DataSet(iris_data, target_column='target')

    # only 4 columns names are feature
    assert len(dataset.feature_names) == 4

    # has target
    assert dataset.has_target()

    # has 3 target
    assert len(dataset.target_names) == 3

    # target is of values 0, 1, 2
    assert_array_equal(pd.unique(dataset.target), [0, 1, 2])


def test_dataset_get_columns_works_as_expected(iris_data):
    dataset = DataSet(iris_data, target_column='target')

    # All except the target, should be 4 columns
    assert len(dataset.get_columns(DataSet.ALL)) == 4

    # include only sepal length (cm),sepal width (cm), should be 2
    assert len(dataset.get_columns(DataSet.INCLUDE, ['sepal length (cm)', 'sepal width (cm)'])) == 2

    # exclude sepal length (cm),sepal width (cm), should be 2
    assert len(dataset.get_columns(DataSet.EXCLUDE, ['sepal length (cm)', 'sepal width (cm)'])) == 2
