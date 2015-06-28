import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from pandas_sklearn import DataSet, DataSetTransformer


@pytest.fixture
def iris_data():
    return pd.read_csv('tests/fixtures/iris.csv')

def assert_equal(l1, l2):
    return len(l1) == len(l2) and sorted(l1) == sorted(l2)

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


def test_dataset_to_dict_works_as_expected(iris_data):
    dataset = DataSet(iris_data, target_column='target')

    dataset_dict = dataset.to_dict()
    expected_columns = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]

    assert assert_equal(dataset_dict[0].keys(), expected_columns)

    expected_columns = ['sl', 'sw', 'pl', 'pw']
    dataset_dict = dataset.to_dict(expected_columns)

    assert assert_equal(dataset_dict[0].keys(), expected_columns)

    with pytest.raises(AssertionError):
        dataset.to_dict(['a', 'b'])


def test_with_iris_dataframe(iris_data):
    dataset = DataSet(iris_data, target_column='target')
    pipeline = Pipeline([
        ('preprocess', DataSetTransformer([
            (['petal length (cm)', 'petal width (cm)'], StandardScaler()),
            ('sepal length (cm)', MinMaxScaler()),
            ('sepal width (cm)', None),
        ])),
        ('classify', SVC(kernel='linear'))
    ])
    scores = cross_val_score(pipeline, dataset, dataset.target)
    assert scores.mean() > 0.96
