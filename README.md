[![Build Status](https://travis-ci.org/mouradmourafiq/pandas2sklearn.svg?branch=master)](https://travis-ci.org/mouradmourafiq/pandas2sklearn)

# pandas2sklearn
An integration of [pandas](http://pandas.pydata.org/) dataframes with [scikit learn](http://scikit-learn.org/stable/).

The module contains:

 * dealing with dataframes in a scikit learn `DataSet` fashion.
 * transformation mechanism that can be easily integrated in scikit learn pipelines, `DataSetTransformer`.
 

# Installation
The module can be easily installed with pip:

```conslole
> pip install pandas2sklearn
```

# Tests
The module contains some basic testing of the provided functionalities.

```console
> py.test
```

# Usage
The module contains two classes:

## DataSet

The `DataSet` is wrapper around pandas `DataFrame`, that converts you can use to select:
 * id
 * features
 * target
 
Example, suppose we have a `DataFrame` that has the following columns;

`df.coumns` = `id, FN1, FN2, FN3, FN4, FN5, FC1, FC2, FC3, FC4, FC5, FC6, target`

```python
from pandas_sklearn import DataSet

dataset = DataSet(df, target_column='target', id_column='id')

dataset.has_target() == True
dataset.has_id() == True
dataset.target == df['target']
dataset.id == df['id']
dataset.target_names == ['FN1', 'FN2', 'FN3', 'FN4', 'FN5', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6']
dataset.data == df[['FN1', 'FN2', 'FN3', 'FN4', 'FN5', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6']]


# removing some features that are not needed FN4, FN5, FC1, FC5, FC6
dataset.set_feature_names(usage=DataSet.EXCLUDE, columns=['FN4', 'FN5', 'FC1', 'FC5', 'FC6'])
dataset.target_names == ['FN1', 'FN2', 'FN3', 'FC2', 'FC3', 'FC4']

# converting the dataset to dictionary
dataset.to_dict() == [
    {'FN1': 12, 'FN2': 23, 'FC2': 'coffee', 'FC2': 'xbox one', 'FC4': 'inch'},
    ...
]
```

## DataSetTransformer
A feature wise transformer, applies a scikit-learn transformer to one or more features. e.g.

```python
DataSetTransformer([
    (['petal length (cm)', 'petal width (cm)'], StandardScaler()),
    ('sepal length (cm)', MinMaxScaler()),
    ('sepal width (cm)', None),
]))
```

It could be used together with pipelines, e.g.
```python
pipeline = Pipeline([
    ('preprocess', DataSetTransformer([
        (['petal length (cm)', 'petal width (cm)'], StandardScaler()),
        ('sepal length (cm)', MinMaxScaler()),
        ('sepal width (cm)', None),
    ])),
    ('classify', SVC(kernel='linear'))
])
```

# Credit
The `DataSetTransformer` is based on the work of Ben Hamner and Paul Butler.
