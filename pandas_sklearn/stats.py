import pandas as pd
import numpy as np
import sys
import gc


def compute_stats(file_name,
                  numeric_columns_slice=None,
                  category_columns_slice=None,
                  target_column=None):
    """
    Compute data statistics
        Categorical Variables: Number of categories, Histogram
        Numerical Variables: Min, Max, Mean
        Targets and observations

    e.g.: compute_stats(file_name='dummy',
                        numeric_columns_slice=(2, 15),
                        category_columns_slice=(15, None),
                        target_column='Label')
    """
    stats_dir = 'data/stats/'
    reader = pd.read_csv(file_name)

    stats = reader.describe()

    if numeric_columns_slice:
        stats_integer = stats.iloc[:, slice(*numeric_columns_slice)]
        stats_integer.to_csv('{}numeric_stats.csv'.format(stats_dir))

    if category_columns_slice:
        stats_category = stats.iloc[:, slice(*category_columns_slice)]
        stats_category.to_csv('{}category_stats.csv'.format(stats_dir))

    if target_column:
        total_target = reader.groupby(target_column).size()
        total_target['total_observations'] = reader.shape[0]
        total_target.to_csv('{}total_target.csv'.format(stats_dir))


def compute_stats_multiple(file_name, chunk_size,
                           numeric_columns_slice=None,
                           category_columns_slice=None,
                           num_categories=0, target_column=None):
    """
    Compute data statistics for a large file across multiple chunks
        Categorical Variables: Number of categories, Histogram
        Integer Variables: Min, Max, Mean
        Targets and observations

    e.g.: compute_stats(file_name='dummy',
                        chunk_size=10000,
                        numeric_columns_slice=(2, 15),
                        category_columns_slice=(15, None),
                        num_categories=27,
                        target_column='Label')
    """
    stats_dir = 'data/stats/'
    reader = pd.read_csv(file_name, chunksize=chunk_size)

    total_target = 0
    total_observations = 0

    stats_integer = pd.DataFrame()
    stats_category = {}
    for i in range(1, num_categories):
        stats_category['C' + str(i)] = pd.DataFrame()

    count = 0
    for chunk in reader:
        print 'Reading line:' + str(count * chunk_size)

        if category_columns_slice:
            chunk_category = chunk.iloc[:, slice(*category_columns_slice)]
            for i in range(1, num_categories):
                category_label = 'C' + str(i)

                category_frame = pd.DataFrame()
                category_frame['category'] = chunk_category.groupby(category_label).size().index
                category_frame['count'] = chunk_category.groupby(category_label).size().values
                stats_category[category_label] = pd.concat([stats_category[category_label], category_frame])

                # Aggregate on common category values
                category_frame = pd.DataFrame()
                category_frame['category'] = stats_category[category_label].groupby('category').sum().index
                category_frame['count'] = stats_category[category_label].groupby("category").sum().values
                stats_category[category_label] = category_frame

                gc.collect()

        if numeric_columns_slice:
            chunk_integer = chunk.iloc[:, slice(*numeric_columns_slice)]
            if count == 0:
                stats_integer['max'] = chunk_integer.max()
                stats_integer['min'] = chunk_integer.min()
                stats_integer['sum'] = chunk_integer.sum()
                stats_integer['count'] = chunk_integer.count()
            else:
                stats_integer['max_chunk'] = chunk_integer.max()
                stats_integer['min_chunk'] = chunk_integer.min()
                stats_integer['sum_chunk'] = chunk_integer.sum()
                stats_integer['count_chunk'] = chunk_integer.count()

                stats_integer['max'] = stats_integer[['max', 'max_chunk']].max(axis=1)
                stats_integer['min'] = stats_integer[['min', 'min_chunk']].max(axis=1)
                stats_integer['sum'] = stats_integer[['sum', 'sum_chunk']].sum(axis=1)
                stats_integer['count'] = stats_integer[['count', 'count_chunk']].sum(axis=1)

                stats_integer.drop(['max_chunk', 'min_chunk', 'sum_chunk', 'count_chunk'],
                                   axis=1,
                                   inplace=True)

        if target_column:
            total_target += chunk.groupby(target_column).size()
        total_observations += chunk.shape[0]

        count += 1

    total_target.to_csv('{}total_target.csv'.format(stats_dir))
    print "Total target:" + str(total_target) + " Total observations:" + str(total_observations)

    if category_columns_slice:
        # categories
        stats_category_agg = pd.DataFrame()
        for i in range(1, 27):
            category_frame = stats_category['C' + str(i)].groupby('category').sum().describe().transpose()
            category_frame.reset_index()
            category_frame.index = ['C' + str(i)]
            stats_category_agg = pd.concat([stats_category_agg, category_frame])

        print stats_category_agg
        stats_category_agg.to_csv('{}category_stats.csv'.format(stats_dir))

    if numeric_columns_slice:
        # integers
        stats_integer['mean'] = stats_integer['sum'] / stats_integer['count']

        reader = pd.read_csv(file_name, chunksize=chunk_size)

        count = 0
        for chunk in reader:
            print 'Reading line:' + str(count * chunk_size)

            chunk_integer = chunk.iloc[:, slice(*numeric_columns_slice)]

            if count == 0:
                stats_integer['sq_sum'] = ((chunk_integer - stats_integer['mean']) ** 2).sum()
            else:
                stats_integer['sq_sum_chunk'] = ((chunk_integer - stats_integer['mean']) ** 2).sum()
                stats_integer['sq_sum'] = stats_integer[['sq_sum', 'sq_sum_chunk']].sum(axis=1)
                stats_integer.drop(['sq_sum_chunk'], axis=1, inplace=True)

            count += 1

        stats_integer['std'] = (stats_integer['sq_sum'] / (stats_integer['count'] - 1)).apply(np.sqrt)
        stats_integer.drop(['sq_sum'], axis=1, inplace=True)

        print stats_integer
        stats_integer.to_csv('{}integer_stats.csv'.format(stats_dir))