import itertools

import polars as pl
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from utils import *


# TODO: colocar em um arquivo de configuraćão depois.
CV = 5
KNN_PARAMS = {
    'n_neighbors': range(3, 15),
    'weights': ('uniform', 'distance'),
    'metric': ('cosine', 'euclidean'),
}

KNN_PARAMS = list(itertools.product(*[KNN_PARAMS[k] for k in KNN_PARAMS]))


if __name__ == '__main__':

    train = pl.read_csv('output/tf_idf/train.csv').to_pandas()

    grid_search_results = {}
    kf = KFold(n_splits=CV, shuffle=True)
    for k, w, m in KNN_PARAMS:

        print('-' * 75)
        print(f'Runing for: n_neighbors = {k} - weights = {w} - '
              f'metric = {m}.')

        curr_search = f'{k} - {w} - {m}'

        model = KNeighborsClassifier(
            n_neighbors=k,
            weights=w,
            metric=m
        )
        grid_search_results[curr_search] = kfold(
            model,
            train.iloc[:, 2:].values,
            train.iloc[:, 0].values,
            kf
        )
        exec_time = np.sum(grid_search_results[curr_search]['exec_time'])
        macro = np.mean(grid_search_results[curr_search]['f1_macro'])
        micro = np.mean(grid_search_results[curr_search]['f1_micro'])

        print(f'F1_macro: {np.round(macro, 3)}.')
        print(f'F1_micro: {np.round(micro, 3)}.')
        print(f'Execution time for {CV}-Fold: {np.round(exec_time, 3)}s.')

    with open('output/results/knn/train_results.json', 'w') as of:
        json.dump(grid_search_results, of, cls=NumpyEncoder)
        print('Results are saved.')
