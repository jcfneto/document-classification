import itertools

import polars as pl
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from utils import *


# TODO: colocar em um arquivo de configuraćão depois.
CV = 5
KNN_PARAMS = {
    'n_neighbors': range(3, 16),
    'weights': ('uniform', 'distance'),
    'metric': ('cosine', 'euclidean'),
}

KNN_PARAMS = list(itertools.product(*[KNN_PARAMS[k] for k in KNN_PARAMS]))


def main(df: pd.DataFrame, embedding_type: str) -> None:
    """

    Args:
        df:
        embedding_type:
    """
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
            df.iloc[:, 2:].values,
            df.iloc[:, 0].values,
            kf
        )
        for j in grid_search_results[curr_search].keys():
            score = np.round(np.mean(grid_search_results[curr_search][j]), 3)
            print(f'{j} = {score}.')

    with open(f'output/results/knn/{embedding_type}_results.json', 'w') as of:
        json.dump(grid_search_results, of, cls=NumpyEncoder)
        print('Results are saved.')


if __name__ == '__main__':

    embeddings = ('tf_idf', 'bert')
    for embedding in embeddings:
        print(f'Starting grid search for "{embedding}" embedding.')
        train = pl.read_csv(f'output/{embedding}/train.csv').to_pandas()
        main(train, embedding)
        print(f'Finish grid search for "{embedding}".\n')

# blabla bla bla bla bla
print('qualquer coisa')