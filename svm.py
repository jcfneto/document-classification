import itertools

import polars as pl
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from utils import *


CV = 5
SVM_PARAMS = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

SVM_PARAMS = list(itertools.product(*[SVM_PARAMS[k] for k in SVM_PARAMS]))


def main(df: pd.DataFrame, embedding_type: str) -> None:
    """

    Args:
        df:
        embedding_type:

    Returns:

    """
    grid_search_results = {}
    kf = KFold(n_splits=CV, shuffle=True)
    for c, g, k in SVM_PARAMS:
        print('-' * 75)
        print(f'Runing for C = {c} - gamma = {g} - kernel = {k}')
        curr_search = f'{c} - {g} - {k}'
        model = SVC(C=c, gamma=g, kernel=k)
        grid_search_results[curr_search] = kfold(
            model,
            df.iloc[:, 2:].values,
            df.iloc[:, 0].values,
            kf
        )
        for j in grid_search_results[curr_search].keys():
            score = np.round(np.mean(grid_search_results[curr_search][j]), 3)
            print(f'{j} = {score}.')

    with open(f'output/results/svm/{embedding_type}_results.json',
              'w') as of:
        json.dump(grid_search_results, of, cls=NumpyEncoder)
        print('Results are saved.')


if __name__ == '__main__':

    embeddings = ('tf_idf', 'bert')
    for embedding in embeddings:
        print(f'Starting grid search for "{embedding}" embedding.')
        train = pl.read_csv(f'output/{embedding}/train_pca.csv').to_pandas()
        main(train, embedding)
        print(f'Finish grid search for "{embedding}".\n')
