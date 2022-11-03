import itertools

import polars as pl
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

from utils import *


CV = 5
RF_PARAMS = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

RF_PARAMS = list(itertools.product(*[RF_PARAMS[k] for k in RF_PARAMS]))


def main(df: pd.DataFrame, embedding_type: str) -> None:
    """

    Args:
        df:
        embedding_type:

    Returns:

    """
    grid_search_results = {}
    kf = KFold(n_splits=CV, shuffle=True)
    for n, mf, md, c in RF_PARAMS:
        print('-' * 75)
        print(f'Runing for n_estimators = {n} - max_features = {mf} - '
              f'max_depth = {md} - criterion = {c}.')
        curr_search = f'{n} - {mf} - {md} - {c}'
        model = RandomForestClassifier(
            n_estimators=n,
            max_features=mf,
            max_depth=md,
            criterion=c
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

    with open(f'output/results/rf/{embedding_type}_results.json',
              'w') as of:
        json.dump(grid_search_results, of, cls=NumpyEncoder)
        print('Results are saved.')


if __name__ == '__main__':

    embeddings = ('tf_idf', 'bert')
    for embedding in embeddings:
        print(f'Starting grid search for "{embedding}" embedding.')
        train = pl.read_csv(f'output/{embedding}/train.csv').to_pandas()
        main(train, embedding)
        print(f'Finish grid search for "{embedding}".\n')
