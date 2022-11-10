import time

import numpy as np
import pandas as pd
import polars as pl

from sklearn.decomposition import PCA

"""
Let's reduce the dimensionality of the vectors produced by sentence_bert so 
that training using SVM is practicable.

PCA is used to decompose a multivariate dataset in a set of successive 
orthogonal components that explain a maximum amount of the variance. 
The PCA object also provides a probabilistic interpretation of the PCA 
that can give a likelihood of data based on the amount of variance it 
explains. In this project we will produce a new vector keeping 75% of 
variance.
"""


def best_n_components(x: np.array, threshold: float) -> [int, PCA]:
    """
    Selects the best value for the number of components.

    Args:
        x: Data to be compressed.
        threshold: Threshold for decision based on explained variance rate.

    Returns:
        n_components: Number of components.
        pca: PCA object.
    """
    varianve_ratio = 0
    n_components = 0

    dim = x.shape[1]
    if x.shape[0] < dim:
        dim = x.shape[0]

    pca = PCA(n_components=dim)
    pca.fit(x)

    for n in pca.explained_variance_ratio_:
        n_components += 1
        varianve_ratio += n
        if varianve_ratio >= threshold:
            break

    print(f'{round(100 * varianve_ratio, 2)}% variance with {n_components} '
          f'components.')

    return n_components, pca


if __name__ == '__main__':

    variance_threshold = 0.75
    for et in ('tf_idf', 'bert'):
        start = time.time()

        # reading data
        train = pl.read_csv(f'../output/{et}/train.csv').to_pandas()

        # searching for the number of components
        n_components, pca = best_n_components(
            train.iloc[:, 2:].values,
            variance_threshold)
        columns_names = train.columns[:2+n_components]

        # join
        test = pl.read_csv(f'../output/{et}/test.csv').to_pandas()
        for df, name in zip((train, test), ('train', 'test')):

            # transforming
            dec = pca.transform(df.iloc[:, 2:].values)[:, :n_components]
            tmp = pd.DataFrame(
                np.c_[df.iloc[:, :2].values, dec],
                columns=columns_names
            )
            tmp.to_csv(f'../output/{et}/{name}_pca.csv', index=False)

        end = time.time()
        print(f'Execution time for "{et}": {round(end - start)}s.')
        print('-' * 50)
