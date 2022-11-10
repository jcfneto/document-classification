import json
import os
import re
import time
from collections import defaultdict
from functools import reduce
from typing import Callable, Sequence, TypeVar

import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import f1_score


T = TypeVar('T')


def reading_files(path: str) -> list:
    """
    Reading files on specific directory.

    Args:
        path: Path.

    Returns:
        files: Namefiles and document content.
    """
    files = []
    for d in os.listdir(path):
        for f in os.listdir(f'{path}{d}'):
            tmp = open(f'{path}{d}/{f}', 'r', encoding='ISO-8859-1').read()
            files.append((d, tmp))
    return files


def remove_html(string: str) -> str:
    """
    Remove html and xml tags.

    Args:
        string: String

    Returns:
        string: String without  tags.
    """
    return re.sub(r'<.*?>', '', string)


def remove_punctuation(string: str) -> str:
    """
    Remove punctuation.

    Args:
        string: String.

    Returns:
        string: String without punctuation.
    """
    return re.sub(r'[^\w\s]', '', string)


def remove_whitespaces(string: str) -> str:
    """
    Remove multiple whitespaces.

    Args:
        string: Strig.

    Returns:
        string: String without multiple whitespaces.
    """
    return re.sub(' +', ' ', string)


def remove_tags(string: str) -> str:
    """
    Remove tags.

    Args:
        string: String.

    Returns:
        stringString without tags.
    """
    return re.sub(r'\n', ' ', string)


def remove_numbers(string: str) -> str:
    """
    Change all number to 0.

    Args:
        string: String.

    Returns:
        string: String with the standardized numbers.
    """
    return re.sub(r'\d+', '', string)


def pipeline(value: T, functions: Sequence[Callable[[T], T]]) -> T:
    """
    Function execution sequencing.

    Args:
        value: Parameters.
        functions: Object function.

    Returns:
        string: Processed string.
    """
    return reduce(lambda v, f: f(v), functions, value)


def remove_stopwords(string: str, stopword_language: str = 'english') -> str:
    """
    Remove stopwords.

    Args:
        string: String.
        stopword_language: String language.

    Returns:
        string: String without stopwords.
    """
    new_string = ''
    for word in word_tokenize(string):
        if word not in stopwords.words(stopword_language):
            new_string += f'{word} '
    return new_string[:-1]


def processing_documents(path: str) -> pd.DataFrame:
    """
    Performs pre-processing of documents.

    Args:
        path: Directory where documents are saved.

    Returns:
        Dataframe with pre-processed documents.
    """
    processed = []
    files = reading_files(path)
    functions = (remove_html,
                 remove_punctuation,
                 remove_tags,
                 remove_numbers,
                 remove_whitespaces)

    for c, s in files:
        string = pipeline(s, functions)
        processed.append((c, string))

    return pd.DataFrame(processed, columns=['doc_class', 'doc_content'])


def f1(y_true: np.array,
       y_pred: np.array,
       results: defaultdict) -> defaultdict:
    """
    Compute f1 metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        results: Object to save the results.

    Returns:
         results: Dictionary with the F1 results.
    """
    metrics = ('micro', 'macro')
    for m in metrics:
        results[m].append(f1_score(y_true, y_pred, average=m))
    return results


def kfold(estimator, X, y, cv) -> defaultdict:
    """
    ...

    Args:
        estimator: The object model.
        X: Predictors.
        y: Targets.
        cv: Number of k-folds.

    Returns:
        results: Dictionary with the results of cross-validation runs.
    """

    results = defaultdict(list)

    for i, j in cv.split(X):

        # spliting data
        X_train, y_train = X[i, :], y[i]
        X_test, y_test = X[j, :], y[j]

        # training and predictions
        start = time.time()
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        end = time.time()

        # validating
        results['exec_time'].append(end-start)
        results = f1(y_test, pred, results)

    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
