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


def reading_files(path: str, lower: bool = True) -> list:
    """
    Reading files on specific directory.

    Args:
        path: Path.
        lower: ...

    Returns:
        files: ...
    """
    files = []
    for d in os.listdir(path):
        for f in os.listdir(f'{path}{d}'):
            tmp = open(f'{path}{d}/{f}', 'r', encoding='ISO-8859-1').read()
            if lower:
                tmp = tmp.lower()
            files.append((d, tmp))
    return files


def remove_html(string: str) -> str:
    """
    ...

    Args:
    string: ...

    Returns:
    string: ...
    """
    return re.sub(r'<.*?>', '', string)


def remove_punctuation(string: str) -> str:
    """
    ...

    Args:
        string: ...

    Returns:
        string: ...
    """
    return re.sub(r'[^\w\s]', '', string)


def remove_whitespaces(string: str) -> str:
    """
    ...

    Args:
        string: ...

    Returns:
        string: ...
    """
    return re.sub(' +', ' ', string)


def remove_tags(string: str) -> str:
    """
    ...

    Args:
        string: ...

    Returns:
        string: ...
    """
    return re.sub(r'\n', ' ', string)


def remove_numbers(string: str) -> str:
    """
    ...

    Args:
        string: ...

    Returns:
        string: ...
    """
    return re.sub(r'\d+', '', string)


def pipeline(value: T, functions: Sequence[Callable[[T], T]]) -> T:
    """
    ...

    Args:
        value: ...
        functions: ...

    Returns:
        string: ...
    """
    return reduce(lambda v, f: f(v), functions, value)


def remove_stopwords(string: str, stopword_language: str = 'english') -> str:
    """
    ...

    Args:
        string: ...
        stopword_language: ...

    Returns:
        string: ...
    """
    new_string = ''
    for word in word_tokenize(string):
        if word not in stopwords.words(stopword_language):
            new_string += f'{word} '
    return new_string[:-1]


def processing_documents(path: str):
    """
    ...

    Args:
        path:

    Returns:
        ...
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
    ...

    Args:
        y_true
        y_pred:
        results:

    Returns:
        ...
    """
    metrics = [('f1', None), ('f1_micro', 'micro'), ('f1_macro', 'macro')]
    for k, f in metrics:
        score = f1_score(y_true, y_pred, average=f)
        results[k].append(score)
    return results


def kfold(estimator, X, y, cv):
    """
    ...

    Args:
        estimator:
        X:
        y:
        cv:

    Returns:
        ...
    """

    results = defaultdict(list)

    for i, j in cv.split(X):

        # spliting data
        X_train, y_train = X[i,:], y[i]
        X_test, y_test = X[j,:], y[j]

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
