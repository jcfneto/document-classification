import re
import os

import pandas as pd

from functools import reduce

from nltk import word_tokenize
from nltk.corpus import stopwords

from typing import TypeVar, Callable, Sequence

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
