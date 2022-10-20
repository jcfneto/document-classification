import time

import polars as pl

from utils import *

from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer


def tf_idf(df_train: pd.DataFrame, df_test: pd.DataFrame) -> \
        tuple[pd.DataFrame, pd.DataFrame]:
    """
    pass...

    Args:
        df_train:
        df_test:

    Returns:
        pass...
    """

    # TF-IDF embedding transformer
    vectorizer = TfidfVectorizer(norm='l2', stop_words='english')
    transformer = vectorizer.fit(df_train.doc_content.values)
    col_names = vectorizer.get_feature_names_out()

    # TF-IDF train
    embedding = transformer.transform(df_train.doc_content.values).toarray()
    train_tf_idf = pd.DataFrame(embedding, columns=col_names)
    df_train = pd.concat([df_train, train_tf_idf], axis=1)

    # TF-IDF test
    embedding = transformer.transform(df_test.doc_content.values).toarray()
    test_tf_idf = pd.DataFrame(embedding, columns=col_names)
    df_test = pd.concat([df_test, test_tf_idf], axis=1)

    return df_train, df_test


def sentence_embedding(df_train: pd.DataFrame, df_test: pd.DataFrame) -> \
        tuple[pd.DataFrame, pd.DataFrame]:
    """
    pass...

    Args:
        df_train:
        df_test:

    Returns:
        pass...
    """

    # sentenceBERT transformer
    col_names = [f'dim{i + 1}' for i in range(768)]
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    # sentenceBERT train
    bert = model.encode(df_train.doc_content.values)
    df_train = pd.concat([df_train, pd.DataFrame(bert, columns=col_names)])

    # sentenceBERT test
    bert = model.encode(df_test.doc_content.values)
    df_test = pd.concat([df_test, pd.DataFrame(bert, columns=col_names)])

    return df_train, df_test


def to_csv(directory: str, filenames: Sequence, dataframes: Sequence) -> None:
    """
    pass...

    Args:
        directory:
        filenames:
        dataframes:

    Returns:
        pass...
    """
    for name, df in zip(filenames, dataframes):
        pl.DataFrame(df).write_csv(f'{directory}/{name}.csv')


if __name__ == '__main__':

    # processing documents
    train = processing_documents('training/')
    test = processing_documents('test/')

    # TF-IDF embedding
    start = time.time()
    train_tf_idf, test_tf_idf = tf_idf(train, test)
    to_csv('output/tf_idf', ('train', 'test'), (train_tf_idf, test_tf_idf))
    print(f'TF-IDF execution time: {time.time() - start}.')

    # sentenceBERT embedding
    start = time.time()
    train_bert, test_bert = sentence_embedding(train, test)
    to_csv('output/bert', ('train', 'test'), (train_bert, test_bert))
    print(f'sentenceBERT execution time: {time.time() - start}.')
