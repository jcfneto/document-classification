import polars as pl

from utils import *

from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer


def tf_idf(df_train: pd.DataFrame,
           df_test: pd.DataFrame,
           colname: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates vector representations based on TF-IDF.

    Args:
        df_train: Train data.
        df_test: Test data.
        colname: Column name with the text data to be converted.

    Returns:
        df_train: Weight training data from the TF-IDF.
        df_test: Weight test data from the TF-IDF.
    """

    # TF-IDF embedding transformer (lowercase = True to default)
    vectorizer = TfidfVectorizer(norm='l2', stop_words='english')
    transformer = vectorizer.fit(df_train[colname].values)
    col_names = vectorizer.get_feature_names_out()

    # converting
    for _ in (df_train, df_test):
        embedding = transformer.transform(_[colname].values).toarray()
        df_tfidf = pd.DataFrame(embedding, columns=col_names)
        _ = pd.concat([_, df_tfidf], axis=1)

    return df_train, df_test


def sentence_embedding(df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       colname: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates vector representations based on sentenceBERT.

    Args:
        df_train: Train data.
        df_test: Test data.
        colname: Column name with the text data to be converted.

    Returns:
        df_train: Weight training data from the sentenceBERT.
        df_test: Weight test data from the sentenceBERT.
    """

    # sentenceBERT transformer
    col_names = [f'dim{i + 1}' for i in range(768)]
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    for _ in (df_train, df_test):
        bert = model.encode(_[colname].values)
        _ = pd.concat([_, pd.DataFrame(bert, columns=col_names)],
                      axis=1)

    return df_train, df_test


def to_csv(directory: str, filenames: Sequence, dataframes: Sequence) -> None:
    """
    Save dataframes in csv.

    Args:
        directory: Directory name.
        filenames: File names.
        dataframes: Dataframes to be saved.
    """
    for name, df in zip(filenames, dataframes):
        pl.DataFrame(df).write_csv(f'{directory}/{name}.csv')


if __name__ == '__main__':
    # processing documents
    train = processing_documents('../training/')
    test = processing_documents('../test/')

    # TF-IDF embedding
    start = time.time()
    train_tfidf, test_tfidf = tf_idf(train, test, 'doc_content')
    to_csv('../output/tf_idf', ('train', 'test'), (train_tfidf, test_tfidf))
    print(f'TF-IDF execution time: {time.time() - start}.')

    # sentenceBERT embedding
    start = time.time()
    train_bert, test_bert = sentence_embedding(train, test, 'doc_content')
    to_csv('../output/bert', ('train', 'test'), (train_bert, test_bert))
    print(f'sentenceBERT execution time: {time.time() - start}.')
