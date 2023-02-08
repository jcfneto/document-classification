# What needs to be done

The goal of this work is to implement text classifiers using machine learning algorithms. Follow the instructions below to complete the work:

- Download the Reuters-21578, 90 categories collection from the website http://disi.unitn.it/moschitti/corpora.htm. The collection is organized into training and testing documents, stored in directories according to each class;

- Preprocess the data to handle character encodings, convert all text to lowercase, remove punctuation, unnecessary symbols, tags (in the case of HTML, XML, etc.), and stopwords, tokenize each word of the text, etc. Prepare the documents to be used as input for the algorithms;

- Choose two classifiers and compare their efficacy for classification in this collection. For each classifier, try the following two strategies for vectorizing the input documents: (1) use the words (tokens) as attributes (features) and the TF-IDF weight scheme as the value for the attributes and (2) use a word embedding scheme for the words and, for the input of each document, use the average of its word embeddings or a document embedding scheme, such as "Sentence Bert";

- Adjust the classifier parameters by doing a grid search using 10-fold cross validation. Use the training data for this. After the best parameters have been chosen, train the classifiers with the entire training collection and evaluate them using the test collection;

- Evaluate the quality of the results using the following metrics: precision, recall and F1 per class, and Macro-F1 and Micro-F1 (accuracy) for the set of classes. Present the results in a table, comparing the classifiers. Do a statistical test using, for example, ANOVA, to verify if one classifier is truly better than the other.

# Organization of the repository

The repository is organized as follows:

- `output`:
  - `bert`: Directory to store the training and testing data with BERT vectorization.
  - `tf_idf`: Directory to store the training and testing data with TF-IDF vectorization.
  - `results`: Directory to store the results obtained from the grid search and the final models.
- `scripts`: Directory with all the used codes.
- `test`: Test documents.
- `training`: Training documents.
- `article`: Directory to store the technical report.
