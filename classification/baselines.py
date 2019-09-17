import os
import sys
import time

import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from classification import evaluation, ordinal, shared_parameters
import folders
from sites.bookcave import bookcave, bookcave_ids


def identity(v):
    return v


def create_mnb():
    return MultinomialNB(fit_prior=True)


def create_lr():
    return LogisticRegression(solver='lbfgs')


def create_rf():
    return RandomForestClassifier(n_estimators=6)


def create_svm():
    return LinearSVC()


def main(argv):
    if len(argv) < 1 or len(argv) > 1:
        print('Usage: <max_words>')
    max_words = int(argv[0])

    stamp = int(time.time())
    base_fname = format(stamp, 'd')

    if not os.path.exists(folders.LOGS_PATH):
        os.mkdir(folders.LOGS_PATH)
    logs_baselines_path = os.path.join(folders.LOGS_PATH, 'baselines')
    if not os.path.exists(logs_baselines_path):
        os.mkdir(logs_baselines_path)
    if not os.path.exists(folders.PREDICTIONS_PATH):
        os.mkdir(folders.PREDICTIONS_PATH)
    predictions_baselines_path = os.path.join(folders.PREDICTIONS_PATH, 'baselines')
    if not os.path.exists(predictions_baselines_path):
        os.mkdir(predictions_baselines_path)

    # Load data.
    print('Retrieving texts...')
    ids_fname = bookcave_ids.get_ids_fname()
    book_ids = bookcave_ids.get_book_ids(ids_fname)
    categories_mode = 'soft'
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'sentence_tokens'},
                          only_ids=book_ids,
                          categories_mode=categories_mode)
    text_sentence_tokens, text_section_ids, text_paragraph_ids = zip(*inputs['sentence_tokens'])
    print('Retrieved {:d} texts.'.format(len(text_sentence_tokens)))

    # Create vectorized representations of the book texts.
    print('Vectorizing text...')
    vectorizer = TfidfVectorizer(
        preprocessor=identity,
        tokenizer=identity,
        analyzer='word',
        token_pattern=None,
        max_features=max_words,
        norm='l2',
        sublinear_tf=True)
    text_tokens = []
    for sentence_tokens in text_sentence_tokens:
        all_tokens = []
        for tokens in sentence_tokens:
            all_tokens.extend(tokens)
        text_tokens.append(all_tokens)
    X = vectorizer.fit_transform(text_tokens)  # TODO: Save to file?
    print('Vectorized text with {:d} unique words.'.format(len(vectorizer.get_feature_names())))

    # Split data set.
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    Y_T = Y.transpose()  # (n, c)
    X_train, X_test, Y_train_T, Y_test_T = train_test_split(X, Y_T, test_size=test_size, random_state=test_random_state)
    Y_train = Y_train_T.transpose()  # (c, n * (1 - b))
    Y_test = Y_test_T.transpose()  # (c, n * b)

    create_models = [create_mnb, create_lr, create_rf, create_svm]
    model_names = ['multinomial_naive_bayes', 'logistic_regression', 'random_forest', 'svm']
    for m, create_model in enumerate(create_models):
        model_name = model_names[m]
        print('Training model `{}`...'.format(model_name))
        Y_pred = []
        for j, category in enumerate(categories):
            print('Classifying category `{}`...'.format(category))

            k = len(category_levels[j])
            y_train = Y_train[j]  # (n * (1 - b))
            y_test = Y_test[j]  # (n * b)

            # Calculate probabilities for derived data sets.
            y_train_ordinal = ordinal.to_multi_hot_ordinal(y_train, k=k)  # (n * (1 - b), k - 1)
            classifiers = [create_model() for _ in range(k - 1)]
            ordinal_p = np.zeros((len(y_test), k - 1))  # (n * b, k - 1)
            for i, classifier in enumerate(classifiers):
                classifier.fit(X_train, y_train_ordinal[:, i])
                ordinal_p[:, i] = classifier.predict(X_test)

            # Calculate the actual class label probabilities.
            p = np.zeros((len(y_test), k))  # (n * b, k)
            for i in range(k):
                if i == 0:
                    p[:, i] = 1 - ordinal_p[:, 0]
                elif i == k - 1:
                    p[:, i] = ordinal_p[:, i - 1]
                else:
                    p[:, i] = ordinal_p[:, i - 1] - ordinal_p[:, i]

            # Choose the most likely class label.
            y_pred = np.argmax(p, axis=1)  # (n * b)
            Y_pred.append(y_pred)

        logs_path = os.path.join(logs_baselines_path, model_name)
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        with open(os.path.join(logs_path, '{}.txt'.format(base_fname)), 'w') as fd:
            fd.write('HYPERPARAMETERS\n')
            fd.write('\nText\n')
            fd.write('ids_fname={}\n'.format(bookcave_ids.get_ids_fname()))
            fd.write('\nLabels\n')
            fd.write('categories_mode=\'{}\'\n'.format(categories_mode))
            fd.write('\nVectorization\n')
            fd.write('max_words={:d}\n'.format(max_words))
            fd.write('vectorizer={}\n'.format(vectorizer.__class__.__name__))
            fd.write('\nTraining\n')
            fd.write('test_size={:.2f}\n'.format(test_size))
            fd.write('test_random_state={:d}\n'.format(test_random_state))
            fd.write('\nRESULTS\n\n')
            fd.write('Data size: {:d}\n'.format(len(text_sentence_tokens)))
            fd.write('Train size: {:d}\n'.format(len(X_train)))
            fd.write('Test size: {:d}\n\n'.format(len(X_test)))
            evaluation.write_confusion_and_metrics(Y_test, Y_pred, fd, categories)

        predictions_path = os.path.join(predictions_baselines_path, model_name)
        if not os.path.exists(predictions_path):
            os.mkdir(predictions_path)
        with open(os.path.join(predictions_path, '{}.txt'.format(base_fname)), 'w') as fd:
            evaluation.write_predictions(Y_test, Y_pred, fd, categories)


if __name__ == '__main__':
    main(sys.argv[1:])
