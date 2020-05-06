import os
import pickle
import time

# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
import tensorflow as tf
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import LinearSVC

from python.util import evaluation, shared_parameters
from python.util import ordinal
from python import folders
from python.sites.bookcave import bookcave


def main():
    max_words = shared_parameters.TEXT_MAX_WORDS

    start_time = int(time.time())
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
    else:
        stamp = start_time
    print('Time stamp: {:d}'.format(stamp))

    # Load data.
    print('Retrieving texts...')
    source = 'paragraph_tokens'
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    return_overall = shared_parameters.DATA_RETURN_OVERALL
    inputs, Y, categories, category_levels = \
        bookcave.get_data({source},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          categories_mode=categories_mode,
                          return_overall=return_overall)
    text_source_tokens = list(zip(*inputs[source]))[0]
    print('Retrieved {:d} texts.'.format(len(text_source_tokens)))

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
    for source_tokens in text_source_tokens:
        all_tokens = []
        for tokens in source_tokens:
            all_tokens.extend(tokens)
        text_tokens.append(all_tokens)
    X = vectorizer.fit_transform(text_tokens)
    print('Vectorized text with {:d} unique words.'.format(len(vectorizer.get_feature_names())))

    # Split data set.
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    Y_T = Y.transpose()  # (n, c)
    X_train, X_test, Y_train_T, Y_test_T = train_test_split(X, Y_T, test_size=test_size, random_state=test_random_state)
    Y_train = Y_train_T.transpose()  # (c, n * (1 - b))
    Y_test = Y_test_T.transpose()  # (c, n * b)

    create_funcs = [
        create_k_nearest_neighbors,
        create_logistic_regression,
        create_multi_layer_perceptron,
        create_multinomial_naive_bayes,
        create_random_forest,
        create_svm
    ]
    model_names = [
        'k_nearest_neighbors',
        'logistic_regression',
        'multi_layer_perceptron',
        'multinomial_naive_bayes',
        'random_forest',
        'svm'
    ]
    for m, create_func in enumerate(create_funcs):
        model_name = model_names[m]
        model_path = folders.ensure(os.path.join(folders.MODELS_PATH, model_name))
        print('Training model `{}`...'.format(model_name))
        for j, category in enumerate(categories):
            print('Classifying category `{}`...'.format(category))
            y_train = Y_train[j]  # (n * (1 - b))
            k = len(category_levels[j])
            classifiers = fit_ordinal(create_func, X_train, y_train, k)
            y_pred = predict_ordinal(classifiers, X_test, k)  # (n * b)
            y_test = Y_test[j]

            base_fname = '{:d}_{:d}'.format(stamp, j)
            logs_path = folders.ensure(os.path.join(folders.LOGS_PATH, model_name))
            with open(os.path.join(logs_path, '{}.txt'.format(base_fname)), 'w') as fd:
                fd.write('HYPERPARAMETERS\n')
                fd.write('\nText\n')
                fd.write('subset_ratio={}\n'.format(str(subset_ratio)))
                fd.write('subset_seed={}\n'.format(str(subset_seed)))
                fd.write('min_len={:d}\n'.format(min_len))
                fd.write('max_len={:d}\n'.format(max_len))
                fd.write('min_tokens={:d}\n'.format(min_tokens))
                fd.write('\nLabels\n')
                fd.write('categories_mode=\'{}\'\n'.format(categories_mode))
                fd.write('return_overall={}\n'.format(return_overall))
                fd.write('\nVectorization\n')
                fd.write('max_words={:d}\n'.format(max_words))
                fd.write('vectorizer={}\n'.format(vectorizer.__class__.__name__))
                fd.write('\nTraining\n')
                fd.write('test_size={}\n'.format(str(test_size)))
                fd.write('test_random_state={:d}\n'.format(test_random_state))
                fd.write('\nRESULTS\n\n')
                fd.write('Data size: {:d}\n'.format(X.shape[0]))
                fd.write('Train size: {:d}\n'.format(X_train.shape[0]))
                fd.write('Test size: {:d}\n\n'.format(X_test.shape[0]))
                evaluation.write_confusion_and_metrics(y_test, y_pred, fd, category)

            predictions_path = folders.ensure(os.path.join(folders.PREDICTIONS_PATH, model_name))
            with open(os.path.join(predictions_path, '{}.txt'.format(base_fname)), 'w') as fd:
                evaluation.write_predictions(y_test, y_pred, fd, category)

            for i, classifier in enumerate(classifiers):
                with open(os.path.join(model_path, '{}_{:d}i.pickle'.format(base_fname, j, i))) as fd:
                    pickle.dump(classifier, fd, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')


def identity(v):
    return v


def create_k_nearest_neighbors():
    return KNeighborsClassifier(n_neighbors=5, weights='distance', p=2, metric='minkowski')


def create_logistic_regression():
    return LogisticRegression(penalty='l2', solver='lbfgs', multi_class='ovr')


def create_multi_layer_perceptron():
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=128)


def create_multinomial_naive_bayes():
    return MultinomialNB(alpha=1.0, fit_prior=True)


def create_random_forest():
    return RandomForestClassifier(n_estimators=6, criterion='gini')


def create_svm():
    return LinearSVC()


def fit_ordinal(create_func, X, y, k):
    # Create and train a classifier for each ordinal index.
    y_train_ordinal = ordinal.to_multi_hot_ordinal(y, k=k)  # (n * (1 - b), k - 1)
    classifiers = [create_func() for _ in range(k - 1)]
    for i, classifier in enumerate(classifiers):
        classifier.fit(X, y_train_ordinal[:, i])
    return classifiers


def predict_ordinal(classifiers, X, k):
    try:
        n = len(X)
    except TypeError:
        n = X.shape[0]

    # Calculate probabilities for derived data sets.
    ordinal_p = np.zeros((n, k - 1))  # (n * b, k - 1)
    for i, classifier in enumerate(classifiers):
        ordinal_p[:, i] = classifier.predict(X)

    # Calculate the actual class label probabilities.
    p = np.zeros((n, k))  # (n * b, k)
    p[:, 0] = 1 - ordinal_p[:, 0]
    for i in range(1, k - 1):
        p[:, i] = ordinal_p[:, i - 1] * (1 - ordinal_p[:, i])
    p[:, k - 1] = ordinal_p[:, k - 2]

    # Choose the most likely class label.
    return np.argmax(p, axis=1)


if __name__ == '__main__':
    main()
