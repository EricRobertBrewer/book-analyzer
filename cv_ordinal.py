# Math.
import numpy as np
# Learning.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
# Data.
import bookcave


def get_train_test_split(x, y, fold, folds, seed=None):
    # Generate a random permutation in order to process the data set in a random order.
    if seed:
        np.random.seed(seed)
    perm = np.random.permutation(len(y))
    # Cross validate...
    test_start = len(y) * fold // folds
    test_end = len(y) * (fold + 1) // folds
    perm_train = np.concatenate((perm[:test_start], perm[test_end:]))
    perm_test = perm[test_start:test_end]
    x_train = x[perm_train]
    x_test = x[perm_test]
    y_train = y[perm_train]
    y_test = y[perm_test]
    return x_train, x_test, y_train, y_test


def to_ordinal(y, ordinal_index):
    # and use ordinal classification as explained in:
    # `Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification."
    # European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2001.`.
    return np.array([1 if level > ordinal_index else 0 for level in y])


def cross_validate(get_vectorizer, get_classifier, folds, inputs, Y, categories, levels, seed=None, verbose=False):
    # Start cross-validation.
    num_correct_totals = np.zeros(len(categories))

    if folds < 2:
        raise ValueError('Parameter `folds` must be greater than 1. Received: {:d}'.format(folds))
    # Start looping through folds before categories because text vectorization is the most time-consuming operation.
    for fold in range(folds):
        if verbose:
            print('Starting fold {}...'.format(fold + 1))

        # Split data into train and test sets for this fold.
        inputs_train, inputs_test, Y_train, Y_test = get_train_test_split(inputs, Y, fold, folds, seed=seed)

        # Create vectorized representations of the book texts.
        if verbose:
            print('Vectorizing text...')
        vectorizer = get_vectorizer()
        vectorizer.fit(inputs_train)  # Be fair, as if we were only allowed to model the training data.
        x_train = vectorizer.transform(inputs_train)
        x_test = vectorizer.transform(inputs_test)

        if verbose:
            print('Classifying...')
        for category_index, category in enumerate(categories):
            category_size = len(levels[category_index])
            y_train = Y_train[:, category_index]
            y_test = Y_test[:, category_index]

            # Get probabilities for binarized ordinal labels.
            ordinal_ps = np.zeros((len(y_test), category_size - 1))
            for ordinal_index in range(category_size - 1):
                # Find P(Target > Class_k) for 0..(k-1)
                classifier = get_classifier()
                y_train_ordinal = to_ordinal(y_train, ordinal_index)
                classifier.fit(x_train, y_train_ordinal)
                ordinal_ps[:, ordinal_index] = classifier.predict(x_test)

            # Calculate the actual class label probabilities.
            ps = np.zeros((len(y_test), category_size))
            for level_index in range(category_size):
                if level_index == 0:
                    ps[:, level_index] = 1 - ordinal_ps[:, 0]
                elif level_index == category_size - 1:
                    ps[:, level_index] = ordinal_ps[:, level_index - 1]
                else:
                    ps[:, level_index] = ordinal_ps[:, level_index - 1] - ordinal_ps[:, level_index]

            # Choose the most likely class label.
            y_pred = np.argmax(ps, axis=1)
            num_correct = accuracy_score(y_test, y_pred, normalize=False)
            num_correct_totals[category_index] += num_correct

    print('Overall accuracies for {:d} categories:'.format(len(categories)))
    accuracies = num_correct_totals/len(Y)
    for i, accuracy in enumerate(accuracies):
        print('`{}` ({:d} levels) overall accuracy: {:.4%}'.format(categories[i], len(levels[i]), accuracy))


def main():
    def get_vectorizer():
        return TfidfVectorizer(
            input='filename',
            encoding='utf-8',
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            norm='l2',
            sublinear_tf=True)

    def get_classifier():
        return MultinomialNB(fit_prior=True)

    folds = 5
    inputs, Y, categories, levels = bookcave.get_data(
        input='filename',
        combine_ratings='max',
        categories_mode='soft',
        verbose=True)
    seed = 1
    cross_validate(get_vectorizer, get_classifier, folds, inputs, Y, categories, levels, seed=seed, verbose=True)


if __name__ == '__main__':
    main()
