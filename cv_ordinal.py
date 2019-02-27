# Math.
import numpy as np
# Learning.
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
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


def get_ordinal_proba(get_classifier, options: dict, size, X_train, X_test, y_train, y_test):
    # Get probabilities for binarized ordinal labels.
    ordinal_p = np.zeros((len(y_test), size - 1))
    for ordinal_index in range(size - 1):
        # Find P(Target > Class_k) for 0..(k-1)
        classifier = get_classifier(options)
        y_train_ordinal = to_ordinal(y_train, ordinal_index)
        classifier.fit(X_train, y_train_ordinal)
        ordinal_p[:, ordinal_index] = classifier.predict(X_test)

    # Calculate the actual class label probabilities.
    p = np.zeros((len(y_test), size))
    for i in range(size):
        if i == 0:
            p[:, i] = 1 - ordinal_p[:, 0]
        elif i == size - 1:
            p[:, i] = ordinal_p[:, i - 1]
        else:
            p[:, i] = ordinal_p[:, i - 1] - ordinal_p[:, i]
    return p


def cross_validate(vectorizer, get_classifier, folds, inputs, Y, categories, levels, seed=None, verbose=False):
    # Validate parameters.
    if folds < 2:
        raise ValueError('Parameter `folds` must be greater than 1. Received: {:d}'.format(folds))

    # Create vectorized representations of the book texts.
    if verbose:
        print('Vectorizing text...')
    X = vectorizer.fit_transform(inputs)
    if verbose:
        print('Vectorized text with {:d} unique words.'.format(len(vectorizer.get_feature_names())))

    for category_index, category in enumerate(categories):
        if verbose:
            print('Classifying category `{}`...'.format(category))

        category_size = len(levels[category_index])
        y = Y[:, category_index]

        # Keep track of overall accuracy.
        num_correct_total = 0
        num_total = 0

        # And keep track of the confusion matrix over all folds.
        confusion_total = np.zeros((category_size, category_size), dtype=np.int32)

        # Split data into train and test sets for this fold.
        skf = StratifiedKFold(n_splits=folds, random_state=seed)
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            options = {'category_index': category_index}
            p = get_ordinal_proba(get_classifier, options, category_size, X_train, X_test, y_train, y_test)

            # Choose the most likely class label.
            y_pred = np.argmax(p, axis=1)

            num_correct = accuracy_score(y_test, y_pred, normalize=False)
            if verbose:
                print('Accuracy for fold {:d}: {:.3%} ({:d}/{:d})'.format(fold + 1,
                                                                          num_correct / len(y_test),
                                                                          int(num_correct),
                                                                          len(y_test)))
            num_correct_total += num_correct
            num_total += len(y_test)

            confusion = confusion_matrix(y_test, y_pred, labels=list(range(category_size)))
            if verbose:
                print(confusion)
            confusion_total += confusion

        accuracy = num_correct_total/num_total
        print('`{}` ({:d} levels) overall accuracy: {:.3%}'.format(category, category_size, accuracy))
        print(confusion_total)


def main():
    for source in ['description', 'preview', 'book']:
        vectorizer = TfidfVectorizer(
            input='content' if source == 'description' else 'filename',
            encoding='utf-8',
            stop_words='english',
            ngram_range=(1, 2),
            min_df=4,
            max_features=8000,
            norm='l2',
            sublinear_tf=True)

        inputs, Y, categories, levels = bookcave.get_data(
            media={'text'},
            text_source=source,
            text_input='filename',
            text_min_len=6,
            categories_mode='soft',
            combine_ratings='max',
            verbose=True)

        # Collect the parameters to use a balanced ensemble classifier.
        ensemble_sizes = []
        for category_index, category in enumerate(categories):
            y = Y[:, category_index]
            bincount = np.bincount(y)
            max_index = np.argmax(bincount)
            min_index = np.argmin(bincount)
            min_size = bincount[min_index]
            ensemble_sizes.append(int(bincount[max_index] / min_size))
        # print(ensemble_sizes)

        def get_classifier(options: dict):
            return BalancedBaggingClassifier(base_estimator=MultinomialNB(fit_prior=True),
                                             n_estimators=ensemble_sizes[options['category_index']],
                                             sampling_strategy='not minority')

        folds = 5
        seed = 1
        cross_validate(vectorizer, get_classifier, folds, inputs, Y, categories, levels, seed=seed, verbose=True)


if __name__ == '__main__':
    main()
