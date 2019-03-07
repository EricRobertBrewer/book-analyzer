# Math.
import numpy as np
# Learning.
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
# Data.
import bookcave


def to_ordinal(y, ordinal_index):
    # and use ordinal classification as explained in:
    # `Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification."
    # European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2001.`.
    return np.array([1 if level > ordinal_index else 0 for level in y])


def get_ordinal_proba(get_classifier, size, X_train, X_test, y_train, y_test):
    # Get probabilities for binarized ordinal labels.
    ordinal_p = np.zeros((len(y_test), size - 1))
    for ordinal_index in range(size - 1):
        # Find P(Target > Class_k) for 0..(k-1)
        y_train_ordinal = to_ordinal(y_train, ordinal_index)
        classifier = get_classifier({'y_train': y_train_ordinal})
        try:
            classifier.fit(X_train, y_train_ordinal)
        except ValueError:
            print('ValueError')
            bincount = np.bincount(y_train, minlength=size)
            print('bincount:')
            for i, count in enumerate(bincount):
                print('{:d}: {:d}'.format(i, count))
            print()
            print('y_train -> y_train_ordinal')
            print('-' * 8)
            for i in range(len(y_train)):
                print('{:d} -> {:d}'.format(y_train[i], y_train_ordinal[i]))
            print('Exiting.')
            exit(1)
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


def cross_validate(vectorizer, get_classifier, folds, texts, Y, categories, levels, seed=None, verbose=False):
    # Validate parameters.
    if folds < 2:
        raise ValueError('Parameter `folds` must be greater than 1. Received: {:d}'.format(folds))

    # Create vectorized representations of the book texts.
    if verbose:
        print()
        print('Vectorizing text...')
    X = vectorizer.fit_transform(texts)
    if verbose:
        print('Vectorized text with {:d} unique words.'.format(len(vectorizer.get_feature_names())))

    for category_index, category in enumerate(categories):
        if verbose:
            print()
            print('Classifying category `{}`...'.format(category))

        category_size = len(levels[category_index])
        y = Y[:, category_index]

        # Keep track of overall accuracy, precision, recall, and F1.
        num_correct_total = 0
        num_total = 0
        y_test_all = []
        y_pred_all = []
        score_average = 'micro'

        # And keep track of the confusion matrix over all folds.
        confusion_total = np.zeros((category_size, category_size), dtype=np.int32)

        # Split data into train and test sets for this fold.
        skf = StratifiedKFold(n_splits=folds, random_state=seed)
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            if verbose:
                print()
                print('Starting fold {:d}...'.format(fold + 1))

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            p = get_ordinal_proba(get_classifier, category_size, X_train, X_test, y_train, y_test)

            # Choose the most likely class label.
            y_pred = np.argmax(p, axis=1)

            num_correct = accuracy_score(y_test, y_pred, normalize=False)
            if verbose:
                print('Accuracy: {:.3%} ({:d}/{:d})'.format(num_correct / len(y_test), int(num_correct), len(y_test)))
            num_correct_total += num_correct
            num_total += len(y_test)

            if verbose:
                precision = precision_score(y_test, y_pred, average=score_average)
                recall = recall_score(y_test, y_pred, average=score_average)
                f1 = f1_score(y_test, y_pred, average=score_average)
                print('Precision: {:.3%}'.format(precision))
                print('Recall: {:.3%}'.format(recall))
                print('F1: {:.3%}'.format(f1))
            y_test_all.extend(y_test)
            y_pred_all.extend(list(y_pred))

            confusion = confusion_matrix(y_test, y_pred, labels=list(range(category_size)))
            if verbose:
                print(confusion)
            confusion_total += confusion

        accuracy_total = num_correct_total/num_total
        precision_total = precision_score(y_test_all, y_pred_all, average=score_average)
        recall_total = recall_score(y_test_all, y_pred_all, average=score_average)
        f1_total = f1_score(y_test_all, y_pred_all, average=score_average)
        print()
        print('`{}` ({:d} levels)'.format(category, category_size))
        print('Overall accuracy: {:.3%}'.format(accuracy_total))
        print('Overall precision: {:.3%}'.format(precision_total))
        print('Overall recall: {:.3%}'.format(recall_total))
        print('Overall F1: {:.3%}'.format(f1_total))
        print(confusion_total)


def main():
    vectorizer = TfidfVectorizer(
        input='filename',
        encoding='utf-8',
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_features=4096,
        norm='l2',
        sublinear_tf=True)

    inputs, Y, categories, levels = bookcave.get_data(
        media={'text'},
        text_source='book',
        text_input='filename',
        text_min_len=6,
        only_categories={1, 3, 5, 6},
        verbose=True)
    texts = inputs['text']

    def get_classifier(options: dict):
        y_train = options['y_train']
        bincount = np.bincount(y_train)
        n_estimators = max(bincount) // min(bincount)
        print('max={:d}; min={:d}; len={:d}; n_estimators={:d}'.format(max(bincount), min(bincount), len(y_train), n_estimators))
        return BalancedBaggingClassifier(
            base_estimator=MultinomialNB(fit_prior=True),
            n_estimators=min(32, n_estimators),
            bootstrap_features=True,
            sampling_strategy='not minority',
            replacement=True)

    folds = 3
    seed = 1
    cross_validate(vectorizer, get_classifier, folds, texts, Y, categories, levels, seed=seed, verbose=True)


if __name__ == '__main__':
    main()
