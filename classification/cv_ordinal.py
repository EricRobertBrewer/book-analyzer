from imblearn.ensemble import BalancedBaggingClassifier
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sites.bookcave import bookcave
from classification import ordinal


def cross_validate(get_base, folds, X, Y, categories, category_levels, seed=None, verbose=0):
    # Validate parameters.
    if folds < 2:
        raise ValueError('Parameter `folds` must be greater than 1. Received: {:d}'.format(folds))

    for category_index, category in enumerate(categories):
        if verbose:
            print()
            print('Classifying category `{}`...'.format(category))

        category_size = len(category_levels[category_index])
        y = Y[category_index]

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
            p = ordinal.get_simple_ordinal_proba(get_classifier, get_base, category_size, X_train, X_test, y_train, y_test)

            # Choose the most likely class label.
            y_pred = np.argmax(p, axis=1)

            num_correct = accuracy_score(y_test, y_pred, normalize=False)
            if verbose:
                print('Accuracy: {:.4} ({:d}/{:d})'.format(num_correct / len(y_test), int(num_correct), len(y_test)))
            num_correct_total += num_correct
            num_total += len(y_test)

            y_test_all.extend(y_test)
            y_pred_all.extend(list(y_pred))

            confusion = confusion_matrix(y_test, y_pred, labels=list(range(category_size)))
            if verbose:
                print(confusion)
            confusion_total += confusion

        accuracy_total = num_correct_total/num_total
        print()
        print('`{}` ({:d} levels)'.format(category, category_size))
        print('Overall accuracy: {:.4}'.format(accuracy_total))
        print(confusion_total)


def get_classifier(get_base, options):
    y_train = options['y_train']
    bincount = np.bincount(y_train)
    n_estimators = max(bincount) // min(bincount)
    return BalancedBaggingClassifier(
        base_estimator=get_base(),
        n_estimators=min(n_estimators, 8),
        bootstrap_features=True,
        sampling_strategy='not minority',
        replacement=True)


def main():
    verbose = 1

    def identity(v):
        return v

    vectorizer = TfidfVectorizer(
        preprocessor=identity,
        tokenizer=identity,
        analyzer='word',
        token_pattern=None,
        max_features=4096,
        norm='l2',
        sublinear_tf=True)

    min_len, max_len = 250, 7500
    inputs, Y, categories, category_levels =\
        bookcave.get_data({'tokens'},
                          min_len=min_len,
                          max_len=max_len)
    text_paragraph_tokens = [paragraph_tokens for paragraph_tokens, _ in inputs['tokens']]
    text_tokens = []
    for paragraph_tokens in text_paragraph_tokens:
        all_tokens = []
        for tokens in paragraph_tokens:
            all_tokens.extend(tokens)
        text_tokens.append(all_tokens)

    # Create vectorized representations of the book texts.
    if verbose:
        print()
        print('Vectorizing text...')
    X = vectorizer.fit_transform(text_tokens)
    if verbose:
        print('Vectorized text with {:d} unique words.'.format(len(vectorizer.get_feature_names())))

    def get_mnb():
        return MultinomialNB(fit_prior=True)

    def get_lr():
        return LogisticRegression(solver='lbfgs')

    def get_rf():
        return RandomForestClassifier(n_estimators=6)

    def get_svm():
        return LinearSVC()

    folds = 3
    seed = 1
    get_bases = [get_mnb, get_lr, get_rf, get_svm]
    base_names = ['mnb', 'lr', 'rf', 'svm']
    for i, get_base in enumerate(get_bases):
        print()
        print('='*72)
        print()
        print(base_names[i])
        print()
        cross_validate(get_base, folds, X, Y, categories, category_levels, seed=seed, verbose=1)


if __name__ == '__main__':
    main()
