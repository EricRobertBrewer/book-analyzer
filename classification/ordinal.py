from __future__ import absolute_import, division, print_function, with_statement
import numpy as np


def to_simple_ordinal(y, i):
    return np.array([1 if value > i else 0 for value in y])


def get_simple_ordinal_proba(get_classifier, get_base, size, X_train, X_test, y_train, y_test):
    """
    # See `Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification."
    # European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2001.`.
    """
    # Get probabilities for binarized ordinal labels.
    ordinal_p = np.zeros((len(y_test), size - 1))
    for i in range(size - 1):
        # Find P(Target > Class_k) for 0..(k-1)
        y_train_ordinal = to_simple_ordinal(y_train, i)
        classifier = get_classifier(get_base, {'y_train': y_train_ordinal})
        classifier.fit(X_train, y_train_ordinal)
        ordinal_p[:, i] = classifier.predict(X_test)

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


def to_multi_hot_ordinal(y, n_classes=None):
    """
    See `http://orca.st.usm.edu/~zwang/files/rank.pdf`.
    """
    if n_classes is None:
        n_classes = max(y) + 1
    return np.array([[1 if value > i else 0 for i in range(n_classes - 1)] for value in y])


def from_multi_hot_ordinal(y_ordinal, threshold=0.5):
    y = []
    for values in y_ordinal:
        i = 0
        while i < len(values) and values[i] > threshold:
            i += 1
        y.append(i)
    return np.array(y)


def main():
    y = np.array([0, 1, 2, 3, 2])
    num_classes = max(y)
    print('y=\n{}'.format(y))
    y_invert = num_classes - y
    print('y_invert=\n{}'.format(y_invert))
    y_ordinal = to_multi_hot_ordinal(y)
    print('y_ordinal=\n{}'.format(y_ordinal))
    y_ordinal_invert = to_multi_hot_ordinal(y_invert)
    print('y_ordinal_invert=\n{}'.format(y_ordinal_invert))
    y_from = from_multi_hot_ordinal(y_ordinal)
    print('y_from=\n{}'.format(y_from))
    y_from_invert = from_multi_hot_ordinal(y_ordinal_invert)
    print('y_from_invert=\n{}'.format(y_from_invert))


if __name__ == '__main__':
    main()
