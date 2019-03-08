import numpy as np


def to_simple_ordinal(y, ordinal_index):
    # and use ordinal classification as explained in:
    # `Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification."
    # European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2001.`.
    return np.array([1 if level > ordinal_index else 0 for level in y])


def get_simple_ordinal_proba(get_classifier, size, X_train, X_test, y_train, y_test):
    # Get probabilities for binarized ordinal labels.
    ordinal_p = np.zeros((len(y_test), size - 1))
    for ordinal_index in range(size - 1):
        # Find P(Target > Class_k) for 0..(k-1)
        y_train_ordinal = to_simple_ordinal(y_train, ordinal_index)
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


def to_multi_hot_ordinal(y, num_classes=None, reverse=False):
    """
    See `http://orca.st.usm.edu/~zwang/files/rank.pdf`.
    """
    if num_classes is None:
        num_classes = max(y) + 1
    if reverse:
        return np.array([[1 if value <= i else 0 for i in range(num_classes - 1)] for value in y])
    return np.array([[1 if value > i else 0 for i in range(num_classes - 1)] for value in y])


def from_multi_hot_ordinal(y_ordinal, threshold=0.5, reverse=False):
    y = []
    for values in y_ordinal:
        i = 0
        if reverse:
            while i < len(values) and values[i] <= threshold:
                i += 1
        else:
            while i < len(values) and values[i] > threshold:
                i += 1
        y.append(i)
    return np.array(y)


def main():
    y = [0, 1, 2, 3]
    print('y=\n{}'.format(y))
    y_ordinal = to_multi_hot_ordinal(y)
    print('y_ordinal=\n{}'.format(y_ordinal))
    y_ordinal_reverse = to_multi_hot_ordinal(y, reverse=True)
    print('y_ordinal_reverse=\n{}'.format(y_ordinal_reverse))
    y_from = from_multi_hot_ordinal(y_ordinal)
    print('y_from=\n{}'.format(y_from))
    y_from_reverse = from_multi_hot_ordinal(y_ordinal_reverse, reverse=True)
    print('y_from_reverse=\n{}'.format(y_from_reverse))


if __name__ == '__main__':
    main()
