__all__ = [
    'to_multi_hot_ordinal',
    'from_multi_hot_ordinal'
]

import numpy as np


def _to_multi_hot_ordinal(value, k):
    return [1 if value > i else 0 for i in range(k - 1)]


def to_multi_hot_ordinal(y, k=None):
    """
    See `Frank, Eibe, and Mark Hall. "A simple approach to ordinal util."
    European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2001.`.
    Also see `http://orca.st.usm.edu/~zwang/files/rank.pdf`.
    """
    if k is None:
        k = max(y) + 1
    return np.array([_to_multi_hot_ordinal(value, k) for value in y])


def _from_multi_hot_ordinal(value_ordinal, threshold=.5):
    i = 0
    while i < len(value_ordinal) and value_ordinal[i] > threshold:
        i += 1
    return i


def from_multi_hot_ordinal(y_ordinal, threshold=.5):
    return np.array([_from_multi_hot_ordinal(value_ordinal, threshold) for value_ordinal in y_ordinal])


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
