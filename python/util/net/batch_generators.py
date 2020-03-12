from imblearn.keras import BalancedBatchGenerator
import numpy as np
from tensorflow.keras.utils import Sequence

from python.util.shared_parameters import ordinal


class SingleInstanceBatchGenerator(Sequence):
    """
    When fitting the model, the batch size must be 1 to accommodate variable numbers of paragraphs per text.
    See `https://datascience.stackexchange.com/a/48814/66450`.
    """
    def __init__(self, X, Y, X_w=None, shuffle=True, row_dropout=0.0):
        super(SingleInstanceBatchGenerator, self).__init__()
        # `fit_generator` wants each `x` to be a NumPy array, not a list.
        self.X = [np.array([x]) for x in X]
        self.X_w = X_w
        # Transform Y from shape (c, n, k-1) to (n, c, 1, k-1) if Y is ordinal,
        # or from shape (c, n, k) to (n, c, 1, k) if Y is not ordinal,
        # where `c` is the number of categories, `n` is the number of instances,
        # and `k` is the number of classes for the current category.
        self.Y = [[np.array([y[i]]) for y in Y]
                  for i in range(len(X))]
        self.indices = np.arange(len(X))
        self.shuffle = shuffle
        self.shuffle_indices()
        self.row_dropout = row_dropout

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        x = self.X[i]
        dropout_mask = np.random.rand(len(self.X[i][0])) >= self.row_dropout
        x = x[:, dropout_mask]
        if self.X_w is not None:
            return [x, self.X_w[i]], self.Y[i]
        return x, self.Y[i]

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class OrdinalBalancedBatchGenerator(BalancedBatchGenerator):

    def __init__(self, X, y, transform_X, k, batch_size=32):
        super(OrdinalBalancedBatchGenerator, self).__init__(X, y, sample_weight=None, batch_size=batch_size)
        self.transform_X = transform_X
        self.k = k

    def __len__(self):
        return super(OrdinalBalancedBatchGenerator, self).__len__()

    def __getitem__(self, index):
        X, y = super(OrdinalBalancedBatchGenerator, self).__getitem__(index)
        return self.transform_X(X), ordinal.to_multi_hot_ordinal(y, k=self.k)
