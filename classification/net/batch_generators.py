import operator

import numpy as np
from tensorflow.keras.utils import Sequence


class SingleInstanceBatchGenerator(Sequence):
    """
    When fitting the model, the batch size must be 1 to accommodate variable numbers of paragraphs per text.
    See `https://datascience.stackexchange.com/a/48814/66450`.
    """
    def __init__(self, X, Y, shuffle=True):
        super(SingleInstanceBatchGenerator, self).__init__()
        # `fit_generator` wants each `x` to be a NumPy array, not a list.
        self.X = [np.array([x]) for x in X]
        # Transform Y from shape (c, n, k-1) to (n, c, 1, k-1) if Y is ordinal,
        # or from shape (c, n, k) to (n, c, 1, k) if Y is not ordinal,
        # where `c` is the number of categories, `n` is the number of instances,
        # and `k` is the number of classes for the current category.
        self.Y = [[np.array([y[i]]) for y in Y]
                  for i in range(len(X))]
        self.indices = np.arange(len(X))
        self.shuffle = shuffle
        self.shuffle_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return self.X[i], self.Y[i]

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class VariableLengthBatchGenerator(Sequence):

    def __init__(self, X, X_shape, Y, Y_shape, batch_size, shuffle=True):
        super(VariableLengthBatchGenerator, self).__init__()
        # Sort X by length.
        lengths, indices = zip(*sorted([(len(x), index) for index, x in enumerate(X)], key=operator.itemgetter(0)))
        # Create batches by iterating over lengths in order.
        self.X_batches = []
        self.Y_batches = []
        n = len(X)
        i = 0
        while i * batch_size < n:
            b = min(batch_size, n - i * batch_size)  # Size of current batch.
            length = lengths[i * batch_size + b - 1]  # The last length in the batch is the longest
            X_batch = np.zeros((b, length, *X_shape), dtype=X[0].dtype)
            Y_batch = [np.zeros((b, *y_shape), dtype=Y[0].dtype) for y_shape in Y_shape]
            for z in range(b):
                index = indices[i * batch_size + z]
                x = X[index]
                X_batch[z, :len(x)] = x
                for j in range(len(Y)):
                    Y_batch[j][z] = Y[j][index]
            self.X_batches.append(X_batch)
            self.Y_batches.append(Y_batch)
            i += 1
        self.indices = np.arange(len(self.X_batches))
        self.shuffle = shuffle
        self.shuffle_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return self.X_batches[i], self.Y_batches[i]

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
