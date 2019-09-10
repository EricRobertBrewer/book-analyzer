import numpy as np
from tensorflow.keras import utils


class SingleInstanceBatchGenerator(utils.Sequence):
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
