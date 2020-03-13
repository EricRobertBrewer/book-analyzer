from imblearn.keras import BalancedBatchGenerator
from keras.utils import Sequence
import numpy as np


class SimpleBatchGenerator(Sequence):

    def __init__(self, X, y, batch_size=32, shuffle=True):
        super(SimpleBatchGenerator, self).__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(len(y))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.X[indices], self.y[indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class SingleInstanceBatchGenerator(Sequence):
    """
    When fitting the model, the easiest way to accommodate variable numbers of paragraphs per text
    is to set the batch size to 1.
    See `https://datascience.stackexchange.com/a/48814/66450`.
    """
    def __init__(self, X, y, shuffle=True):
        super(SingleInstanceBatchGenerator, self).__init__()
        # `fit_generator` wants each `x` to be a NumPy array, not a list.
        self.X = [np.array([x]) for x in X]
        self.y = [np.array([value]) for value in y]
        self.indices = np.arange(len(X))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return self.X[i], self.y[i]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class TransformBalancedBatchGenerator(BalancedBatchGenerator):

    def __init__(self, X, y, transform_X=None, transform_y=None, batch_size=32, **kwargs):
        super(TransformBalancedBatchGenerator, self).__init__(X, y, batch_size=batch_size)
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.kwargs = kwargs

    def __len__(self):
        return super(TransformBalancedBatchGenerator, self).__len__()

    def __getitem__(self, index):
        X, y = super(TransformBalancedBatchGenerator, self).__getitem__(index)
        if self.transform_X is not None:
            X = self.transform_X(X, **self.kwargs)
        if self.transform_y is not None:
            y = self.transform_y(y, **self.kwargs)
        return X, y
