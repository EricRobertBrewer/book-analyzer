import os
import sys
import time

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import initializers as initializers, regularizers, constraints
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, Dropout, Embedding, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, GRU, Input, Layer, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

from classification import evaluation, ordinal
import folders
from sites.bookcave import bookcave
from text import load_embeddings


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(
            self,
            W_regularizer=None,
            u_regularizer=None,
            b_regularizer=None,
            W_constraint=None,
            u_constraint=None,
            b_constraint=None,
            bias=True,
            **kwargs
    ):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W = None
        self.b = None
        self.u = None

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        _dim = int(input_shape[-1])
        self.W = self.add_weight('{}_W'.format(self.name),
                                 (_dim, _dim,),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight('{}_b'.format(self.name),
                                     (_dim,),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight('{}_u'.format(self.name),
                                 (_dim,),
                                 initializer=self.init,
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, inp, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number (epsilon) to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def create_model(
        n_classes,
        n_tokens,
        embedding_matrix,
        embedding_trainable=False,
        word_rnn=GRU,
        word_rnn_units=128,
        word_dense_units=64,
        book_dense_units=1024,
        is_ordinal=True,
        dropout_regularizer=.5,
        l2_regularizer=None
):
    if l2_regularizer is None:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(l2_regularizer)

    # Word encoder.
    input_w = Input(shape=(n_tokens,), dtype='float32')  # (t)
    max_words, d = embedding_matrix.shape
    x_w = Embedding(max_words, d, weights=[embedding_matrix], trainable=embedding_trainable)(input_w)  # (t, d)
    x_w = Bidirectional(word_rnn(word_rnn_units, return_sequences=True, kernel_regularizer=kernel_regularizer))(x_w)  # (t, h_w)
    x_w = TimeDistributed(Dense(word_dense_units, kernel_regularizer=kernel_regularizer))(x_w)  # (2t, c_w)
    x_w = AttentionWithContext()(x_w)  # (c_w)
    word_encoder = Model(input_w, x_w)

    # Consider maximum and average signals among all paragraphs.
    input_p = Input(shape=(None, n_tokens), dtype='float32')  # (s, t); s is not constant!
    x_p = TimeDistributed(word_encoder)(input_p)  # (s, c_w)
    g_max = GlobalMaxPooling1D()(x_p)  # (c_w)
    g_avg = GlobalAveragePooling1D()(x_p)  # (c_w)
    x_p = Concatenate()([g_max, g_avg])  # (2c_w)
    x_p = Dense(book_dense_units, kernel_regularizer=kernel_regularizer)(x_p)  # (c_b)
    x_p = Dropout(dropout_regularizer)(x_p)  # (c_b)
    activation = 'sigmoid' if is_ordinal else 'softmax'
    outputs = [Dense(n - 1 if is_ordinal else n, activation=activation)(x_p)
               for n in n_classes]
    model = Model(input_p, outputs)
    return model


class SingleInstanceBatchGenerator(Sequence):
    """
    When fitting the model, the batch size must be 1 to accommodate variable numbers of paragraphs per text.
    See `https://datascience.stackexchange.com/a/48814/66450`.
    """
    def __init__(self, X, Y, shuffle=True):
        super(SingleInstanceBatchGenerator, self).__init__()
        self.X = [np.array([x]) for x in X]  # `fit_generator` wants each `x` to be a NumPy array, not a list.
        # Transform Y from shape (C, n, c-1) to (n, C, 1, c-1) where `C` is the number of categories,
        # `n` is the number of instances, and `c` is the number of classes for the current category.
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


def main():
    stamp = int(time.time())
    print('Time stamp: {:d}'.format(stamp))

    # Load data.
    if verbose:
        print('\nRetrieving texts...')
    min_len, max_len = 256, 4096
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'tokens'},
                          subset_ratio=1,
                          subset_seed=1,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=6)
    text_paragraph_tokens, _ = zip(*inputs['tokens'])
    if verbose:
        print('Retrieved {:d} texts.'.format(len(text_paragraph_tokens)))

    # Tokenize.
    if verbose:
        print('\nTokenizing...')
    max_words = 8192  # The maximum size of the vocabulary.
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_sentences = []
    for paragraph_tokens in text_paragraph_tokens:
        for tokens in paragraph_tokens:
            all_sentences.append(split.join(tokens))
    tokenizer.fit_on_texts(all_sentences)
    if verbose:
        print('Done.')

    # Convert to sequences.
    if verbose:
        print('\nConverting texts to sequences...')
    n_tokens = 128  # The maximum number of tokens to process in each paragraph.
    X = [np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in paragraph_tokens]),
                                maxlen=n_tokens,
                                padding='pre',
                                truncating='pre'))
         for paragraph_tokens in text_paragraph_tokens]
    if verbose:
        print('Done.')

    # Load embedding.
    if verbose:
        print('\nLoading embedding matrix...')
    embedding_matrix = load_embeddings.get_embedding(tokenizer, folders.EMBEDDING_GLOVE_100_PATH, max_words)
    if verbose:
        print('Done.')

    # Create model.
    if verbose:
        print('\nCreating model...')
    n_classes = [len(levels) for levels in category_levels]
    embedding_trainable = False
    word_rnn = GRU
    word_rnn_units = 128
    word_dense_units = 64
    book_dense_units = 512
    is_ordinal = True
    model = create_model(
        n_classes,
        n_tokens,
        embedding_matrix,
        embedding_trainable=embedding_trainable,
        word_rnn=word_rnn,
        word_rnn_units=word_rnn_units,
        word_dense_units=word_dense_units,
        book_dense_units=book_dense_units,
        is_ordinal=is_ordinal)
    if verbose:
        print(model.summary())

    # Split data set.
    if verbose:
        print('\nSplitting data into training and test sets...')
    test_size = .25  # b
    test_random_state = 1
    val_size = .1  # v
    val_random_state = 1
    YT = Y.transpose()  # (n, C)
    X_train, X_test, YT_train, YT_test = train_test_split(X, YT, test_size=test_size, random_state=test_random_state)
    X_train, X_val, YT_train, YT_val = train_test_split(X_train, YT_train, test_size=val_size, random_state=val_random_state)
    Y_train = YT_train.transpose()  # (C, n * (1 - b) * (1 - v))
    Y_val = YT_val.transpose()  # (C, n * (1 - b) * v)
    Y_test = YT_test.transpose()  # (C, n * b)
    if verbose:
        print('Done.')

    # Weight classes inversely proportional to their frequency.
    class_weights = []
    for category_i, y_train in enumerate(Y_train):
        bincount = np.bincount(y_train, minlength=n_classes[category_i])
        class_weight = {i: 1 / (count + 1) for i, count in enumerate(bincount)}
        class_weights.append(class_weight)

    # Train.
    optimizer = Adam()
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    Y_train_ordinal = [ordinal.to_multi_hot_ordinal(Y_train[i], n_classes=n) for i, n in enumerate(n_classes)]
    train_generator = SingleInstanceBatchGenerator(X_train, Y_train_ordinal, shuffle=True)
    Y_val_ordinal = [ordinal.to_multi_hot_ordinal(Y_val[i], n_classes=n) for i, n in enumerate(n_classes)]
    val_generator = SingleInstanceBatchGenerator(X_val, Y_val_ordinal, shuffle=False)
    history = model.fit_generator(train_generator,
                                  epochs=epochs,
                                  verbose=verbose,
                                  validation_data=val_generator,
                                  class_weight=class_weights)

    # Save the history to visualize loss over time.
    if verbose:
        print('\nSaving training history...')
    history_fname = 'book_net-{:d}.txt'.format(stamp)
    with open(os.path.join(folders.HISTORY_PATH, history_fname), 'w') as fd:
        for key in history.history.keys():
            values = history.history.get(key)
            fd.write('{} {}\n'.format(key, ' '.join(str(value) for value in values)))
    if verbose:
        print('Done.')

    # Evaluate.
    test_generator = SingleInstanceBatchGenerator(X_test, Y_train_ordinal, shuffle=False)
    Y_preds_ordinal = model.predict_generator(test_generator, steps=len(test_generator))
    Y_preds = [ordinal.from_multi_hot_ordinal(y_ordinal, threshold=.5) for y_ordinal in Y_preds_ordinal]
    for category_i, category in enumerate(categories):
        print('\n`{}`'.format(category))
        evaluation.print_metrics(Y_test[category_i], Y_preds[category_i])


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise ValueError('Usage: <epochs> [verbose]')
    epochs = int(sys.argv[1])
    if len(sys.argv) > 2:
        verbose = int(sys.argv[2])
    else:
        verbose = 0
    main()
