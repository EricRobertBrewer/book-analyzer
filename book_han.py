from __future__ import absolute_import, division, print_function, with_statement
import keras
from keras import backend as K
from keras import initializers as initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.layers import Bidirectional, Dense, Dropout, Embedding, GRU, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import sys

from classification import evaluation, ordinal
import folders
from sites.bookcave import bookcave
from text import load_embeddings


def patch_tokenizer():
    """
    https://github.com/keras-team/keras/issues/1072#issuecomment-295470970
    """

    def text_to_word_sequence(text,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True,
                              split=' '):
        if lower:
            text = text.lower()
        if type(text) == unicode:
            translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
        else:
            translate_table = keras.preprocessing.text.maketrans(filters, split * len(filters))
        text = text.translate(translate_table)
        seq = text.split(split)
        return [i for i in seq if i]

    keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence


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
    else:
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

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
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
        n_paragraphs,
        n_tokens,
        embedding_matrix,
        embedding_trainable=False,
        rnn=GRU,
        rnn_units=128,
        dense_units=64,
        is_ordinal=True,
        dropout_regularizer=.5,
        l2_regularizer=None
):
    if l2_regularizer is None:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(l2_regularizer)

    # Word encoder.
    input_w = Input(shape=(n_tokens,), dtype='float32')
    embed_shape = embedding_matrix.shape
    x_w = Embedding(embed_shape[0], embed_shape[1], weights=[embedding_matrix], trainable=embedding_trainable)(input_w)
    x_w = Bidirectional(rnn(rnn_units, return_sequences=True, kernel_regularizer=kernel_regularizer))(x_w)
    x_w = TimeDistributed(Dense(dense_units, kernel_regularizer=kernel_regularizer))(x_w)
    x_w = AttentionWithContext()(x_w)
    word_encoder = Model(input_w, x_w)

    # Sentence encoder.
    input_s = Input(shape=(n_paragraphs, n_tokens), dtype='float32')
    x_s = TimeDistributed(word_encoder)(input_s)
    x_s = Bidirectional(rnn(rnn_units, return_sequences=True, kernel_regularizer=kernel_regularizer))(x_s)
    x_s = TimeDistributed(Dense(dense_units, kernel_regularizer=kernel_regularizer))(x_s)
    x_s = AttentionWithContext()(x_s)
    x_s = Dropout(dropout_regularizer)(x_s)
    activation = 'sigmoid' if is_ordinal else 'softmax'
    outputs = [Dense(n - 1 if is_ordinal else n, activation=activation)(x_s)
               for n in n_classes]
    model = Model(input_s, outputs)
    return model


def main():
    # Load data.
    if verbose:
        print('\nRetrieving texts...')
    min_len = 250  # The minimum number of paragraphs in each text.
    min_tokens = 6  # The minimum number of tokens in each paragraph.
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'tokens'},
                          subset_ratio=1,
                          subset_seed=1,
                          min_len=min_len,
                          min_tokens=min_tokens)
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
    patch_tokenizer()
    tokenizer.fit_on_texts(all_sentences)
    if verbose:
        print('Done.')

    # Convert to sequences.
    if verbose:
        print('\nConverting texts to sequences...')
    n_paragraphs = 1024  # The maximum number of paragraphs to process in each text.
    n_tokens = 128  # The maximum number of tokens to process in each paragraph.
    X = np.zeros((len(text_paragraph_tokens), n_paragraphs, n_tokens), dtype=np.float32)
    for text_i, paragraph_tokens in enumerate(text_paragraph_tokens):
        if len(paragraph_tokens) > n_paragraphs:
            # Truncate two thirds of the remainder at the beginning and one third at the end.
            start = int(2/3*(len(paragraph_tokens) - n_paragraphs))
            usable_paragraph_tokens = paragraph_tokens[start:start+n_paragraphs]
        else:
            usable_paragraph_tokens = paragraph_tokens
        sequences = tokenizer.texts_to_sequences([split.join(tokens) for tokens in usable_paragraph_tokens])
        X[text_i, :len(sequences)] = pad_sequences(sequences, maxlen=n_tokens, padding='pre', truncating='pre')
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
    rnn = GRU
    rnn_units = 128
    dense_units = 64
    is_ordinal = True
    model = create_model(
        n_classes,
        n_paragraphs,
        n_tokens,
        embedding_matrix,
        embedding_trainable=embedding_trainable,
        rnn=rnn,
        rnn_units=rnn_units,
        dense_units=dense_units,
        is_ordinal=is_ordinal)
    if verbose:
        print(model.summary())

    # Split data set.
    if verbose:
        print('\nSplitting data into training and test sets...')
    test_size = .25  # b
    random_state = 1
    YT = Y.transpose()  # (num_texts, C)
    X_train, X_test, YT_train, YT_test = train_test_split(X, YT, test_size=test_size, random_state=random_state)
    Y_train, Y_test = YT_train.transpose(), YT_test.transpose()  # (C, (1 - b) * num_texts), (C, b * num_texts)
    if verbose:
        print('Done.')

    # Weight classes inversely proportional to their frequency.
    class_weights = []
    for category_i, y_train in enumerate(Y_train):
        bincount = np.bincount(y_train, minlength=n_classes[category_i])
        class_weight = {i: 1 / (count + 1) for i, count in enumerate(bincount)}
        class_weights.append(class_weight)

    # Train.
    batch_size = 32
    optimizer = Adam()
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    Y_train_ordinal = [ordinal.to_multi_hot_ordinal(Y_train[i], n_classes=n) for i, n in enumerate(n_classes)]
    history = model.fit(X_train,
                        Y_train_ordinal,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        class_weight=class_weights)

    # Evaluate.
    Y_preds_ordinal = model.predict(X_test)
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
