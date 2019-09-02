import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers as initializers, regularizers, constraints
from tensorflow.keras.layers import Bidirectional, Concatenate, CuDNNGRU, CuDNNLSTM, Dense, Dropout, Embedding, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, Input, Layer, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical
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
        output_names,
        n_tokens,
        embedding_matrix,
        embedding_trainable=False,
        word_rnn=GRU,
        word_rnn_units=128,
        word_rnn_l2=.01,
        word_dense_units=64,
        word_dense_activation='linear',
        word_dense_l2=.01,
        book_dense_units=512,
        book_dense_activation='relu',
        book_dense_l2=.01,
        book_dropout=.5,
        is_ordinal=True
):
    # Word encoder.
    input_w = Input(shape=(n_tokens,), dtype='float32')  # (t)
    max_words, d = embedding_matrix.shape
    x_w = Embedding(max_words,
                    d,
                    weights=[embedding_matrix],
                    trainable=embedding_trainable)(input_w)  # (t, d)
    x_w = Bidirectional(word_rnn(word_rnn_units,
                                 kernel_regularizer=regularizers.l2(word_rnn_l2),
                                 return_sequences=True))(x_w)  # (t, h_w)
    x_w = TimeDistributed(Dense(word_dense_units,
                                activation=word_dense_activation,
                                kernel_regularizer=regularizers.l2(word_dense_l2)))(x_w)  # (2t, c_w)
    x_w = AttentionWithContext()(x_w)  # (c_w)
    word_encoder = Model(input_w, x_w)

    # Consider maximum and average signals among all paragraphs of books.
    input_p = Input(shape=(None, n_tokens), dtype='float32')  # (s, t); s is not constant!
    x_p = TimeDistributed(word_encoder)(input_p)  # (s, c_w)
    g_max_p = GlobalMaxPooling1D()(x_p)  # (c_w)
    g_avg_p = GlobalAveragePooling1D()(x_p)  # (c_w)
    x_p = Concatenate()([g_max_p, g_avg_p])  # (2c_w)
    x_p = Dense(book_dense_units,
                activation=book_dense_activation,
                kernel_regularizer=regularizers.l2(book_dense_l2))(x_p)  # (c_b)
    x_p = Dropout(book_dropout)(x_p)  # (c_b)
    outputs = [Dense(n - 1 if is_ordinal else n,
                     activation='sigmoid' if is_ordinal else 'softmax',
                     name=output_names[i])(x_p)
               for i, n in enumerate(n_classes)]
    model = Model(input_p, outputs)
    return model


class SingleInstanceBatchGenerator(Sequence):
    """
    When fitting the model, the batch size must be 1 to accommodate variable numbers of paragraphs per text.
    See `https://datascience.stackexchange.com/a/48814/66450`.
    """
    def __init__(self, X, Y, sample_weights=None, shuffle=True):
        super(SingleInstanceBatchGenerator, self).__init__()
        # `fit_generator` wants each `x` to be a NumPy array, not a list.
        self.X = [np.array([x]) for x in X]
        # Transform Y from shape (C, n, c-1) to (n, C, 1, c-1) if Y is ordinal,
        # or from shape (C, n, c) to (n, C, 1, c) if Y is not ordinal,
        # where `C` is the number of categories, `n` is the number of instances,
        # and `c` is the number of classes for the current category.
        self.Y = [[np.array([y[i]]) for y in Y]
                  for i in range(len(X))]
        if sample_weights is not None:
            self.sample_weights = [[np.array([sample_weight[i]]) for sample_weight in sample_weights]
                                   for i in range(len(X))]
        else:
            self.sample_weights = None
        self.indices = np.arange(len(X))
        self.shuffle = shuffle
        self.shuffle_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        if self.sample_weights is not None:
            return self.X[i], self.Y[i], self.sample_weights[i]
        return self.X[i], self.Y[i]

    def on_epoch_end(self):
        self.shuffle_indices()

    def shuffle_indices(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def main():
    start_time = int(time.time())
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
    else:
        stamp = start_time
    print('Time stamp: {:d}'.format(stamp))
    if note is not None:
        print('Note: {}'.format(note))

    # Load data.
    print('Retrieving texts...')
    subset_ratio = 1.0
    subset_seed = 1
    min_len = 256
    max_len = 4096
    min_tokens = 6
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'tokens'},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens)
    text_paragraph_tokens, _ = zip(*inputs['tokens'])
    print('Retrieved {:d} texts.'.format(len(text_paragraph_tokens)))

    # Tokenize.
    print('Tokenizing...')
    max_words = 8192  # The maximum size of the vocabulary.
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_sentences = []
    for paragraph_tokens in text_paragraph_tokens:
        for tokens in paragraph_tokens:
            all_sentences.append(split.join(tokens))
    tokenizer.fit_on_texts(all_sentences)
    print('Done.')

    # Convert to sequences.
    print('Converting texts to sequences...')
    n_tokens = 128  # The maximum number of tokens to process in each paragraph.
    padding = 'pre'
    truncating = 'pre'
    X = [np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in paragraph_tokens]),
                                maxlen=n_tokens,
                                padding=padding,
                                truncating=truncating))
         for paragraph_tokens in text_paragraph_tokens]
    print('Done.')

    # Load embedding.
    print('Loading embedding matrix...')
    embedding_path = folders.EMBEDDING_GLOVE_100_PATH
    embedding_matrix = load_embeddings.get_embedding(tokenizer, embedding_path, max_words)
    print('Done.')

    # Create model.
    print('Creating model...')
    n_classes = [len(levels) for levels in category_levels]
    embedding_trainable = False
    word_rnn = GRU
    if tf.test.is_gpu_available(cuda_only=True):
        if word_rnn == GRU:
            word_rnn = CuDNNGRU
        elif word_rnn == LSTM:
            word_rnn = CuDNNLSTM
    word_rnn_units = 128
    word_rnn_l2 = .01
    word_dense_units = 64
    word_dense_activation = 'linear'
    word_dense_l2 = .01
    book_dense_units = 512
    book_dense_activation = 'relu'
    book_dense_l2 = .01
    book_dropout = .5
    is_ordinal = True
    model = create_model(
        n_classes,
        categories,
        n_tokens,
        embedding_matrix,
        embedding_trainable=embedding_trainable,
        word_rnn=word_rnn,
        word_rnn_units=word_rnn_units,
        word_rnn_l2=word_rnn_l2,
        word_dense_units=word_dense_units,
        word_dense_activation=word_dense_activation,
        word_dense_l2=word_dense_l2,
        book_dense_units=book_dense_units,
        book_dense_activation=book_dense_activation,
        book_dense_l2=book_dense_l2,
        book_dropout=book_dropout,
        is_ordinal=is_ordinal)
    print('Done.')

    # Compile.
    lr = .001
    optimizer = Adam(lr=lr)
    if is_ordinal:
        loss = 'binary_crossentropy'
        metric = 'binary_accuracy'
    else:
        loss = 'categorical_crossentropy'
        metric = 'categorical_accuracy'
    model.compile(optimizer, loss=loss, metrics=[metric])

    # Give each instance (sample) a weight that is inversely proportional to the frequency of its class.
    sample_weights = np.zeros(Y.shape, dtype=np.float32)  # (C, n)
    for category_i, y in enumerate(Y):
        bincount = np.bincount(y, minlength=n_classes[category_i])
        for j, value in enumerate(y):
            sample_weights[category_i, j] = 1/(bincount[value] + 1)

    # Split data set.
    test_size = .25  # b
    test_random_state = 1
    val_size = .1  # v
    val_random_state = 1
    Y_T = Y.transpose()  # (n, C)
    sample_weights_T = sample_weights.transpose()  # (n, C)
    X_train, X_test, Y_train_T, Y_test_T, sample_weights_train_T, sample_weights_test_T = \
        train_test_split(X, Y_T, sample_weights_T, test_size=test_size, random_state=test_random_state)
    X_train, X_val, Y_train_T, Y_val_T, sample_weights_train_T, sample_weights_val_T = \
        train_test_split(X_train, Y_train_T, sample_weights_train_T, test_size=val_size, random_state=val_random_state)
    Y_train = Y_train_T.transpose()  # (C, n * (1 - b) * (1 - v))
    Y_val = Y_val_T.transpose()  # (C, n * (1 - b) * v)
    Y_test = Y_test_T.transpose()  # (C, n * b)
    sample_weights_train = sample_weights_train_T.transpose()  # (C, n * (1 - b) * (1 - v))
    sample_weights_val = sample_weights_val_T.transpose()  # (C, n * (1 - b) * v)
    sample_weights_test = sample_weights_test_T.transpose()  # (C, n * b)
    use_sample_weights = False
    if not use_sample_weights:
        sample_weights_train = None
        sample_weights_val = None
        sample_weights_test = None

    # Train.
    use_class_weights = True
    if is_ordinal:
        Y_train = [ordinal.to_multi_hot_ordinal(Y_train[i], n_classes=n) for i, n in enumerate(n_classes)]
        Y_val = [ordinal.to_multi_hot_ordinal(Y_val[i], n_classes=n) for i, n in enumerate(n_classes)]
        category_class_weights = []  # [[dict]]
        for category_i, y_train in enumerate(Y_train):
            class_weights = []
            for i in range(y_train.shape[1]):
                ones_count = sum(y_train[:, i] == 1)
                class_weight = {0: 1 / (len(y_train) - ones_count + 1), 1: 1 / (ones_count + 1)}
                class_weights.append(class_weight)
            category_class_weights.append(class_weights)
    else:
        category_class_weights = []  # [dict]
        for category_i, y_train in enumerate(Y_train):
            bincount = np.bincount(y_train, minlength=n_classes[category_i])
            class_weight = {i: 1 / (count + 1) for i, count in enumerate(bincount)}
            category_class_weights.append(class_weight)
        Y_train = [to_categorical(Y_train[i], num_classes=n) for i, n in enumerate(n_classes)]
        Y_val = [to_categorical(Y_val[i], num_classes=n) for i, n in enumerate(n_classes)]
    train_generator = SingleInstanceBatchGenerator(X_train, Y_train, sample_weights=sample_weights_train, shuffle=True)
    val_generator = SingleInstanceBatchGenerator(X_val, Y_val, sample_weights=sample_weights_val, shuffle=False)
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None,
                                  epochs=epochs,
                                  class_weight=category_class_weights if use_class_weights else None,
                                  validation_data=val_generator)

    # Save the history to visualize loss over time.
    print('Saving training history...')
    if not os.path.exists(folders.HISTORY_PATH):
        os.mkdir(folders.HISTORY_PATH)
    history_path = os.path.join(folders.HISTORY_PATH, 'book_net')
    if not os.path.exists(history_path):
        os.mkdir(history_path)
    with open(os.path.join(history_path, '{:d}.txt'.format(stamp)), 'w') as fd:
        for key in history.history.keys():
            values = history.history.get(key)
            fd.write('{} {}\n'.format(key, ' '.join(str(value) for value in values)))
    print('Done.')

    # Predict test instances.
    print('Predicting test instances...')
    test_generator = SingleInstanceBatchGenerator(X_test, Y_test, sample_weights=sample_weights_test, shuffle=False)
    Y_preds = model.predict_generator(test_generator)
    if is_ordinal:
        Y_preds = [ordinal.from_multi_hot_ordinal(y, threshold=.5) for y in Y_preds]
    else:
        Y_preds = [np.argmax(y, axis=1) for y in Y_preds]
    print('Done.')

    # Calculate elapsed time.
    end_time = int(time.time())
    elapsed_s = end_time - start_time
    elapsed_m, elapsed_s = elapsed_s // 60, elapsed_s % 60
    elapsed_h, elapsed_m = elapsed_m // 60, elapsed_m % 60

    # Write results.
    print('Writing results....')
    if not os.path.exists(folders.LOGS_PATH):
        os.mkdir(folders.LOGS_PATH)
    logs_path = os.path.join(folders.LOGS_PATH, 'book_net')
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    with open(os.path.join(logs_path, '{:d}.txt'.format(stamp)), 'w') as fd:
        if note is not None:
            fd.write('Note: {}\n\n'.format(note))
        fd.write('PARAMETERS\n\n')
        fd.write('steps_per_epoch={:d}\n'.format(steps_per_epoch))
        fd.write('epochs={:d}\n'.format(epochs))
        fd.write('\nHYPERPARAMETERS\n')
        fd.write('\nText\n')
        fd.write('subset_ratio={:.3f}\n'.format(subset_ratio))
        fd.write('subset_seed={:d}\n'.format(subset_seed))
        fd.write('min_len={:d}\n'.format(min_len))
        fd.write('max_len={:d}\n'.format(max_len))
        fd.write('min_tokens={:d}\n'.format(min_tokens))
        fd.write('\nTokenization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('n_tokens={:d}\n'.format(n_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        fd.write('word_rnn={}\n'.format(word_rnn.__name__))
        fd.write('word_rnn_units={:d}\n'.format(word_rnn_units))
        fd.write('word_rnn_l2={:.3f}\n'.format(word_rnn_l2))
        fd.write('word_dense_units={:d}\n'.format(word_dense_units))
        fd.write('word_dense_activation=\'{}\'\n'.format(word_dense_activation))
        fd.write('word_dense_l2={:.3f}\n'.format(word_dense_l2))
        fd.write('book_dense_units={:d}\n'.format(book_dense_units))
        fd.write('book_dense_activation=\'{}\'\n'.format(book_dense_activation))
        fd.write('book_dense_l2={:.3f}\n'.format(book_dense_l2))
        fd.write('book_dropout={:.1f}\n'.format(book_dropout))
        fd.write('is_ordinal={}\n'.format(is_ordinal))
        model.summary(print_fn=lambda x: fd.write('{}\n'.format(x)))
        fd.write('\nTraining\n')
        fd.write('optimizer={}\n'.format(optimizer.__class__.__name__))
        fd.write('lr={:.4f}\n'.format(lr))
        fd.write('loss=\'{}\'\n'.format(loss))
        fd.write('metric=\'{}\'\n'.format(metric))
        fd.write('test_size={:.2f}\n'.format(test_size))
        fd.write('test_random_state={:d}\n'.format(test_random_state))
        fd.write('val_size={:.2f}\n'.format(val_size))
        fd.write('val_random_state={:d}\n'.format(val_random_state))
        fd.write('use_sample_weights={}\n'.format(use_sample_weights))
        fd.write('use_class_weights={}\n'.format(use_class_weights))
        fd.write('\nRESULTS\n\n')
        fd.write('data size: {:d}\n'.format(len(text_paragraph_tokens)))
        fd.write('train size: {:d}\n'.format(len(X_train)))
        fd.write('validation size: {:d}\n'.format(len(X_val)))
        fd.write('test size: {:d}\n'.format(len(X_test)))
        fd.write('time elapsed: {:d}h {:d}m {:d}s\n'.format(elapsed_h, elapsed_m, elapsed_s))
        for category_i, category in enumerate(categories):
            fd.write('\n`{}`\n'.format(category))
            confusion, metrics = evaluation.get_metrics(Y_test[category_i], Y_preds[category_i])
            fd.write(np.array2string(confusion))
            fd.write('\n')
            for name, value in metrics:
                fd.write('{}={:.4f}\n'.format(name, value))
    print('Done.')


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        raise ValueError('Usage: <steps_per_epoch> <epochs> [note]')
    steps_per_epoch = int(sys.argv[1])
    epochs = int(sys.argv[2])
    note = None
    if len(sys.argv) > 3:
        note = sys.argv[3]
    main()
