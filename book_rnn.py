from __future__ import absolute_import, division, print_function, with_statement
from keras.layers import Bidirectional, Dense, Dropout, Embedding, GlobalMaxPool1D, GRU, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import sys

from classification import evaluation, ordinal
import folders
import monkey
from sites.bookcave import bookcave
from text import load_embeddings


def create_model(n_classes, n_tokens, embedding_matrix, hidden_size, dense_size, embedding_trainable=True):
    inp = Input(shape=(n_tokens,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=embedding_trainable)(inp)
    x = Bidirectional(GRU(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Each model outputs ordinal labels; thus, one less than the number of classes.
    outs = [Dense(n - 1, activation='sigmoid')(x) for n in n_classes]
    model = Model(inp, outs)

    weights_fname = 'book_rnn_{:d}t_{:d}v_{:d}d_{:d}h_{:d}f{}.h5'.format(n_tokens,
                                                                         embedding_matrix.shape[0],
                                                                         embedding_matrix.shape[1],
                                                                         hidden_size,
                                                                         dense_size,
                                                                         '' if embedding_trainable else '_static')

    return model, weights_fname


def main():
    # Load data.
    if verbose:
        print('\nRetrieving texts...')
    min_len, max_len = 250, 7500
    inputs, Y, categories, category_levels, book_ids, books_df, _, _, categories_df =\
        bookcave.get_data({'tokens'},
                          min_len=min_len,
                          max_len=max_len,
                          return_meta=True)
    text_paragraph_tokens, _ = zip(*inputs['tokens'])
    if verbose:
        print('Retrieved {:d} texts.'.format(len(text_paragraph_tokens)))

    # Represent each text as a long sequence of tokens.
    text_tokens = []
    for paragraph_tokens in text_paragraph_tokens:
        all_tokens = []
        for tokens in paragraph_tokens:
            all_tokens.extend(tokens)
        text_tokens.append(all_tokens)

    # Tokenize.
    if verbose:
        print('\nTokenizing...')
    max_words = 8192
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words)
    monkey.patch_tokenizer()
    tokenizer.fit_on_texts([split.join(tokens) for tokens in text_tokens])
    if verbose:
        print('Done.')

    # Load embedding matrix.
    if verbose:
        print('\nLoading embedding...')
    embedding_matrix = load_embeddings.get_embedding(tokenizer, folders.EMBEDDING_GLOVE_100_PATH, max_words)
    if verbose:
        print('Done.')

    # Convert to sequences.
    if verbose:
        print('\nConverting texts to sequences...')
    n_tokens = 16384  # t
    X = np.zeros((len(text_tokens), n_tokens), dtype=np.float32)
    for text_i, tokens in enumerate(text_tokens):
        if len(tokens) > n_tokens:
            # Truncate two thirds of the remainder at the beginning and one third at the end.
            start = int(2/3*(len(tokens) - n_tokens))
            usable_tokens = tokens[start:start + n_tokens]
        else:
            usable_tokens = tokens
        sequences = tokenizer.texts_to_sequences([split.join(usable_tokens)])[0]
        X[text_i] = pad_sequences(sequences, maxlen=n_tokens, padding='pre', truncating='pre')
    if verbose:
        print('Done.')

    # Create a new model.
    if verbose:
        print('\nCreating model...')
    hidden_size = 64
    dense_size = 32
    embedding_trainable = True
    n_classes = [len(levels) for levels in category_levels]
    model, weights_fname = create_model(n_classes,
                                        n_tokens,
                                        embedding_matrix,
                                        hidden_size,
                                        dense_size,
                                        embedding_trainable=embedding_trainable)
    if verbose:
        print(model.summary())

    # Split data.
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
