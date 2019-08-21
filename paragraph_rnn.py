from __future__ import absolute_import, division, print_function, with_statement
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GlobalMaxPool1D, GRU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

from classification import evaluation, ordinal
import folders
from sites.bookcave import bookcave
from text import load_embeddings


def get_input_array(sequence, n_tokens):
    x = np.zeros((n_tokens,), dtype=np.int32)
    if len(sequence) > n_tokens:
        # Truncate center.
        x[:n_tokens//2] = sequence[:n_tokens//2]
        x[-n_tokens//2:] = sequence[-n_tokens//2:]
    else:
        # Pad beginning ('pre').
        x[-len(sequence):] = sequence
    return x


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

    weights_fname = 'model_{:d}t_{:d}v_{:d}d_{:d}h_{:d}f{}.h5'.format(n_tokens,
                                                                      embedding_matrix.shape[0],
                                                                      embedding_matrix.shape[1],
                                                                      hidden_size,
                                                                      dense_size,
                                                                      '' if embedding_trainable else '_static')

    return model, weights_fname


def predict_book_labels(model, X, locations, Y, get_label):
    category_Q_pred_ordinal = model.predict(X)  # (C, m, k_c - 1)
    category_Q_pred = [ordinal.from_multi_hot_ordinal(q_pred_ordinal)
                       for q_pred_ordinal in category_Q_pred_ordinal]  # (C, m)
    # Iterate over locations; calculate book label on text change or end.
    Y_pred = np.zeros(Y.shape, dtype=np.int32)  # (C, m)
    text_i = locations[0][0]
    category_text_pred = [[] for _ in range(len(Y_pred))]
    for location_i, location in enumerate(locations):
        if location[0] != text_i:
            # Calculate book labels.
            labels = [get_label(text_pred) for text_pred in category_text_pred]
            Y_pred[:, text_i] = labels
            # Reset text counter and paragraph predictions.
            text_i = location[0]
            category_text_pred = [[] for _ in range(len(Y_pred))]
        # Append paragraph predictions.
        for category_i, q_pred in enumerate(category_Q_pred):
            category_text_pred[category_i].append(q_pred[location_i])
    # Calculate book labels for final book.
    labels = [get_label(text_pred) for text_pred in category_text_pred]
    Y_pred[:, text_i] = labels
    return Y_pred


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

    # Flatten tokens.
    if verbose:
        print('\nFlattening tokens...')
    all_locations = []
    all_tokens = []
    for text_i, paragraph_tokens in enumerate(text_paragraph_tokens):
        for paragraph_i, tokens in enumerate(paragraph_tokens):
            all_locations.append((text_i, paragraph_i))
            all_tokens.append(tokens)
    if verbose:
        print('Paragraphs: {:d}'.format(len(all_tokens)))

    # Tokenize.
    if verbose:
        print('\nTokenizing...')
    max_words = 8192
    tokenizer = Tokenizer(num_words=max_words, oov_token='__UNKNOWN__')
    tokenizer.fit_on_texts(all_tokens)
    if verbose:
        print('Done.')

    # Load embedding matrix.
    if verbose:
        print('\nLoading embedding...')
    embedding_matrix = load_embeddings.get_embedding(tokenizer, folders.EMBEDDING_GLOVE_100_PATH, max_words)
    if verbose:
        print('Done.')

    # Load paragraph labels.
    if verbose:
        print('\nLoading paragraph labels...')
    tokens_min_len = 3
    train_locations = []
    train_tokens = []
    train_paragraph_labels = []
    for text_i, paragraph_tokens in enumerate(text_paragraph_tokens):
        book_id = book_ids[text_i]
        asin = books_df[books_df['id'] == book_id].iloc[0]['asin']
        category_labels = [bookcave.get_labels(asin, category) for category in categories]
        if any(labels is None for labels in category_labels):
            continue
        for paragraph_i, tokens in enumerate(paragraph_tokens):
            paragraph_labels = [labels[paragraph_i] for labels in category_labels]
            if any(label == -1 for label in paragraph_labels):
                continue
            if len(tokens) < tokens_min_len:
                continue
            train_locations.append((text_i, paragraph_i))
            train_tokens.append(tokens)
            train_paragraph_labels.append(paragraph_labels)
    test_text_indices = list({text_i for text_i, _ in train_locations})
    if verbose:
        print('Finished loading labels for {:d} books.'.format(len(test_text_indices)))

    # Split data.
    if verbose:
        print('\nSplitting data into training and test sets...')
    n_tokens = 160  # t
    test_size = .25  # b
    random_state = 1
    train_sequences = tokenizer.texts_to_sequences(train_tokens)
    P = np.array([get_input_array(sequence, n_tokens) for sequence in train_sequences])  # (n, t)
    Q = np.array(train_paragraph_labels)  # (n, C)
    P_train, P_test, Q_train, Q_test = train_test_split(P, Q, test_size=test_size, random_state=random_state)
    if verbose:
        print('Training instances: {:d}'.format(len(P_train)))
        print('Test instances: {:d}'.format(len(P_test)))

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

    # Weight classes inversely proportional to their frequency.
    n_classes = [len(levels) for levels in category_levels]
    class_weights = []
    for category_i in range(Q_train.shape[1]):
        q_train = Q_train[:, category_i]
        bincount = np.bincount(q_train, minlength=n_classes[category_i])
        class_weight = {i: 1 / (count + 1) for i, count in enumerate(bincount)}
        class_weights.append(class_weight)

    # Train on paragraphs.
    batch_size = 32
    optimizer = Adam()
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    category_Q_train_ordinal = [ordinal.to_multi_hot_ordinal(Q_train[:, category_i], n_classes=n_classes[category_i])
                                for category_i in range(Q_train.shape[1])]  # (C, (1 - b)*n, k_c - 1)
    _ = model.fit(P_train,
                  category_Q_train_ordinal,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  class_weight=class_weights)

    # Evaluate paragraphs.
    category_Q_pred_ordinal = model.predict(P_test)  # (C, b*n, k_c - 1)
    for category_i, category in enumerate(categories):
        print('\nParagraph metrics for category `{}`:'.format(category))
        q_test = Q_test[:, category_i]
        q_pred_ordinal = category_Q_pred_ordinal[category_i]  # (b*n, k_c - 1)
        q_pred = ordinal.from_multi_hot_ordinal(q_pred_ordinal)  # (b*n,)
        evaluation.print_metrics(q_test, q_pred)

    # Evaluate books.
    def get_label_from_paragraph_labels(q_pred_):
        return max(q_pred_)

    # Evaluate only books from which the training set of paragraphs came.
    if verbose:
        print('\nEvaluating training set...')
    test_locations = []
    test_tokens = []
    Y_test = Y[:, test_text_indices]
    for location_i, text_i in enumerate(test_text_indices):
        for paragraph_i, tokens in enumerate(text_paragraph_tokens[text_i]):
            test_locations.append((location_i, paragraph_i))
            test_tokens.append(tokens)
    test_sequences = tokenizer.texts_to_sequences(test_tokens)
    X_test = np.array([get_input_array(sequence, n_tokens) for sequence in test_sequences])
    Y_pred_test = predict_book_labels(model, X_test, test_locations, Y_test, get_label_from_paragraph_labels)
    for category_i, category in enumerate(categories):
        print('\nTraining set of books for category `{}`:'.format(category))
        y_test, y_pred_test = Y_test[category_i], Y_pred_test[category_i]
        evaluation.print_metrics(y_test, y_pred_test)

    # Evaluate all books.
    if verbose:
        print('\nEvaluating all books...')
    all_sequences = tokenizer.texts_to_sequences(all_tokens)
    X_all = np.array([get_input_array(sequence, n_tokens) for sequence in all_sequences])
    Y_pred_all = predict_book_labels(model, X_all, all_locations, Y, get_label_from_paragraph_labels)
    for category_i, category in enumerate(categories):
        print('\nAll books for category `{}`:'.format(category))
        y, y_pred_all = Y[category_i], Y_pred_all[category_i]
        evaluation.print_metrics(y, y_pred_all)


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise ValueError('Usage: <epochs> [verbose]')
    epochs = int(sys.argv[1])
    if len(sys.argv) > 2:
        verbose = int(sys.argv[2])
    else:
        verbose = 0
    main()
