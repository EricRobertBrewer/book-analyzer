import numpy as np
from keras.models import Input, Model
from keras.layers import Bidirectional, Dense, Dropout, Embedding, GlobalMaxPool1D, GRU
from keras.optimizers import Adam
from keras import regularizers
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from sites.bookcave import bookcave
from classification import ordinal
from text import preprocessing


def get_char_to_index(processed_title_chars):
    # Map each character to a unique index.
    chars = set()
    for title_chars in processed_title_chars:
        chars.update(title_chars)
    chars_sorted = sorted(list(chars))
    # Reserve index `0` for any unseen character.
    return {char: i + 1 for i, char in enumerate(chars_sorted)}


def get_model(optimizer, input_shape, input_dim, label_count):
    inp = Input(input_shape)

    # Transform character indices (integers) to their one-hot encoding equivalent.
    embedding_matrix = np.eye(input_dim, input_dim, dtype=np.float32)
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inp)

    # Bi-directional RNN.
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

    # Reduce axes (of the matrix) to a vector.
    x = GlobalMaxPool1D()(x)

    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2())(x)
    x = Dropout(0.5)(x)

    x = Dense(label_count - 1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(
        optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy', 'categorical_accuracy'])
    return model


def main():
    _, Y, categories, levels, \
        book_ids, books_df, _, _, _ = \
        bookcave.get_data(
            {'text', 'images'},
            text_input='filename',
            only_categories={1, 3, 5, 6},
            return_meta=True)

    # Collect and pre-process book titles.
    titles = books_df['title'].values
    processed_titles = [preprocessing.normalize(title).lower() for title in titles]

    # Convert strings to arrays of characters.
    processed_title_chars = [list(title) for title in processed_titles]

    # Convert arrays of characters to arrays of one-hot-encoded integers.
    char_to_index = get_char_to_index(processed_title_chars)

    # The cardinality of the input, i.e., how many unique characters are there (plus the reserved unknown character).
    input_dim = len(char_to_index) + 1

    # Choose the maximum number of characters to consider in titles.
    seq_len = 100

    # Populate the input.
    X = np.zeros((len(processed_title_chars), seq_len), dtype=np.int32)
    for i, title_chars in enumerate(processed_title_chars):
        # Get the list of indices for this title, up to the maximum sequence length.
        indices = [char_to_index[char] for char in title_chars[:min(len(title_chars), seq_len)]]
        # Leave 0-padding at the front, if any.
        X[i, -len(indices):] = indices

    # Train.
    for category_index, category in enumerate(categories):
        print('\nStarting `{}`...'.format(category))

        label_count = len(levels[category_index])
        y = Y[:, category_index]
        bincount = np.bincount(y, minlength=label_count)
        class_weight = {i: 1 / count for i, count in enumerate(bincount)}

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_ordinal = ordinal.to_multi_hot_ordinal(y_train, label_count)

            optimizer = Adam()
            input_shape = X[0].shape
            model = get_model(optimizer, input_shape, input_dim, label_count)
            history = model.fit(X_train, y_train_ordinal, epochs=1, batch_size=32, class_weight=class_weight)

            y_pred_ordinal = model.predict(X_test)
            y_pred = ordinal.from_multi_hot_ordinal(y_pred_ordinal)

            print('Accuracy: {:.3%}'.format(accuracy_score(y_test, y_pred)))
            confusion = confusion_matrix(y_test, y_pred)
            print(confusion)


if __name__ == '__main__':
    main()
