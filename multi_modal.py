import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Input, Flatten, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import bookcave


def get_model(images_size, num_classes, optimizer):
    inp = Input((*images_size, 3))

    x = Conv2D(32, (3, 3), input_shape=(3, *images_size), activation='relu')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = [Dense(num_classes[i], activation='sigmoid')(x) for i in range(len(num_classes))]
    model = Model(inp, outputs)
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def to_one_hot_ordinal(y, num_classes=None):
    if num_classes is None:
        num_classes = max(y) - 1
    return np.array([[1 if value > i else 0 for i in range(num_classes)] for value in y])


def from_one_hot_ordinal(Y_ordinal, threshold=0.5):
    y = []
    for values in Y_ordinal:
        i = 0
        while i < len(values) and values[i] > threshold:
            i += 1
        y.append(i)
    return np.array(y)


def main():
    images_size = (512, 512)

    # Here, `Y` has shape (n, m) where `n` is the number of books and `m` is the number of maturity categories.
    inputs, Y, categories, levels = bookcave.get_data(
        {'text', 'images'},
        text_input='filename',
        images_source='cover',
        images_size=images_size,
        only_categories={1, 3, 5, 6})

    # Transform file paths into images, then images into numerical tensors.
    images = [load_img(book_images[0]) for book_images in inputs['images']]
    X = np.array([img_to_array(image) for image in images])

    # Train.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Transpose `Y` to make its new shape (m, n).
        # With this shape, the net can more cleanly learn multiple outputs at once.
        Y_train_transpose = Y_train.transpose(1, 0)

        # Turn the discrete labels into an ordinal one-hot encoding.
        # See `http://orca.st.usm.edu/~zwang/files/rank.pdf`.
        num_classes = [len(category_levels) - 1 for category_levels in levels]
        Y_train_transpose_ordinal = [to_one_hot_ordinal(y, num_classes=num_classes[i])
                                     for i, y in enumerate(Y_train_transpose)]

        optimizer = Adam()
        model = get_model(images_size, num_classes, optimizer)
        history = model.fit(X_train, Y_train_transpose_ordinal, epochs=1, batch_size=32)
        Y_pred_transpose_ordinal = model.predict(X_test)

        # Convert the ordinal one-hot encoding back to discrete labels.
        Y_pred_transpose = np.array([from_one_hot_ordinal(y_pred_transpose_ordinal)
                                     for y_pred_transpose_ordinal in Y_pred_transpose_ordinal])
        Y_pred = Y_pred_transpose.transpose(1, 0)

        for category_index, category in enumerate(categories):
            print('`{}`:'.format(category))
            y_test = Y_test[:, category_index]
            y_pred = Y_pred[:, category_index]
            print('Accuracy: {:.3%}'.format(accuracy_score(y_test, y_pred)))
            confusion = confusion_matrix(y_test, y_pred)
            print(confusion)


if __name__ == '__main__':
    main()
