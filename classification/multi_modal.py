import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Input, Flatten, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from keras import regularizers
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import bookcave
import ordinal


def get_model(images_size, num_classes, optimizer):
    inp = Input((*images_size, 3))

    x = Conv2D(32, (3, 3), padding='same', input_shape=(3, *images_size), activation='relu')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2())(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes - 1, activation='sigmoid')(x)
    model = Model(inp, outputs)
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'categorical_accuracy'])
    return model


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
    for category_index, category in enumerate(categories):
        print('Starting `{}`...'.format(category))

        y = Y[:, category_index]
        num_classes = len(levels[category_index])

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Weight classes inversely proportional to their frequency.
            bincount = np.bincount(y_train, minlength=num_classes)
            class_weight = {i: 1 / count ** 2 for i, count in enumerate(bincount)}

            # Turn the discrete labels into an ordinal one-hot encoding.
            # See `http://orca.st.usm.edu/~zwang/files/rank.pdf`.
            y_train_ordinal = ordinal.to_multi_hot_ordinal(y_train, num_classes=num_classes)

            optimizer = Adam()
            model = get_model(images_size, num_classes, optimizer)
            history = model.fit(X_train, y_train_ordinal, epochs=1, batch_size=32, class_weight=class_weight)
            y_pred_ordinal = model.predict(X_test)

            # Convert the ordinal one-hot encoding back to discrete labels.
            y_pred = ordinal.from_multi_hot_ordinal(y_pred_ordinal)

            print('Accuracy: {:.3%}'.format(accuracy_score(y_test, y_pred)))
            confusion = confusion_matrix(y_test, y_pred)
            print(confusion)


if __name__ == '__main__':
    main()
