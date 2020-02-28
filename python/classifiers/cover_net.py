import os
import time

import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, LeakyReLU, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from python.util import ordinal, shared_parameters

from python import folders
from python.sites.bookcave import bookcave


def get_model(images_size, category_k, optimizer):
    inp = Input((*images_size, 3))
    x = inp

    for _ in range(5):
        x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.3))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation=LeakyReLU(alpha=0.3))(x)
    x = Dropout(0.5)(x)

    outputs = [Dense(k - 1, activation='sigmoid')(x) for k in category_k]
    model = Model(inp, outputs)
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    classifier_name = 'cover_net'
    start_time = int(time.time())
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
    else:
        stamp = start_time
    base_fname = format(stamp, 'd')

    images_size = (256, 256)

    # Here, `Y` has shape (n, m) where `n` is the number of books and `m` is the number of maturity categories.
    inputs, Y, categories, levels = \
        bookcave.get_data({'images'},
                          subset_ratio=1/4,  # shared_parameters.
                          subset_seed=1,
                          image_size=images_size)

    category_index = 5
    if category_index != -1:
        Y = np.array([Y[category_index]])
        categories = [categories[category_index]]
        levels = [levels[category_index]]

    # Transform file paths into images, then images into numerical tensors.
    images = [load_img(book_images[0]) for book_images in inputs['images']]
    X = np.array([img_to_array(image) for image in images])

    # Train.
    Y_T = Y.transpose()
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    val_size = shared_parameters.EVAL_VAL_SIZE
    val_random_state = shared_parameters.EVAL_VAL_RANDOM_STATE
    X_train, X_test, Y_train_T, Y_test_T = \
        train_test_split(X, Y_T, test_size=test_size, random_state=test_random_state)
    X_train, X_val, Y_train_T, Y_val_T = \
        train_test_split(X_train, Y_train_T, test_size=val_size, random_state=val_random_state)
    Y_train = Y_train_T.transpose()  # (c, n * (1 - b) * (1 - v))
    Y_val = Y_val_T.transpose()
    Y_test = Y_test_T.transpose()  # (c, n * b)

    # Turn the discrete labels into an ordinal one-hot encoding.
    # See `http://orca.st.usm.edu/~zwang/files/rank.pdf`.
    label_mode = shared_parameters.LABEL_MODE_ORDINAL
    category_k = [len(category_levels) for category_levels in levels]
    Y_train = shared_parameters.transform_labels(Y_train, category_k, label_mode)
    Y_val = shared_parameters.transform_labels(Y_val, category_k, label_mode)

    optimizer = Adam(lr=2**-10)
    model = get_model(images_size, category_k, optimizer)
    weight = shared_parameters.get_category_class_weights(Y_train,
                                                          label_mode=label_mode)
    history = model.fit(X_train,
                        Y_train,
                        epochs=8,
                        batch_size=64,
                        steps_per_epoch=None,
                        callbacks=None,  # [PrintCallback(model, X_test)],
                        validation_data=(X_val, Y_val),
                        class_weight=weight)
    Y_pred_transpose_ordinal = [model.predict(X_test)]

    # Convert the ordinal one-hot encoding back to discrete labels.
    Y_pred = np.array([ordinal.from_multi_hot_ordinal(y_pred_transpose_ordinal)
                       for y_pred_transpose_ordinal in Y_pred_transpose_ordinal])

    for category_index, category in enumerate(categories):
        print('`{}`:'.format(category))
        y_test = Y_test[category_index]
        y_pred = Y_pred[category_index]
        print('Accuracy: {:.3%}'.format(accuracy_score(y_test, y_pred)))
        confusion = confusion_matrix(y_test, y_pred)
        print(confusion)

    history_path = folders.ensure(os.path.join(folders.HISTORY_PATH, classifier_name))
    with open(os.path.join(history_path, '{}.txt'.format(base_fname)), 'w') as fd:
        for key in history.history.keys():
            values = history.history.get(key)
            fd.write('{} {}\n'.format(key, ' '.join(str(value) for value in values)))


class PrintCallback(Callback):

    def __init__(self, model, X_test):
        super(PrintCallback).__init__()
        self.model = model
        self.X_test = X_test

    def on_epoch_begin(self, epoch, logs=None):
        Y_pred_transpose_ordinal = [self.model.predict(self.X_test)]
        Y_pred = np.array([ordinal.from_multi_hot_ordinal(y_pred_transpose_ordinal) for y_pred_transpose_ordinal in
                           Y_pred_transpose_ordinal])
        print(Y_pred)


if __name__ == '__main__':
    main()
