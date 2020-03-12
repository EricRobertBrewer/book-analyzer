from argparse import ArgumentParser, RawTextHelpFormatter
import os
import time

import numpy as np
from keras.callbacks import Callback
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Input, Flatten, LeakyReLU, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from python import folders
from python.sites.bookcave import bookcave
from python.util import ordinal, shared_parameters
from python.util.net.batch_generators import TransformBalancedBatchGenerator


def get_model(images_size, k, optimizer):
    inp = Input((*images_size, 3))
    x = inp

    for _ in range(5):
        x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.3))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation=LeakyReLU(alpha=0.3))(x)
    x = Dropout(0.5)(x)

    outputs = Dense(k - 1, activation='sigmoid')(x)
    model = Model(inp, outputs)
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    parser = ArgumentParser(
        description='Classify the maturity level of a book by its cover.',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('category_index',
                        type=int,
                        help='The category index.\n  {}'.format(
                            '\n  '.join(['{:d} {}'.format(j, bookcave.CATEGORY_NAMES[category])
                                         for j, category in enumerate(bookcave.CATEGORIES)]
                        )))
    args = parser.parse_args()

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

    # Reduce the labels to the specified category.
    y = Y[args.category_index]
    category = categories[args.category_index]
    levels = levels[args.category_index]

    # Transform file paths into images, then images into numerical tensors.
    image_paths = inputs['images']

    # Train.
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    val_size = shared_parameters.EVAL_VAL_SIZE
    val_random_state = shared_parameters.EVAL_VAL_RANDOM_STATE
    image_paths_train, image_paths_test, y_train, y_test = \
        train_test_split(image_paths, y, test_size=test_size, random_state=test_random_state)
    image_paths_train, image_paths_val, y_train, y_val = \
        train_test_split(image_paths_train, y_train, test_size=val_size, random_state=val_random_state)

    # Turn the discrete labels into an ordinal one-hot encoding.
    # See `http://orca.st.usm.edu/~zwang/files/rank.pdf`.
    k = len(levels)

    optimizer = Adam(lr=2**-10)
    model = get_model(images_size, k, optimizer)
    train_generator = OrdinalBalancedBatchGenerator(image_paths_train, y_train, image_paths_to_tensors, k, batch_size=32)
    val_generator = None  # OrdinalBalancedBatchGenerator(X_val, y_val, k, batch_size=32)
    history = model.fit_generator(train_generator,
                                  epochs=8,
                                  validation_data=val_generator)

    X_test = image_paths_to_tensors(image_paths_test)
    y_pred_ordinal = model.predict(X_test)

    # Convert the ordinal one-hot encoding back to discrete labels.
    y_pred = ordinal.from_multi_hot_ordinal(y_pred_ordinal, threshold=0.5)

    print('`{}`:'.format(category))
    print('Accuracy: {:.3%}'.format(accuracy_score(y_test, y_pred)))
    confusion = confusion_matrix(y_test, y_pred)
    print(confusion)

    history_path = folders.ensure(os.path.join(folders.HISTORY_PATH, classifier_name))
    with open(os.path.join(history_path, '{}.txt'.format(base_fname)), 'w') as fd:
        for key in history.history.keys():
            values = history.history.get(key)
            fd.write('{} {}\n'.format(key, ' '.join(str(value) for value in values)))


def image_paths_to_tensors(image_paths):
    images = [load_img(book_images[0]) for book_images in image_paths]
    return np.array([img_to_array(image) for image in images])


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
