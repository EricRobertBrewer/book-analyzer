from argparse import ArgumentParser, RawTextHelpFormatter
import os
import time

import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Input, Flatten, LeakyReLU, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from python import folders
from python.sites.bookcave import bookcave
from python.util import ordinal, shared_parameters
from python.util.net.batch_generators import SimpleBatchGenerator, TransformBalancedBatchGenerator


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
    parser.add_argument('--label_mode',
                        default=shared_parameters.LABEL_MODE_ORDINAL,
                        choices=[shared_parameters.LABEL_MODE_ORDINAL,
                                 shared_parameters.LABEL_MODE_CATEGORICAL,
                                 shared_parameters.LABEL_MODE_REGRESSION],
                        help='The way that labels will be interpreted. '
                             'Default is `{}`.'.format(shared_parameters.LABEL_MODE_ORDINAL))
    args = parser.parse_args()

    classifier_name = 'cover_net'
    start_time = int(time.time())
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
    else:
        stamp = start_time
    base_fname = '{:d}_{:d}'.format(stamp, args.category_index)

    images_size = (256, 256)

    # Here, `Y` has shape (n, m) where `n` is the number of books and `m` is the number of maturity categories.
    inputs, Y, categories, levels = \
        bookcave.get_data({'images'},
                          subset_ratio=1/4,  # shared_parameters.
                          subset_seed=1,
                          image_size=images_size)
    image_paths = inputs['images']

    # Reduce the labels to the specified category.
    y = Y[args.category_index]
    category = categories[args.category_index]
    levels = levels[args.category_index]
    k = len(levels)

    # Split data set.
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    val_size = shared_parameters.EVAL_VAL_SIZE
    val_random_state = shared_parameters.EVAL_VAL_RANDOM_STATE
    image_paths_train, image_paths_test, y_train, y_test = \
        train_test_split(image_paths, y, test_size=test_size, random_state=test_random_state)
    image_paths_train, image_paths_val, y_train, y_val = \
        train_test_split(image_paths_train, y_train, test_size=val_size, random_state=val_random_state)
    X_val = image_paths_to_tensors(image_paths_val)
    y_val_transform = shared_parameters.transform_labels(y_val, k, args.label_mode)
    X_test = image_paths_to_tensors(image_paths_test)

    # Train.
    optimizer = Adam(lr=2**-10)
    model = get_model(images_size, k, optimizer)
    plateau_monitor = 'val_loss'
    plateau_factor = .5
    early_stopping_monitor = 'val_loss'
    early_stopping_min_delta = 2 ** -10
    plateau_patience = 10
    early_stopping_patience = 20
    callbacks = [
        ReduceLROnPlateau(monitor=plateau_monitor,
                          factor=plateau_factor,
                          patience=plateau_patience),
        EarlyStopping(monitor=early_stopping_monitor,
                      min_delta=early_stopping_min_delta,
                      patience=early_stopping_patience)
    ]
    train_generator = TransformBalancedBatchGenerator(image_paths_train,
                                                      y_train,
                                                      transform_X=image_paths_to_tensors,
                                                      transform_y=transform_y,
                                                      batch_size=32,
                                                      k=k,
                                                      label_mode=shared_parameters.LABEL_MODE_ORDINAL)
    val_generator = SimpleBatchGenerator(X_val, y_val_transform, batch_size=32)
    history = model.fit_generator(train_generator,
                                  callbacks=callbacks,
                                  epochs=1000,
                                  validation_data=val_generator)

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


def image_paths_to_tensors(image_paths, **kwargs):
    images = [load_img(book_images[0]) for book_images in image_paths]
    return np.array([img_to_array(image) for image in images])


def transform_y(y, **kwargs):
    k = kwargs['k']
    label_mode = kwargs['label_mode']
    return shared_parameters.transform_labels(y, k, label_mode)


if __name__ == '__main__':
    main()
