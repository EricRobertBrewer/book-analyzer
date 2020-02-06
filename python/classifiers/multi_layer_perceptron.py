import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from python.util import evaluation, shared_parameters
from python.util import ordinal
from python import folders
from python.sites.bookcave import bookcave


def identity(v):
    return v


def create_model(
        max_words,
        dense_1_units, dense_1_activation, dense_1_l2,
        dense_2_units, dense_2_activation, dense_2_l2,
        dropout, output_k, output_names, label_mode):
    input_b = Input(shape=(max_words,), dtype='float32')
    if dense_1_l2 is not None:
        dense_1_l2 = regularizers.l2(dense_1_l2)
    x_b = Dense(dense_1_units,
                kernel_regularizer=dense_1_l2)(input_b)
    x_b = dense_1_activation(x_b)  # (c_b)
    if dense_2_l2 is not None:
        dense_2_l2 = regularizers.l2(dense_2_l2)
    x_b = Dense(dense_2_units,
                kernel_regularizer=dense_2_l2)(x_b)
    x_b = dense_2_activation(x_b)  # (c_b)
    x_b = Dropout(dropout)(x_b)  # (c_b)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        outputs = [Dense(k - 1, activation='sigmoid', name=output_names[i])(x_b) for i, k in enumerate(output_k)]
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        outputs = [Dense(k, activation='softmax', name=output_names[i])(x_b) for i, k in enumerate(output_k)]
    elif label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        outputs = [Dense(1, activation='linear', name=output_names[i])(x_b) for i in range(len(output_k))]
    else:
        raise ValueError('Unknown value for `label_mode`: {}'.format(label_mode))
    model = Model(input_b, outputs)
    return model


def main(argv):
    if len(argv) < 3 or len(argv) > 4:
        raise ValueError('Usage: <batch_size> <steps_per_epoch> <epochs> [note]')
    batch_size = int(argv[0])
    steps_per_epoch = int(argv[1])
    epochs = int(argv[2])
    note = None
    if len(argv) > 3:
        note = argv[3]

    script_name = os.path.basename(__file__)
    classifier_name = script_name[:script_name.index('.')]

    start_time = int(time.time())
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
    else:
        stamp = start_time
    print('Time stamp: {:d}'.format(stamp))
    if note is not None:
        print('Note: {}'.format(note))
        base_fname = '{:d}_{}'.format(stamp, note)
    else:
        base_fname = format(stamp, 'd')

    # Load data.
    print('Retrieving texts...')
    source = 'paragraph_tokens'
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    return_overall = shared_parameters.DATA_RETURN_OVERALL
    inputs, Y, categories, category_levels = \
        bookcave.get_data({source},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          categories_mode=categories_mode,
                          return_overall=return_overall)
    text_source_tokens = list(zip(*inputs[source]))[0]
    print('Retrieved {:d} texts.'.format(len(text_source_tokens)))

    # Create vectorized representations of the book texts.
    print('Vectorizing text...')
    max_words = shared_parameters.TEXT_MAX_WORDS
    vectorizer = TfidfVectorizer(
        preprocessor=identity,
        tokenizer=identity,
        analyzer='word',
        token_pattern=None,
        max_features=max_words,
        norm='l2',
        sublinear_tf=True)
    text_tokens = []
    for source_tokens in text_source_tokens:
        all_tokens = []
        for tokens in source_tokens:
            all_tokens.extend(tokens)
        text_tokens.append(all_tokens)
    X = vectorizer.fit_transform(text_tokens).todense()
    print('Vectorized text with {:d} unique words.'.format(len(vectorizer.get_feature_names())))

    # Create model.
    print('Creating model...')
    category_k = [len(levels) for levels in category_levels]
    dense_1_units = 256
    dense_1_activation = tf.keras.layers.LeakyReLU(alpha=.1)
    dense_1_l2 = .01
    dense_2_units = 256
    dense_2_activation = tf.keras.layers.LeakyReLU(alpha=.1)
    dense_2_l2 = .01
    dropout = .5
    label_mode = shared_parameters.LABEL_MODE_ORDINAL
    model = create_model(
        max_words,
        dense_1_units, dense_1_activation, dense_1_l2,
        dense_2_units, dense_2_activation, dense_2_l2,
        dropout, category_k, categories, label_mode)
    lr = 2**-10
    optimizer = Adam(lr=lr)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        loss = 'binary_crossentropy'
        metric = 'binary_accuracy'
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        loss = 'categorical_crossentropy'
        metric = 'categorical_accuracy'
    elif label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        loss = 'mse'
        metric = 'accuracy'
    else:
        raise ValueError('Unknown value for `label_mode`: {}'.format(label_mode))
    model.compile(optimizer, loss=loss, metrics=[metric])
    print('Done.')

    # Split data set.
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    val_size = shared_parameters.EVAL_VAL_SIZE  # v
    val_random_state = shared_parameters.EVAL_VAL_RANDOM_STATE
    Y_T = Y.transpose()  # (n, c)
    X_train, X_test, Y_train_T, Y_test_T = \
        train_test_split(X, Y_T, test_size=test_size, random_state=test_random_state)
    X_train, X_val, Y_train_T, Y_val_T = \
        train_test_split(X_train, Y_train_T, test_size=val_size, random_state=val_random_state)
    Y_train = Y_train_T.transpose()  # (c, n * (1 - b) * (1 - v))
    Y_val = Y_val_T.transpose()  # (c, n * (1 - b) * v)
    Y_test = Y_test_T.transpose()  # (c, n * b)

    # Transform labels based on the label mode.
    Y_train = shared_parameters.transform_labels(Y_train, category_k, label_mode)
    Y_val = shared_parameters.transform_labels(Y_val, category_k, label_mode)

    # Calculate class weights.
    use_class_weights = True
    class_weight_f = 'square inverse'
    if use_class_weights:
        category_class_weights = shared_parameters.get_category_class_weights(Y_train, label_mode, f=class_weight_f)
    else:
        category_class_weights = None

    # Train.
    shuffle = True
    plateau_monitor = 'val_loss'
    plateau_factor = .5
    plateau_patience = 12
    early_stopping_monitor = 'val_loss'
    early_stopping_min_delta = 2**-10
    early_stopping_patience = 24
    callbacks = [
        ReduceLROnPlateau(monitor=plateau_monitor, factor=plateau_factor, patience=plateau_patience),
        EarlyStopping(monitor=early_stopping_monitor, min_delta=early_stopping_min_delta, patience=early_stopping_patience)
    ]
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_val, Y_val),
        shuffle=shuffle,
        class_weight=category_class_weights,
        steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None
    )

    # Save the history to visualize loss over time.
    print('Saving training history...')
    if not os.path.exists(folders.HISTORY_PATH):
        os.mkdir(folders.HISTORY_PATH)
    history_path = os.path.join(folders.HISTORY_PATH, classifier_name)
    if not os.path.exists(history_path):
        os.mkdir(history_path)
    with open(os.path.join(history_path, '{}.txt'.format(base_fname)), 'w') as fd:
        for key in history.history.keys():
            values = history.history.get(key)
            fd.write('{} {}\n'.format(key, ' '.join(str(value) for value in values)))
    print('Done.')

    # Predict test instances.
    print('Predicting test instances...')
    Y_pred = model.predict(X_test)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        Y_pred = [ordinal.from_multi_hot_ordinal(y, threshold=.5) for y in Y_pred]
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        Y_pred = [np.argmax(y, axis=1) for y in Y_pred]
    elif label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        Y_pred = [np.maximum(0, np.minimum(k - 1, np.round(Y_pred[i] * k))) for i, k in enumerate(category_k)]
    else:
        raise ValueError('Unknown value for `label_mode`: {}'.format(label_mode))
    print('Done.')

    # Save model.
    save_model = False
    if save_model:
        models_path = os.path.join(folders.MODELS_PATH, classifier_name)
        label_mode_path = os.path.join(models_path, label_mode)
        model_path = os.path.join(label_mode_path, '{}.h5'.format(base_fname))
        print('Saving model to `{}`...'.format(model_path))
        if not os.path.exists(folders.MODELS_PATH):
            os.mkdir(folders.MODELS_PATH)
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        if not os.path.exists(label_mode_path):
            os.mkdir(label_mode_path)
        model.save(model_path)
        print('Done.')
    else:
        model_path = None

    # Calculate elapsed time.
    end_time = int(time.time())
    elapsed_s = end_time - start_time
    elapsed_m, elapsed_s = elapsed_s // 60, elapsed_s % 60
    elapsed_h, elapsed_m = elapsed_m // 60, elapsed_m % 60

    # Write results.
    print('Writing results...')

    if not os.path.exists(folders.LOGS_PATH):
        os.mkdir(folders.LOGS_PATH)
    logs_path = os.path.join(folders.LOGS_PATH, classifier_name)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    with open(os.path.join(logs_path, '{}.txt'.format(base_fname)), 'w') as fd:
        if note is not None:
            fd.write('{}\n\n'.format(note))
        fd.write('PARAMETERS\n\n')
        fd.write('batch_size={:d}\n'.format(batch_size))
        fd.write('steps_per_epoch={:d}\n'.format(steps_per_epoch))
        fd.write('epochs={:d}\n'.format(epochs))
        fd.write('\nHYPERPARAMETERS\n')
        fd.write('\nText\n')
        fd.write('subset_ratio={}\n'.format(str(subset_ratio)))
        fd.write('subset_seed={}\n'.format(str(subset_seed)))
        fd.write('min_len={:d}\n'.format(min_len))
        fd.write('max_len={:d}\n'.format(max_len))
        fd.write('min_tokens={:d}\n'.format(min_tokens))
        fd.write('\nLabels\n')
        fd.write('categories_mode=\'{}\'\n'.format(categories_mode))
        fd.write('return_overall={}\n'.format(return_overall))
        fd.write('\nVectorization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('\nModel\n')
        fd.write('dense_1_units={:d}\n'.format(dense_1_units))
        fd.write('dense_1_activation={} {}\n'.format(dense_1_activation.__class__.__name__,
                                                     dense_1_activation.__dict__))
        fd.write('dense_1_l2={}\n'.format(str(dense_2_l2)))
        fd.write('dense_2_units={:d}\n'.format(dense_2_units))
        fd.write('dense_2_activation={} {}\n'.format(dense_2_activation.__class__.__name__,
                                                     dense_2_activation.__dict__))
        fd.write('dense_2_l2={}\n'.format(str(dense_1_l2)))
        fd.write('dropout={:.1f}\n'.format(dropout))
        fd.write('label_mode={}\n'.format(label_mode))
        model.summary(print_fn=lambda x: fd.write('{}\n'.format(x)))
        fd.write('\nTraining\n')
        fd.write('optimizer={}\n'.format(optimizer.__class__.__name__))
        fd.write('lr={}\n'.format(str(lr)))
        fd.write('loss=\'{}\'\n'.format(loss))
        fd.write('metric=\'{}\'\n'.format(metric))
        fd.write('test_size={:.2f}\n'.format(test_size))
        fd.write('test_random_state={:d}\n'.format(test_random_state))
        fd.write('val_size={:.2f}\n'.format(val_size))
        fd.write('val_random_state={:d}\n'.format(val_random_state))
        fd.write('use_class_weights={}\n'.format(use_class_weights))
        if use_class_weights:
            fd.write('class_weight_f={}\n'.format(class_weight_f))
        fd.write('shuffle={}\n'.format(shuffle))
        fd.write('plateau_monitor={}\n'.format(plateau_monitor))
        fd.write('plateau_factor={}\n'.format(str(plateau_factor)))
        fd.write('plateau_patience={:d}\n'.format(plateau_patience))
        fd.write('early_stopping_monitor={}\n'.format(early_stopping_monitor))
        fd.write('early_stopping_min_delta={}\n'.format(str(early_stopping_min_delta)))
        fd.write('early_stopping_patience={:d}\n'.format(early_stopping_patience))
        fd.write('\nRESULTS\n\n')
        fd.write('Data size: {:d}\n'.format(len(X)))
        fd.write('Train size: {:d}\n'.format(len(X_train)))
        fd.write('Validation size: {:d}\n'.format(len(X_val)))
        fd.write('Test size: {:d}\n'.format(len(X_test)))
        if save_model:
            fd.write('Model path: \'{}\'\n'.format(model_path))
        else:
            fd.write('Model not saved.\n')
        fd.write('Time elapsed: {:d}h {:d}m {:d}s\n\n'.format(elapsed_h, elapsed_m, elapsed_s))
        evaluation.write_confusion_and_metrics(Y_test, Y_pred, fd, categories, overall_last=return_overall)

    if not os.path.exists(folders.PREDICTIONS_PATH):
        os.mkdir(folders.PREDICTIONS_PATH)
    predictions_path = os.path.join(folders.PREDICTIONS_PATH, classifier_name)
    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
    with open(os.path.join(predictions_path, '{}.txt'.format(base_fname)), 'w') as fd:
        evaluation.write_predictions(Y_test, Y_pred, fd, categories)

    print('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])
