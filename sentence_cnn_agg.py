import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, Dense, Dropout, Embedding, Flatten, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, Input, MaxPool2D, Reshape, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

from classification import evaluation, ordinal, shared_parameters
from classification.net.batch_generators import SingleInstanceBatchGenerator, VariableLengthBatchGenerator
import folders
from sites.bookcave import bookcave, bookcave_ids
from text import generate_data


def create_model(
        n_sentence_tokens, embedding_matrix, embedding_trainable,
        sent_cnn_filters, sent_cnn_filter_sizes, sent_cnn_activation, sent_cnn_l2,
        book_dense_units, book_dense_activation, book_dense_l2,
        book_dropout, output_k, output_names, label_mode):
    # Sentence encoder.
    input_s = Input(shape=(n_sentence_tokens,), dtype='float32')  # (t)
    max_words, d = embedding_matrix.shape
    x_s = Embedding(max_words,
                    d,
                    weights=[embedding_matrix],
                    trainable=embedding_trainable)(input_s)  # (t, d)
    x_s = Reshape((n_sentence_tokens, d, 1))(x_s)  # (t, d, 1)
    if sent_cnn_l2 is not None:
        sent_cnn_l2 = regularizers.l2(sent_cnn_l2)
    X_s = [Conv2D(sent_cnn_filters,
                  (filter_size, d),
                  strides=(1, 1),
                  padding='valid',
                  activation=sent_cnn_activation,
                  kernel_regularizer=sent_cnn_l2)(x_s)
           for filter_size in sent_cnn_filter_sizes]  # [(t - z + 1, f)]; z = filter_size, f = filters
    X_s = [MaxPool2D(pool_size=(n_sentence_tokens - sent_cnn_filter_sizes[i] + 1, 1),
                     strides=(1, 1),
                     padding='valid')(x_s)
           for i, x_s in enumerate(X_s)]  # [(f, 1)]
    x_s = Concatenate(axis=1)(X_s)  # (f * |Z|); |Z| = length of filter_sizes
    x_s = Flatten()(x_s)
    sentence_encoder = Model(input_s, x_s)

    # Consider maximum and average signals among all sentences of books.
    input_b = Input(shape=(None, n_sentence_tokens), dtype='float32')  # (s, t); s is not constant!
    x_b = TimeDistributed(sentence_encoder)(input_b)  # (p, c_s)
    g_max_b = GlobalMaxPooling1D()(x_b)  # (c_s)
    g_avg_b = GlobalAveragePooling1D()(x_b)  # (c_s)
    x_b = Concatenate()([g_max_b, g_avg_b])  # (2c_s)
    if book_dense_l2 is not None:
        book_dense_l2 = regularizers.l2(book_dense_l2)
    x_b = Dense(book_dense_units,
                kernel_regularizer=book_dense_l2)(x_b)  # (c_b)
    x_b = book_dense_activation(x_b)  # (c_b)
    x_b = Dropout(book_dropout)(x_b)  # (c_b)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        outputs = [Dense(k - 1, activation='sigmoid', name=output_names[i])(x_b) for i, k in enumerate(output_k)]
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        outputs = [Dense(k, activation='softmax', name=output_names[i])(x_b) for i, k in enumerate(output_k)]
    elif label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        outputs = [Dense(1, activation='linear', name=output_names[i])(x_b) for i in range(len(output_k))]
    else:
        raise ValueError('Unknown value for `1abel_mode`: {}'.format(label_mode))
    model = Model(input_b, outputs)
    return sentence_encoder, model


def main(argv):
    if len(argv) < 5 or len(argv) > 6:
        raise ValueError('Usage: <max_words> <n_sentence_tokens> <batch_size> <steps_per_epoch> <epochs> [note]')
    max_words = int(argv[0])  # The maximum size of the vocabulary.
    n_sentence_tokens = int(argv[1])  # The maximum number of tokens to process in each sentence.
    batch_size = int(argv[2])
    steps_per_epoch = int(argv[3])
    epochs = int(argv[4])
    note = None
    if len(argv) > 5:
        note = argv[5]

    script_name = os.path.basename(__file__)
    classifier_name = script_name[:script_name.rindex('.')]

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
    print('Loading data...')
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    padding = 'pre'
    truncating = 'pre'
    categories_mode = 'soft'
    embedding_paths = [
        folders.EMBEDDING_GLOVE_100_PATH
    ]
    X = generate_data.load_X_sentences(max_words,
                                       n_sentence_tokens,
                                       padding=padding,
                                       truncating=truncating,
                                       subset_ratio=subset_ratio,
                                       subset_seed=subset_seed)
    Y = generate_data.load_Y(categories_mode, subset_ratio=subset_ratio, subset_seed=subset_seed)
    embedding_matrix = generate_data.load_embedding_matrix(max_words, embedding_path=embedding_paths[0])
    for i in range(1, len(embedding_paths)):
        other_embedding_matrix = generate_data.load_embedding_matrix(max_words, embedding_path=embedding_paths[i])
        embedding_matrix = np.concatenate((embedding_matrix, other_embedding_matrix), axis=1)
    categories = bookcave.CATEGORIES
    category_levels = bookcave.CATEGORY_LEVELS[categories_mode]
    print('Done.')

    # Create model.
    print('Creating model...')
    category_k = [len(levels) for levels in category_levels]
    embedding_trainable = False
    sent_cnn_filters = 8
    sent_cnn_filter_sizes = [1, 2, 3, 4]
    sent_cnn_activation = 'elu'
    sent_cnn_l2 = .01
    book_dense_units = 128
    book_dense_activation = tf.keras.layers.LeakyReLU(alpha=.1)
    book_dense_l2 = .01
    book_dropout = .5
    label_mode = shared_parameters.LABEL_MODE_ORDINAL
    sentence_encoder, model = create_model(
        n_sentence_tokens, embedding_matrix, embedding_trainable,
        sent_cnn_filters, sent_cnn_filter_sizes, sent_cnn_activation, sent_cnn_l2,
        book_dense_units, book_dense_activation, book_dense_l2,
        book_dropout, category_k, categories, label_mode)
    lr = 2**-16
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
        raise ValueError('Unknown value for `1abel_mode`: {}'.format(label_mode))
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

    # Calculate class weights.
    use_class_weights = True
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        Y_train = [ordinal.to_multi_hot_ordinal(Y_train[j], k=k) for j, k in enumerate(category_k)]
        Y_val = [ordinal.to_multi_hot_ordinal(Y_val[j], k=k) for j, k in enumerate(category_k)]
        if use_class_weights:
            category_class_weights = []  # [[dict]]
            for y_train in Y_train:
                class_weights = []
                for i in range(y_train.shape[1]):
                    ones_count = sum(y_train[:, i] == 1)
                    class_weight = {0: 1 / (len(y_train) - ones_count + 1), 1: 1 / (ones_count + 1)}
                    class_weights.append(class_weight)
                category_class_weights.append(class_weights)
        else:
            category_class_weights = None
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        if use_class_weights:
            category_class_weights = []  # [dict]
            for j, y_train in enumerate(Y_train):
                bincount = np.bincount(y_train, minlength=category_k[j])
                class_weight = {i: 1 / (count + 1) for i, count in enumerate(bincount)}
                category_class_weights.append(class_weight)
        else:
            category_class_weights = None
        Y_train = [utils.to_categorical(Y_train[j], num_classes=k) for j, k in enumerate(category_k)]
        Y_val = [utils.to_categorical(Y_val[j], num_classes=k) for j, k in enumerate(category_k)]
    elif label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        category_class_weights = None
        Y_train = [Y_train[j] / k for j, k in enumerate(category_k)]
        Y_val = [Y_val[j] / k for j, k in enumerate(category_k)]
    else:
        raise ValueError('Unknown value for `1abel_mode`: {}'.format(label_mode))

    # Create generators.
    shuffle = True
    if batch_size == 1:
        train_generator = SingleInstanceBatchGenerator(X_train, Y_train, shuffle=shuffle)
        val_generator = SingleInstanceBatchGenerator(X_val, Y_val, shuffle=False)
        test_generator = SingleInstanceBatchGenerator(X_test, Y_test, shuffle=False)
    else:
        X_shape = (n_sentence_tokens,)
        Y_shape = [(len(y[0]),) for y in Y_train]
        train_generator = VariableLengthBatchGenerator(X_train, X_shape, Y_train, Y_shape, batch_size, shuffle=shuffle)
        val_generator = VariableLengthBatchGenerator(X_val, X_shape, Y_val, Y_shape, batch_size, shuffle=False)
        test_generator = VariableLengthBatchGenerator(X_test, X_shape, Y_test, Y_shape, batch_size, shuffle=False)

    # Train.
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None,
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  class_weight=category_class_weights)

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
    Y_pred = model.predict_generator(test_generator)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        Y_pred = [ordinal.from_multi_hot_ordinal(y, threshold=.5) for y in Y_pred]
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        Y_pred = [np.argmax(y, axis=1) for y in Y_pred]
    elif label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        Y_pred = [np.maximum(0, np.minimum(k - 1, np.round(Y_pred[i] * k))) for i, k in enumerate(category_k)]
    else:
        raise ValueError('Unknown value for `1abel_mode`: {}'.format(label_mode))
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
        fd.write('steps_per_epoch={:d}\n'.format(steps_per_epoch))
        fd.write('epochs={:d}\n'.format(epochs))
        fd.write('\nHYPERPARAMETERS\n')
        fd.write('\nText\n')
        fd.write('subset_ratio={}\n'.format(str(subset_ratio)))
        fd.write('subset_seed={}\n'.format(str(subset_seed)))
        fd.write('ids_fname={}\n'.format(bookcave_ids.get_ids_fname()))
        fd.write('\nLabels\n')
        fd.write('categories_mode=\'{}\'\n'.format(categories_mode))
        fd.write('\nTokenization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('n_sentence_tokens={:d}\n'.format(n_sentence_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        for embedding_path in embedding_paths:
            fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        fd.write('sent_cnn_filters={:d}\n'.format(sent_cnn_filters))
        fd.write('sent_cnn_filter_sizes={}\n'.format(str(sent_cnn_filter_sizes)))
        fd.write('sent_cnn_activation=\'{}\'\n'.format(sent_cnn_activation))
        fd.write('sent_cnn_l2={}\n'.format(str(sent_cnn_l2)))
        fd.write('book_dense_units={:d}\n'.format(book_dense_units))
        fd.write('book_dense_activation={} {}\n'.format(book_dense_activation.__class__.__name__,
                                                        book_dense_activation.__dict__))
        fd.write('book_dense_l2={}\n'.format(str(book_dense_l2)))
        fd.write('book_dropout={:.1f}\n'.format(book_dropout))
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
        fd.write('shuffle={}\n'.format(shuffle))
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
        evaluation.write_confusion_and_metrics(Y_test, Y_pred, fd, categories)

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
