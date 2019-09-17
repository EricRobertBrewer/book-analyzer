import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Concatenate, CuDNNGRU, Dense, Dropout, Embedding, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

from classification import evaluation, ordinal, shared_parameters
from classification.net.attention_with_context import AttentionWithContext
from classification.net.batch_generators import SingleInstanceBatchGenerator, VariableLengthBatchGenerator
import folders
from sites.bookcave import bookcave, bookcave_ids
from text import generate_data


def create_model(n_tokens, embedding_matrix, embedding_trainable,
                 sent_rnn, sent_rnn_units, sent_rnn_l2, sent_dense_units, sent_dense_activation, sent_dense_l2,
                 book_dense_units, book_dense_activation, book_dense_l2,
                 book_dropout, output_k, output_names, label_mode):
    # Sentence encoder.
    input_s = Input(shape=(n_tokens,), dtype='int32')  # (t)
    max_words, d = embedding_matrix.shape
    x_s = Embedding(max_words,
                    d,
                    weights=[embedding_matrix],
                    trainable=embedding_trainable)(input_s)  # (t, d)
    if sent_rnn_l2 is not None:
        sent_rnn_l2 = regularizers.l2(sent_rnn_l2)
    x_s = Bidirectional(sent_rnn(sent_rnn_units,
                                 kernel_regularizer=sent_rnn_l2,
                                 return_sequences=True))(x_s)  # (2t, h_s)
    if sent_dense_l2 is not None:
        sent_dense_l2 = regularizers.l2(sent_dense_l2)
    x_s = TimeDistributed(Dense(sent_dense_units,
                                activation=sent_dense_activation,
                                kernel_regularizer=sent_dense_l2))(x_s)  # (2t, c_s)
    x_s = AttentionWithContext()(x_s)  # (c_s)
    sentence_encoder = Model(input_s, x_s)

    # Consider maximum and average signals among all sentences of books.
    input_b = Input(shape=(None, n_tokens), dtype='float32')  # (s, t); s is not constant!
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
    return model


def main(argv):
    if len(argv) < 5 or len(argv) > 6:
        raise ValueError('Usage: <max_words> <n_tokens> <batch_size> <steps_per_epoch> <epochs> [note]')
    max_words = int(argv[0])  # The maximum size of the vocabulary.
    n_tokens = int(argv[1])  # The maximum number of tokens to process in each sentence.
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
    embedding_paths = [
        folders.EMBEDDING_GLOVE_300_PATH
    ]
    padding = 'pre'
    truncating = 'pre'
    categories_mode = 'soft'
    X = generate_data.load_X_sentences(max_words,
                                       n_tokens,
                                       padding=padding,
                                       truncating=truncating)
    Y = generate_data.load_Y(categories_mode)
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
    embedding_trainable = True
    sent_rnn = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
    sent_rnn_units = 128
    sent_rnn_l2 = .01
    sent_dense_units = 64
    sent_dense_activation = 'linear'
    sent_dense_l2 = .01
    book_dense_units = 512
    book_dense_activation = tf.keras.layers.LeakyReLU(alpha=.1)
    book_dense_l2 = .01
    book_dropout = .5
    label_mode = shared_parameters.LABEL_MODE_ORDINAL
    model = create_model(n_tokens, embedding_matrix, embedding_trainable,
                         sent_rnn, sent_rnn_units, sent_rnn_l2, sent_dense_units, sent_dense_activation, sent_dense_l2,
                         book_dense_units, book_dense_activation, book_dense_l2,
                         book_dropout, category_k, categories, label_mode)
    lr = .0001
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

    # Train.
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
    if batch_size == 1:
        train_generator = SingleInstanceBatchGenerator(X_train, Y_train, shuffle=True)
        val_generator = SingleInstanceBatchGenerator(X_val, Y_val, shuffle=True)
        test_generator = SingleInstanceBatchGenerator(X_test, Y_test, shuffle=True)
    else:
        X_shape = (n_tokens,)
        Y_shape = [(len(y[0]),) for y in Y_train]
        train_generator = VariableLengthBatchGenerator(X_train, X_shape, Y_train, Y_shape, batch_size, shuffle=True)
        val_generator = VariableLengthBatchGenerator(X_val, X_shape, Y_val, Y_shape, batch_size, shuffle=False)
        test_generator = VariableLengthBatchGenerator(X_test, X_shape, Y_test, Y_shape, batch_size, shuffle=False)
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None,
                                  epochs=epochs,
                                  class_weight=category_class_weights,
                                  validation_data=val_generator)

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
    Y_preds = model.predict_generator(test_generator)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        Y_preds = [ordinal.from_multi_hot_ordinal(y, threshold=.5) for y in Y_preds]
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        Y_preds = [np.argmax(y, axis=1) for y in Y_preds]
    elif label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        Y_preds = [np.maximum(0, np.minimum(k - 1, np.round(Y_preds[i] * k))) for i, k in enumerate(category_k)]
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
        fd.write('ids_fname={}\n'.format(bookcave_ids.get_ids_fname()))
        fd.write('\nLabels\n')
        fd.write('categories_mode=\'{}\'\n'.format(categories_mode))
        fd.write('\nTokenization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('n_tokens={:d}\n'.format(n_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        for embedding_path in embedding_paths:
            fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        fd.write('sent_rnn={}\n'.format(sent_rnn.__name__))
        fd.write('sent_rnn_units={:d}\n'.format(sent_rnn_units))
        fd.write('sent_rnn_l2={}\n'.format(str(sent_rnn_l2)))
        fd.write('sent_dense_units={:d}\n'.format(sent_dense_units))
        fd.write('sent_dense_activation=\'{}\'\n'.format(sent_dense_activation))
        fd.write('sent_dense_l2={}\n'.format(str(sent_dense_l2)))
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
        fd.write('\nRESULTS\n\n')
        fd.write('Data size: {:d}\n'.format(len(X)))
        fd.write('Train size: {:d}\n'.format(len(X_train)))
        fd.write('Validation size: {:d}\n'.format(len(X_val)))
        fd.write('Test size: {:d}\n'.format(len(X_test)))
        if save_model:
            fd.write('Model path: \'{}\'\n'.format(model_path))
        else:
            fd.write('Model not saved.\n')
        fd.write('Time elapsed: {:d}h {:d}m {:d}s\n'.format(elapsed_h, elapsed_m, elapsed_s))
        # Calculate statistics for predictions.
        category_confusion, category_metrics = zip(*[evaluation.get_confusion_and_metrics(Y_test[j], Y_preds[j])
                                                     for j in range(len(categories))])
        category_width = max(7, max([len(category) for category in categories]))
        averages = [sum([metrics[metric_i] for metrics in category_metrics])/len(category_metrics)
                    for metric_i in range(len(evaluation.METRIC_NAMES))]
        # Metric abbreviations.
        fd.write('\n')
        fd.write('{:>{w}}'.format('Metric', w=category_width))
        for abbreviation in evaluation.METRIC_ABBREVIATIONS:
            fd.write(' | {:^7}'.format(abbreviation))
        fd.write(' |\n')
        # Horizontal line.
        fd.write('{:>{w}}'.format('', w=category_width))
        for _ in range(len(category_metrics)):
            fd.write('-+-{}'.format('-'*7))
        fd.write('-+\n')
        # Metrics per category.
        for j, metrics in enumerate(category_metrics):
            fd.write('{:>{w}}'.format(categories[j], w=category_width))
            for value in metrics:
                fd.write(' | {:.5f}'.format(value))
            fd.write(' |\n')
        # Horizontal line.
        fd.write('{:>{w}}'.format('', w=category_width))
        for _ in range(len(category_metrics)):
            fd.write('-+-{}'.format('-'*7))
        fd.write('-+\n')
        # Average metrics.
        fd.write('{:>{w}}'.format('Average', w=category_width))
        for value in averages:
            fd.write(' | {:.5f}'.format(value))
        fd.write(' |\n')
        # Confusion matrices.
        for j, category in enumerate(categories):
            fd.write('\n`{}`\n'.format(category))
            confusion = category_confusion[j]
            fd.write(np.array2string(confusion))
            fd.write('\n')
    print('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])
