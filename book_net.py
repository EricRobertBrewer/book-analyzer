import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Concatenate, Conv2D, CuDNNGRU, Dense, Dropout, Embedding, \
    Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, Input, MaxPool2D, Reshape, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers
# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
from sklearn.model_selection import train_test_split

from classification import evaluation, ordinal, shared_parameters
from classification.net.attention_with_context import AttentionWithContext
from classification.net.batch_generators import SingleInstanceBatchGenerator, VariableLengthBatchGenerator
import folders
from sites.bookcave import bookcave
from text import load_embeddings


def create_source_rnn(net_params, x_p):
    rnn = net_params['rnn']
    rnn_units = net_params['rnn_units']
    rnn_l2 = regularizers.l2(net_params['rnn_l2']) if net_params['rnn_l2'] is not None else None
    rnn_dense_units = net_params['rnn_dense_units']
    rnn_dense_activation = net_params['rnn_dense_activation']
    rnn_dense_l2 = regularizers.l2(net_params['rnn_dense_l2']) if net_params['rnn_dense_l2'] is not None else None
    rnn_agg = net_params['rnn_agg']
    x_p = Bidirectional(rnn(rnn_units,
                            kernel_regularizer=rnn_l2,
                            return_sequences=True))(x_p)  # (2T, h_p)
    x_p = TimeDistributed(Dense(rnn_dense_units,
                                activation=rnn_dense_activation,
                                kernel_regularizer=rnn_dense_l2))(x_p)  # (2T, c_p)
    if rnn_agg == 'attention':
        x_p = AttentionWithContext()(x_p)  # (c_p)
    elif rnn_agg == 'maxavg':
        x_p = Concatenate()([
            GlobalMaxPooling1D()(x_p),
            GlobalAveragePooling1D()(x_p)
        ])  # (2c_p)
    elif rnn_agg == 'max':
        x_p = GlobalMaxPooling1D()(x_p)  # (c_p)
    else:
        raise ValueError('Unknown `rnn_agg`: {}'.format(rnn_agg))
    return x_p


def create_source_cnn(n_tokens, d, net_params, x_p):
    cnn_filters = net_params['cnn_filters']
    cnn_filter_sizes = net_params['cnn_filter_sizes']
    cnn_activation = net_params['cnn_activation']
    cnn_l2 = regularizers.l2(net_params['cnn_l2']) if net_params['cnn_l2'] is not None else None
    x_p = Reshape((n_tokens, d, 1))(x_p)  # (T, d, 1)
    X_p = [Conv2D(cnn_filters,
                  (filter_size, d),
                  strides=(1, 1),
                  padding='valid',
                  activation=cnn_activation,
                  kernel_regularizer=cnn_l2)(x_p)
           for filter_size in cnn_filter_sizes]  # [(T - z + 1, f)]; z = filter_size, f = filters
    X_p = [MaxPool2D(pool_size=(n_tokens - cnn_filter_sizes[i] + 1, 1),
                     strides=(1, 1),
                     padding='valid')(x_p)
           for i, x_p in enumerate(X_p)]  # [(f, 1)]
    x_p = Concatenate(axis=1)(X_p)  # (f * |Z|); |Z| = length of filter_sizes
    x_p = Flatten()(x_p)  # (f * |Z|)
    return x_p


def create_model(
        n_tokens, embedding_matrix, embedding_trainable,
        net_mode, net_params,
        agg_mode, agg_params,
        book_dense_units, book_dense_activation, book_dense_l2,
        book_dropout, output_k, output_names, label_mode):
    # Source encoder.
    input_p = Input(shape=(n_tokens,), dtype='float32')  # (T)
    max_words, d = embedding_matrix.shape
    x_p = Embedding(max_words,
                    d,
                    weights=[embedding_matrix],
                    trainable=embedding_trainable)(input_p)  # (T, d)
    if net_mode == 'rnn':
        x_p = create_source_rnn(net_params, x_p)
    elif net_mode == 'cnn':
        x_p = create_source_cnn(n_tokens, d, net_params, x_p)
    elif net_mode == 'rnncnn':
        x_p = Concatenate()([
            create_source_rnn(net_params, x_p),
            create_source_cnn(n_tokens, d, net_params, x_p)
        ])
    else:
        raise ValueError('Unknown `net_mode`: {}'.format(net_mode))
    source_encoder = Model(input_p, x_p)  # (m_p); constant per configuration

    # Consider signals among all sources of books.
    input_b = Input(shape=(None, n_tokens), dtype='float32')  # (P, T); P is not constant per instance!
    x_b = TimeDistributed(source_encoder)(input_b)  # (P, m_p)
    if agg_mode == 'maxavg':
        x_b = Concatenate()([
            GlobalMaxPooling1D()(x_b),
            GlobalAveragePooling1D()(x_b)
        ])  # (2m_p)
    elif agg_mode == 'max':
        x_b = GlobalMaxPooling1D()(x_b)  # (m_p)
    elif agg_mode == 'avg':
        x_b = GlobalAveragePooling1D()(x_b)  # (m_p)
    elif agg_mode == 'rnn':
        agg_rnn = agg_params['rnn']
        agg_rnn_units = agg_params['rnn_units']
        agg_rnn_l2 = regularizers.l2(agg_params['rnn_l2']) if agg_params['rnn_l2'] is not None else None
        x_b = Bidirectional(agg_rnn(agg_rnn_units,
                                    kernel_regularizer=agg_rnn_l2,
                                    return_sequences=False),
                            merge_mode='concat')(x_b)  # (2h_b)
    else:
        raise ValueError('Unknown `agg_mode`: {}'.format(agg_mode))
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
        raise ValueError('Unknown value for `label_mode`: {}'.format(label_mode))
    return Model(input_b, outputs)


def main(argv):
    if len(argv) < 7 or len(argv) > 8:
        raise ValueError('Usage: <source_mode> <net_mode> <agg_mode> <label_mode> <batch_size> <steps_per_epoch> <epochs> [note]')
    source_mode = argv[0]
    net_mode = argv[1]
    agg_mode = argv[2]
    label_mode = argv[3]
    batch_size = int(argv[4])
    steps_per_epoch = int(argv[5])
    epochs = int(argv[6])
    note = None
    if len(argv) > 7:
        note = argv[7]

    classifier_name = '{}_{}_{}_{}'.format(source_mode, net_mode, agg_mode, label_mode)

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
    if source_mode == 'paragraph':
        source = 'paragraph_tokens'
        min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
        max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    elif source_mode == 'sentence':
        source = 'sentence_tokens'
        min_len = shared_parameters.DATA_SENTENCE_MIN_LEN
        max_len = shared_parameters.DATA_SENTENCE_MAX_LEN
    else:
        raise ValueError('Unknown `source_mode`: {}'.format(source_mode))
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    inputs, Y, categories, category_levels = \
        bookcave.get_data({source},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          categories_mode=categories_mode)
    text_source_tokens = list(zip(*inputs[source]))[0]
    print('Retrieved {:d} texts.'.format(len(text_source_tokens)))

    # Tokenize.
    print('Tokenizing...')
    max_words = shared_parameters.TEXT_MAX_WORDS
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_sources = []
    for source_tokens in text_source_tokens:
        for tokens in source_tokens:
            all_sources.append(split.join(tokens))
    tokenizer.fit_on_texts(all_sources)
    print('Done.')

    # Convert to sequences.
    print('Converting texts to sequences...')
    if source_mode == 'paragraph':
        n_tokens = shared_parameters.TEXT_N_PARAGRAPH_TOKENS
    elif source_mode == 'sentence':
        n_tokens = shared_parameters.TEXT_N_SENTENCE_TOKENS
    else:
        raise ValueError('Unknown `source_mode`: {}'.format(source_mode))
    padding = shared_parameters.TEXT_PADDING
    truncating = shared_parameters.TEXT_TRUNCATING
    X = [np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in source_tokens]),
                                maxlen=n_tokens,
                                padding=padding,
                                truncating=truncating))
         for source_tokens in text_source_tokens]
    print('Done.')

    # Load embedding.
    print('Loading embedding matrix...')
    embedding_path = folders.EMBEDDING_GLOVE_300_PATH
    embedding_matrix = load_embeddings.load_embedding(tokenizer, embedding_path, max_words)
    print('Done.')

    # Create model.
    print('Creating model...')
    category_k = [len(levels) for levels in category_levels]
    embedding_trainable = False
    net_params = dict()
    if net_mode == 'rnn' or net_mode == 'rnncnn':
        net_params['rnn'] = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
        net_params['rnn_units'] = 128
        net_params['rnn_l2'] = .01
        net_params['rnn_dense_units'] = 64
        net_params['rnn_dense_activation'] = 'elu'
        net_params['rnn_dense_l2'] = .01
        net_params['rnn_agg'] = 'attention'
    if net_mode == 'cnn' or net_mode == 'rnncnn':
        net_params['cnn_filters'] = 16
        net_params['cnn_filter_sizes'] = [1, 2, 3, 4]
        net_params['cnn_activation'] = 'elu'
        net_params['cnn_l2'] = .01
    agg_params = dict()
    if agg_mode == 'maxavg':
        pass
    elif agg_mode == 'max':
        pass
    elif agg_mode == 'avg':
        pass
    elif agg_mode == 'rnn':
        agg_params['rnn'] = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
        agg_params['rnn_units'] = 64
        agg_params['rnn_l2'] = .01
    else:
        raise ValueError('Unknown `agg_mode`: {}'.format(agg_mode))
    book_dense_units = 128
    book_dense_activation = tf.keras.layers.LeakyReLU(alpha=.1)
    book_dense_l2 = .01
    book_dropout = .5
    model = create_model(
        n_tokens, embedding_matrix, embedding_trainable,
        net_mode, net_params,
        agg_mode, agg_params,
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

    # Create generators.
    shuffle = True
    if batch_size == 1:
        train_generator = SingleInstanceBatchGenerator(X_train, Y_train, shuffle=shuffle)
        val_generator = SingleInstanceBatchGenerator(X_val, Y_val, shuffle=False)
        test_generator = SingleInstanceBatchGenerator(X_test, Y_test, shuffle=False)
    else:
        X_shape = (n_tokens,)
        Y_shape = [(len(y[0]),) for y in Y_train]
        train_generator = VariableLengthBatchGenerator(X_train, X_shape, Y_train, Y_shape, batch_size, shuffle=shuffle)
        val_generator = VariableLengthBatchGenerator(X_val, X_shape, Y_val, Y_shape, batch_size, shuffle=False)
        test_generator = VariableLengthBatchGenerator(X_test, X_shape, Y_test, Y_shape, batch_size, shuffle=False)

    # Train.
    plateau_monitor = 'val_loss'
    plateau_factor = .5
    if net_mode == 'rnn' or net_mode == 'rnncnn':
        plateau_patience = 3
    elif net_mode == 'cnn':
        plateau_patience = 6
    else:
        raise ValueError('Unknown `net_mode`: {}'.format(net_mode))
    early_stopping_monitor = 'val_loss'
    early_stopping_min_delta = 2**-10
    if net_mode == 'rnn' or net_mode == 'rnncnn':
        early_stopping_patience = 6
    elif net_mode == 'cnn':
        early_stopping_patience = 12
    else:
        raise ValueError('Unknown `net_mode`: {}'.format(net_mode))
    callbacks = [
        ReduceLROnPlateau(monitor=plateau_monitor, factor=plateau_factor, patience=plateau_patience),
        EarlyStopping(monitor=early_stopping_monitor, min_delta=early_stopping_min_delta, patience=early_stopping_patience)
    ]
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None,
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  class_weight=category_class_weights,
                                  callbacks=callbacks)

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
        fd.write('\nTokenization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('n_tokens={:d}\n'.format(n_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        if net_mode == 'rnn' or net_mode == 'rnncnn':
            fd.write('rnn={}\n'.format(net_params['rnn'].__name__))
            fd.write('rnn_units={:d}\n'.format(net_params['rnn_units']))
            fd.write('rnn_l2={}\n'.format(str(net_params['rnn_l2'])))
            fd.write('rnn_dense_units={:d}\n'.format(net_params['rnn_dense_units']))
            fd.write('rnn_dense_activation=\'{}\'\n'.format(net_params['rnn_dense_activation']))
            fd.write('rnn_dense_l2={}\n'.format(str(net_params['rnn_dense_l2'])))
            fd.write('rnn_agg={}\n'.format(net_params['rnn_agg']))
        if net_mode == 'cnn' or net_mode == 'rnncnn':
            fd.write('cnn_filters={:d}\n'.format(net_params['cnn_filters']))
            fd.write('cnn_filter_sizes={}\n'.format(str(net_params['cnn_filter_sizes'])))
            fd.write('cnn_activation=\'{}\'\n'.format(net_params['cnn_activation']))
            fd.write('cnn_l2={}\n'.format(str(net_params['cnn_l2'])))
        if agg_mode == 'maxavg':
            pass
        elif agg_mode == 'max':
            pass
        elif agg_mode == 'avg':
            pass
        elif agg_mode == 'rnn':
            fd.write('agg_rnn={}\n'.format(agg_params['rnn'].__name__))
            fd.write('agg_rnn_units={:d}\n'.format(agg_params['rnn_units']))
            fd.write('agg_rnn_l2={}\n'.format(str(agg_params['rnn_l2'])))
        else:
            raise ValueError('Unknown `agg_mode`: {}'.format(agg_mode))
        fd.write('book_dense_units={:d}\n'.format(book_dense_units))
        fd.write('book_dense_activation={} {}\n'.format(book_dense_activation.__class__.__name__,
                                                        book_dense_activation.__dict__))
        fd.write('book_dense_l2={}\n'.format(str(book_dense_l2)))
        fd.write('book_dropout={:.1f}\n'.format(book_dropout))
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
