import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Concatenate, CuDNNGRU, Dense, Dropout, Embedding, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

from classification import evaluation, ordinal, shared_parameters
from classification.net.attention_with_context import AttentionWithContext
from classification.net.batch_generators import SingleInstanceBatchGenerator
import folders
from sites.bookcave import bookcave
from text import load_embeddings


def create_model(output_k, output_names, n_tokens, embedding_matrix, embedding_trainable,
                 para_rnn, para_rnn_units, para_rnn_l2, para_dense_units, para_dense_activation, para_dense_l2,
                 book_dense_units, book_dense_activation, book_dense_l2,
                 book_dropout, label_mode):
    # Paragraph encoder.
    input_p = Input(shape=(n_tokens,), dtype='float32')  # (t)
    max_words, d = embedding_matrix.shape
    x_p = Embedding(max_words,
                    d,
                    weights=[embedding_matrix],
                    trainable=embedding_trainable)(input_p)  # (t, d)
    if para_rnn_l2 is not None:
        para_rnn_l2 = regularizers.l2(para_rnn_l2)
    x_p = Bidirectional(para_rnn(para_rnn_units,
                                 kernel_regularizer=para_rnn_l2,
                                 return_sequences=True))(x_p)  # (2t, h_p)
    if para_dense_l2 is not None:
        para_dense_l2 = regularizers.l2(para_dense_l2)
    x_p = TimeDistributed(Dense(para_dense_units,
                                activation=para_dense_activation,
                                kernel_regularizer=para_dense_l2))(x_p)  # (2t, c_p)
    x_p = AttentionWithContext()(x_p)  # (c_p)
    paragraph_encoder = Model(input_p, x_p)

    # Consider maximum and average signals among all paragraphs of books.
    input_b = Input(shape=(None, n_tokens), dtype='float32')  # (p, t); p is not constant!
    x_b = TimeDistributed(paragraph_encoder)(input_b)  # (p, c_p)
    g_max_b = GlobalMaxPooling1D()(x_b)  # (c_p)
    g_avg_b = GlobalAveragePooling1D()(x_b)  # (c_p)
    x_b = Concatenate()([g_max_b, g_avg_b])  # (2c_p)
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
    if len(argv) < 2 or len(argv) > 3:
        raise ValueError('Usage: <steps_per_epoch> <epochs> [note]')
    steps_per_epoch = int(argv[0])
    epochs = int(argv[1])
    note = None
    if len(argv) > 2:
        note = argv[2]

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

    # Load data.
    print('Retrieving texts...')
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_MIN_LEN
    max_len = shared_parameters.DATA_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'paragraph_tokens'},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens)
    text_paragraph_tokens, _ = zip(*inputs['paragraph_tokens'])
    print('Retrieved {:d} texts.'.format(len(text_paragraph_tokens)))

    # Tokenize.
    print('Tokenizing...')
    max_words = 8192  # The maximum size of the vocabulary.
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_paragraphs = []
    for paragraph_tokens in text_paragraph_tokens:
        for tokens in paragraph_tokens:
            all_paragraphs.append(split.join(tokens))
    tokenizer.fit_on_texts(all_paragraphs)
    print('Done.')

    # Convert to sequences.
    print('Converting texts to sequences...')
    n_tokens = 128  # The maximum number of tokens to process in each paragraph.
    padding = 'pre'
    truncating = 'pre'
    X = [np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in paragraph_tokens]),
                                maxlen=n_tokens,
                                padding=padding,
                                truncating=truncating))
         for paragraph_tokens in text_paragraph_tokens]  # [text_i][paragraph_i][token_i]
    print('Done.')

    # Load embedding.
    print('Loading embedding matrix...')
    embedding_path = folders.EMBEDDING_GLOVE_100_PATH
    embedding_matrix = load_embeddings.get_embedding(tokenizer, embedding_path, max_words, header=False)
    print('Done.')

    # Create model.
    print('Creating model...')
    category_k = [len(levels) for levels in category_levels]
    embedding_trainable = False
    para_rnn = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
    para_rnn_units = 128
    para_rnn_l2 = .01
    para_dense_units = 64
    para_dense_activation = 'linear'
    para_dense_l2 = .01
    book_dense_units = 512
    book_dense_activation = tf.keras.layers.LeakyReLU(alpha=.1)
    book_dense_l2 = .01
    book_dropout = .5
    label_mode = shared_parameters.LABEL_MODE_ORDINAL
    model = create_model(category_k, categories, n_tokens, embedding_matrix, embedding_trainable,
                         para_rnn, para_rnn_units, para_rnn_l2, para_dense_units, para_dense_activation, para_dense_l2,
                         book_dense_units, book_dense_activation, book_dense_l2, book_dropout,
                         label_mode)
    lr = .000015625
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
    train_generator = SingleInstanceBatchGenerator(X_train, Y_train, shuffle=True)
    val_generator = SingleInstanceBatchGenerator(X_val, Y_val, shuffle=False)
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
    with open(os.path.join(history_path, '{:d}.txt'.format(stamp)), 'w') as fd:
        for key in history.history.keys():
            values = history.history.get(key)
            fd.write('{} {}\n'.format(key, ' '.join(str(value) for value in values)))
    print('Done.')

    # Predict test instances.
    print('Predicting test instances...')
    test_generator = SingleInstanceBatchGenerator(X_test, Y_test, shuffle=False)
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
        model_path = os.path.join(label_mode_path, '{:d}.h5'.format(stamp))
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
    with open(os.path.join(logs_path, '{:d}.txt'.format(stamp)), 'w') as fd:
        if note is not None:
            fd.write('Note: {}\n\n'.format(note))
        fd.write('PARAMETERS\n\n')
        fd.write('steps_per_epoch={:d}\n'.format(steps_per_epoch))
        fd.write('epochs={:d}\n'.format(epochs))
        fd.write('\nHYPERPARAMETERS\n')
        fd.write('\nText\n')
        fd.write('subset_ratio={:.3f}\n'.format(subset_ratio))
        fd.write('subset_seed={:d}\n'.format(subset_seed))
        fd.write('min_len={:d}\n'.format(min_len))
        fd.write('max_len={:d}\n'.format(max_len))
        fd.write('min_tokens={:d}\n'.format(min_tokens))
        fd.write('\nTokenization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('n_tokens={:d}\n'.format(n_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        fd.write('para_rnn={}\n'.format(para_rnn.__name__))
        fd.write('para_rnn_units={:d}\n'.format(para_rnn_units))
        fd.write('para_rnn_l2={}\n'.format(str(para_rnn_l2)))
        fd.write('para_dense_units={:d}\n'.format(para_dense_units))
        fd.write('para_dense_activation=\'{}\'\n'.format(para_dense_activation))
        fd.write('para_dense_l2={}\n'.format(str(para_dense_l2)))
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
        fd.write('Data size: {:d}\n'.format(len(text_paragraph_tokens)))
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
        averages = [sum([metrics[metric_i] for metrics in category_metrics])/len(category_metrics)
                    for metric_i in range(len(evaluation.METRIC_NAMES))]
        # Metric abbreviations.
        fd.write('\n')
        fd.write('{:>24}'.format('Metric'))
        for abbreviation in evaluation.METRIC_ABBREVIATIONS:
            fd.write(' | {:^7}'.format(abbreviation))
        fd.write(' |\n')
        # Horizontal line.
        fd.write('{:>24}'.format(''))
        for _ in range(len(category_metrics)):
            fd.write('-+-{}'.format('-'*7))
        fd.write('-+\n')
        # Metrics per category.
        for j, metrics in enumerate(category_metrics):
            fd.write('{:>24}'.format(categories[j]))
            for value in metrics:
                fd.write(' | {:.5f}'.format(value))
            fd.write(' |\n')
        # Horizontal line.
        fd.write('{:>24}'.format(''))
        for _ in range(len(category_metrics)):
            fd.write('-+-{}'.format('-'*7))
        fd.write('-+\n')
        # Average metrics.
        fd.write('{:>24}'.format('Average'))
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
