import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, CuDNNGRU, Dense, Dropout, Embedding, GRU, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

from classification import evaluation, ordinal, shared_parameters
from classification.net import attention_with_context
import folders
from sites.bookcave import bookcave
from text import load_embeddings


def create_model(output_k, output_names, n_paragraphs, n_tokens, embedding_matrix, embedding_trainable,
                 word_rnn, word_rnn_units, word_rnn_l2, word_dense_units, word_dense_activation, word_dense_l2,
                 book_rnn, book_rnn_units, book_rnn_l2, book_dense_units, book_dense_activation, book_dense_l2,
                 book_dropout, label_mode):
    # Word encoder.
    input_w = Input(shape=(n_tokens,), dtype='float32')  # (t)
    max_words, d = embedding_matrix.shape
    x_w = Embedding(max_words,
                    d,
                    weights=[embedding_matrix],
                    trainable=embedding_trainable)(input_w)  # (t, d)
    if word_rnn_l2 is not None:
        word_rnn_l2 = regularizers.l2(word_rnn_l2)
    x_w = Bidirectional(word_rnn(word_rnn_units,
                                 kernel_regularizer=word_rnn_l2,
                                 return_sequences=True))(x_w)  # (t, h_w)
    if word_dense_l2 is not None:
        word_dense_l2 = regularizers.l2(word_dense_l2)
    x_w = TimeDistributed(Dense(word_dense_units,
                                activation=word_dense_activation,
                                kernel_regularizer=word_dense_l2))(x_w)  # (2t, c_w)
    x_w = attention_with_context.AttentionWithContext()(x_w)  # (c_w)
    word_encoder = Model(input_w, x_w)

    # Paragraph encoder.
    input_p = Input(shape=(n_paragraphs, n_tokens), dtype='float32')  # (s, t)
    x_p = TimeDistributed(word_encoder)(input_p)  # (s, c_w)
    if book_rnn_l2 is not None:
        book_rnn_l2 = regularizers.l2(book_rnn_l2)
    x_p = Bidirectional(book_rnn(book_rnn_units,
                                 kernel_regularizer=book_rnn_l2,
                                 return_sequences=True))(x_p)  # (s, h_b)
    if book_dense_l2 is not None:
        book_dense_l2 = regularizers.l2(book_dense_l2)
    x_p = TimeDistributed(Dense(book_dense_units,
                                activation=book_dense_activation,
                                kernel_regularizer=book_dense_l2))(x_p)  # (s, c_b)
    x_p = attention_with_context.AttentionWithContext()(x_p)  # (c_b)
    x_p = Dropout(book_dropout)(x_p)  # (c_b)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        outputs = [Dense(k - 1, activation='sigmoid', name=output_names[i])(x_p) for i, k in enumerate(output_k)]
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        outputs = [Dense(k, activation='softmax', name=output_names[i])(x_p) for i, k in enumerate(output_k)]
    elif label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        outputs = [Dense(1, activation='linear', name=output_names[i])(x_p) for i in range(len(output_k))]
    else:
        raise ValueError('Unknown value for `1abel_mode`: {}'.format(label_mode))
    model = Model(input_p, outputs)
    return model


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        raise ValueError('Usage: <batch_size> <epochs> [note]')
    batch_size = int(argv[0])
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
    print('\nRetrieving texts...')
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_MIN_LEN
    max_len = shared_parameters.DATA_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'tokens'},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens)
    text_paragraph_tokens, _ = zip(*inputs['tokens'])
    print('Retrieved {:d} texts.'.format(len(text_paragraph_tokens)))

    # Tokenize.
    print('\nTokenizing...')
    max_words = 8192  # The maximum size of the vocabulary.
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_sentences = []
    for paragraph_tokens in text_paragraph_tokens:
        for tokens in paragraph_tokens:
            all_sentences.append(split.join(tokens))
    tokenizer.fit_on_texts(all_sentences)
    print('Done.')

    # Convert to sequences.
    print('\nConverting texts to sequences...')
    n_paragraphs = max_len  # The maximum number of paragraphs to process in each text.
    n_tokens = 128  # The maximum number of tokens to process in each paragraph.
    padding = 'pre'
    truncating = 'pre'
    X = np.zeros((len(text_paragraph_tokens), n_paragraphs, n_tokens), dtype=np.float32)
    for text_i, paragraph_tokens in enumerate(text_paragraph_tokens):
        if len(paragraph_tokens) > n_paragraphs:
            # Truncate two thirds of the remainder at the beginning and one third at the end.
            start = int(2/3*(len(paragraph_tokens) - n_paragraphs))
            usable_paragraph_tokens = paragraph_tokens[start:start + n_paragraphs]
        else:
            usable_paragraph_tokens = paragraph_tokens
        sequences = tokenizer.texts_to_sequences([split.join(tokens) for tokens in usable_paragraph_tokens])
        X[text_i, :len(sequences)] = pad_sequences(sequences,
                                                   maxlen=n_tokens,
                                                   padding=padding,
                                                   truncating=truncating)
    print('Done.')

    # Load embedding.
    print('\nLoading embedding matrix...')
    embedding_path = folders.EMBEDDING_GLOVE_100_PATH
    embedding_matrix = load_embeddings.get_embedding(tokenizer, embedding_path, max_words)
    print('Done.')

    # Create model.
    print('\nCreating model...')
    category_k = [len(levels) for levels in category_levels]
    embedding_trainable = False
    word_rnn = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
    word_rnn_units = 128
    word_rnn_l2 = .01
    word_dense_units = 64
    word_dense_activation = 'linear'
    word_dense_l2 = .01
    book_rnn = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
    book_rnn_units = 128
    book_rnn_l2 = .01
    book_dense_units = 64
    book_dense_activation = 'linear'
    book_dense_l2 = .01
    book_dropout = .5
    label_mode = shared_parameters.LABEL_MODE_ORDINAL
    model = create_model(category_k, categories, n_paragraphs, n_tokens, embedding_matrix, embedding_trainable,
                         word_rnn, word_rnn_units, word_rnn_l2, word_dense_units, word_dense_activation, word_dense_l2,
                         book_rnn, book_rnn_units, book_rnn_l2, book_dense_units, book_dense_activation, book_dense_l2,
                         book_dropout, label_mode)
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
    test_size = .25  # b
    test_random_state = 1
    val_size = .1  # v
    val_random_state = 1
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
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=Y_val,
                        class_weight=category_class_weights)

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
    Y_preds = model.predict(X_test)
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
        fd.write('batch_size={:d}\n'.format(batch_size))
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
        fd.write('n_paragraphs={:d}\n'.format(n_paragraphs))
        fd.write('n_tokens={:d}\n'.format(n_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        fd.write('word_rnn={}\n'.format(word_rnn.__name__))
        fd.write('word_rnn_units={:d}\n'.format(word_rnn_units))
        fd.write('word_rnn_l2={}\n'.format(str(word_rnn_l2)))
        fd.write('word_dense_units={:d}\n'.format(word_dense_units))
        fd.write('word_dense_activation=\'{}\'\n'.format(word_dense_activation))
        fd.write('word_dense_l2={}\n'.format(str(word_dense_l2)))
        fd.write('book_rnn={}\n'.format(book_rnn.__name__))
        fd.write('book_rnn_units={:d}\n'.format(book_rnn_units))
        fd.write('book_rnn_l2={}\n'.format(str(book_rnn_l2)))
        fd.write('book_dense_units={:d}\n'.format(book_dense_units))
        fd.write('book_dense_activation=\'{}\'\n'.format(book_dense_activation))
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
