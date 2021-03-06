import os
import sys
import time

import numpy as np
# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Concatenate, CuDNNGRU, Dense, Dropout, Embedding, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers

from python.util import evaluation, shared_parameters
from python.util import ordinal
from python.util.net.attention_with_context import AttentionWithContext
from python.util.net.batch_generators import SingleInstanceBatchGenerator
from python import folders
from python.sites.bookcave import bookcave
from python.text import load_embeddings


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
        base_fname = '{:d}_{}'.format(stamp, note)
    else:
        base_fname = format(stamp, 'd')

    # Load data.
    print('Loading data...')
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_SENTENCE_MIN_LEN
    max_len = shared_parameters.DATA_SENTENCE_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'sentence_tokens'},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          categories_mode=categories_mode)
    text_sentence_tokens, text_section_ids, text_paragraph_ids = zip(*inputs['sentence_tokens'])
    print('Retrieved {:d} texts.'.format(len(text_sentence_tokens)))

    # Tokenize.
    print('Tokenizing...')
    max_words = shared_parameters.TEXT_MAX_WORDS
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_sentences = []
    for sentence_tokens in text_sentence_tokens:
        for tokens in sentence_tokens:
            all_sentences.append(split.join(tokens))
    tokenizer.fit_on_texts(all_sentences)
    print('Done.')

    # Convert to sequences.
    print('Converting texts to sequences...')
    n_sentences = shared_parameters.TEXT_N_SENTENCES
    n_sentence_tokens = shared_parameters.TEXT_N_SENTENCE_TOKENS
    padding = shared_parameters.TEXT_PADDING
    truncating = shared_parameters.TEXT_TRUNCATING
    text_sentence_sequences = [pad_sequences(tokenizer.texts_to_sequences([split.join(tokens)
                                                                           for tokens in sentence_tokens]),
                                             maxlen=n_sentence_tokens,
                                             padding=padding,
                                             truncating=truncating)
                               for sentence_tokens in text_sentence_tokens]
    X = []
    for text_i, sentence_sequences in enumerate(text_sentence_sequences):
        section_ids = text_section_ids[text_i]
        paragraph_ids = text_paragraph_ids[text_i]
        n_paragraphs = len(np.unique(list(zip(text_section_ids[text_i], text_paragraph_ids[text_i])), axis=0))
        x = np.zeros((n_paragraphs, n_sentences, n_sentence_tokens))  # [paragraph_i][sentence_i][token_i]
        paragraph_i = 0
        sentence_i = 0
        last_section_paragraph_id = None
        for sequence_i, sentence_sequence in enumerate(sentence_sequences):
            section_paragraph_id = (section_ids[sequence_i], paragraph_ids[sequence_i])
            if last_section_paragraph_id is not None and section_paragraph_id != last_section_paragraph_id:
                paragraph_i += 1
                sentence_i = 0
            if sentence_i < n_sentences:
                x[paragraph_i, sentence_i] = sentence_sequence
            sentence_i += 1
            last_section_paragraph_id = section_paragraph_id
        X.append(x)
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
    sent_rnn = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
    sent_rnn_units = 128
    sent_rnn_l2 = .01
    sent_dense_units = 64
    sent_dense_activation = 'elu'
    sent_dense_l2 = .01
    para_rnn = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
    para_rnn_units = 128
    para_rnn_l2 = .01
    para_dense_units = 64
    para_dense_activation = 'elu'
    para_dense_l2 = .01
    book_dense_units = 128
    book_dense_activation = tf.keras.layers.LeakyReLU(alpha=.1)
    book_dense_l2 = .01
    book_dropout = .5
    label_mode = shared_parameters.LABEL_MODE_ORDINAL
    sentence_encoder, paragraph_encoder, model = create_model(
        n_sentences, n_sentence_tokens, embedding_matrix, embedding_trainable,
        sent_rnn, sent_rnn_units, sent_rnn_l2, sent_dense_units, sent_dense_activation, sent_dense_l2,
        para_rnn, para_rnn_units, para_rnn_l2, para_dense_units, para_dense_activation, para_dense_l2,
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

    # Transform labels based on the label mode.
    Y_train = shared_parameters.transform_labels(Y_train, category_k, label_mode)
    Y_val = shared_parameters.transform_labels(Y_val, category_k, label_mode)

    # Calculate class weights.
    use_class_weights = True
    class_weight_f = 'inverse'
    if use_class_weights:
        category_class_weights = shared_parameters.get_category_class_weights(Y_train, label_mode, f=class_weight_f)
    else:
        category_class_weights = None

    # Create generators.
    shuffle = True
    train_generator = SingleInstanceBatchGenerator(X_train, Y_train, shuffle=shuffle)
    val_generator = SingleInstanceBatchGenerator(X_val, Y_val, shuffle=False)
    test_generator = SingleInstanceBatchGenerator(X_test, Y_test, shuffle=False)

    # Train.
    plateau_monitor = 'val_loss'
    plateau_factor = .5
    plateau_patience = 3
    early_stopping_monitor = 'val_loss'
    early_stopping_min_delta = 2**-10
    early_stopping_patience = 6
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
            fd.write('Note: {}\n\n'.format(note))
        fd.write('PARAMETERS\n\n')
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
        fd.write('n_sentences={:d}\n'.format(n_sentences))
        fd.write('n_sentence_tokens={:d}\n'.format(n_sentence_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        fd.write('sent_rnn={}\n'.format(sent_rnn.__name__))
        fd.write('sent_rnn_units={:d}\n'.format(sent_rnn_units))
        fd.write('sent_rnn_l2={}\n'.format(str(sent_rnn_l2)))
        fd.write('sent_dense_units={:d}\n'.format(sent_dense_units))
        fd.write('sent_dense_activation=\'{}\'\n'.format(sent_dense_activation))
        fd.write('sent_dense_l2={}\n'.format(str(sent_dense_l2)))
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


def create_model(
        n_sentences, n_sentence_tokens, embedding_matrix, embedding_trainable,
        sent_rnn, sent_rnn_units, sent_rnn_l2, sent_dense_units, sent_dense_activation, sent_dense_l2,
        para_rnn, para_rnn_units, para_rnn_l2, para_dense_units, para_dense_activation, para_dense_l2,
        book_dense_units, book_dense_activation, book_dense_l2,
        book_dropout, output_k, output_names, label_mode):
    # Sentence encoder.
    input_s = Input(shape=(n_sentence_tokens,), dtype='float32')  # (t)
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

    # Paragraph encoder.
    input_p = Input(shape=(n_sentences, n_sentence_tokens), dtype='float32')  # (s, t)
    x_p = TimeDistributed(sentence_encoder)(input_p)  # (s, c_s)
    if para_rnn_l2 is not None:
        para_rnn_l2 = regularizers.l2(para_rnn_l2)
    x_p = Bidirectional(para_rnn(para_rnn_units,
                                 kernel_regularizer=para_rnn_l2,
                                 return_sequences=True))(x_p)  # (2s, h_p)
    if para_dense_l2 is not None:
        para_dense_l2 = regularizers.l2(para_dense_l2)
    x_p = TimeDistributed(Dense(para_dense_units,
                                activation=para_dense_activation,
                                kernel_regularizer=para_dense_l2))(x_p)  # (2s, c_p)
    x_p = AttentionWithContext()(x_p)  # (c_p)
    paragraph_encoder = Model(input_p, x_p)

    # Consider maximum and average signals among all paragraphs of books.
    input_b = Input(shape=(None, n_sentences, n_sentence_tokens), dtype='float32')  # (p, s, t); p is not constant!
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
    return sentence_encoder, paragraph_encoder, model


if __name__ == '__main__':
    main(sys.argv[1:])
