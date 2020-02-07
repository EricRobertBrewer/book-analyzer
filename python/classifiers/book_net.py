import argparse
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Bidirectional, Concatenate, Conv2D, CuDNNGRU, Dense, Dropout, Embedding, \
    Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, Input, MaxPool2D, Reshape, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers
# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from python import folders
from python.sites.bookcave import bookcave
from python.text import load_embeddings
from python.util import data_utils, evaluation, shared_parameters
from python.util import ordinal
from python.util.net.attention_with_context import AttentionWithContext
from python.util.net.batch_generators import SingleInstanceBatchGenerator


def main():
    parser = argparse.ArgumentParser(
        description='BookNet is a neural network classifier used to determine the maturity level of books.'
    )
    parser.add_argument('--source_mode',
                        default='paragraph',
                        choices=['paragraph', 'sentence'],
                        help='The source of text.`')
    parser.add_argument('--net_mode',
                        default='cnn',
                        choices=['rnn', 'cnn', 'rnncnn'],
                        help='The type of neural network.')
    parser.add_argument('--agg_mode',
                        default='maxavg',
                        choices=['max', 'avg', 'maxavg', 'rnn'],
                        help='The way the network will aggregate paragraphs or sentences.')
    parser.add_argument('--label_mode',
                        default=shared_parameters.LABEL_MODE_ORDINAL,
                        choices=[shared_parameters.LABEL_MODE_ORDINAL,
                                 shared_parameters.LABEL_MODE_CATEGORICAL,
                                 shared_parameters.LABEL_MODE_REGRESSION],
                        help='The way that labels will be interpreted.')
    parser.add_argument('--balance_mode',
                        choices=['reduce majority', 'sample union'],
                        help='Balance the data set. Optional.')
    parser.add_argument('--bag_mode',
                        action='store_true',
                        help='Option to add a 2-layer MLP using bag-of-words representations of texts. Optional.')
    parser.add_argument('--paragraph_dropout',
                        default=0.0,
                        type=float,
                        help='Probability to drop paragraphs during training. Default is 0.')
    parser.add_argument('--book_dropout',
                        default=0.5,
                        type=float,
                        help='Dropout probability before final classification layer. Default is 0.5.')
    parser.add_argument('--use_class_weights',
                        action='store_true',
                        help='Option to use a weighted loss function for imbalanced data.')
    parser.add_argument('--category_index',
                        default=-1,
                        type=int,
                        help='The index of the category that should be classified, or `-1` to classify all categories.')
    parser.add_argument('--steps_per_epoch',
                        default=0,
                        type=int,
                        help='Number of books to train on per epoch, or 0 to train on all books every epoch.')
    parser.add_argument('--epochs',
                        default=1,
                        type=int,
                        help='Epochs.')
    parser.add_argument('--note',
                        help='An optional note that will be appended to the names of generated files.')
    args = parser.parse_args()

    classifier_name = '{}_{}_{}_{}'.format(args.source_mode, args.net_mode, args.agg_mode, args.label_mode)

    start_time = int(time.time())
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
    else:
        stamp = start_time
    print('Time stamp: {:d}'.format(stamp))
    if args.note is not None:
        print('Note: {}'.format(args.note))
        base_fname = '{:d}_{}'.format(stamp, args.note)
    else:
        base_fname = format(stamp, 'd')

    # Load data.
    print('Retrieving texts...')
    if args.source_mode == 'paragraph':
        source = 'paragraph_tokens'
        min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
        max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    else:  # args.source_mode == 'sentence':
        source = 'sentence_tokens'
        min_len = shared_parameters.DATA_SENTENCE_MIN_LEN
        max_len = shared_parameters.DATA_SENTENCE_MAX_LEN
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
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

    # Reduce labels to the specified category, if needed.
    if args.category_index != -1:
        Y = np.array([Y[args.category_index]])
        categories = [categories[args.category_index]]
        category_levels = [category_levels[args.category_index]]

    # Balance data set, if needed.
    balance_majority_tolerance = 6
    balance_sample_seed = 1
    if args.balance_mode is not None:
        index_set = set()
        if args.balance_mode == 'reduce majority':
            majority_indices = data_utils.get_majority_indices(Y,
                                                               minlengths=[len(category_levels[j])
                                                                           for j in range(len(category_levels))],
                                                               tolerance=balance_majority_tolerance)
            index_set.update(set(np.arange(len(text_source_tokens))) - set(majority_indices))
        else:  # args.balance_mode == 'sample union':
            category_balanced_indices = [data_utils.get_balanced_indices_sample(y,
                                                                                minlength=len(category_levels[j]),
                                                                                seed=balance_sample_seed)
                                         for j, y in enumerate(Y)]
            for indices in category_balanced_indices:
                index_set.update(set(indices))
        text_source_tokens = [source_tokens for i, source_tokens in enumerate(text_source_tokens) if i in index_set]
        Y_T = Y.transpose()
        Y_T = np.array([instance for i, instance in enumerate(Y_T) if i in index_set])
        Y = Y_T.transpose()

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
    if args.source_mode == 'paragraph':
        n_tokens = shared_parameters.TEXT_N_PARAGRAPH_TOKENS
    else:  # args.source_mode == 'sentence':
        n_tokens = shared_parameters.TEXT_N_SENTENCE_TOKENS
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

    # Add a 2-layer MLP, if needed.
    if args.bag_mode:
        # Create vectorized representations of the book texts.
        print('Vectorizing text...')
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
        X_w = vectorizer.fit_transform(text_tokens).todense()
        print('Vectorized text with {:d} unique words.'.format(len(vectorizer.get_feature_names())))
    else:
        X_w = None

    # Create model.
    print('Creating model...')
    category_k = [len(levels) for levels in category_levels]
    embedding_trainable = False
    net_params = dict()
    if args.net_mode == 'rnn' or args.net_mode == 'rnncnn':
        net_params['rnn'] = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
        net_params['rnn_units'] = 128
        net_params['rnn_l2'] = .01
        net_params['rnn_dense_units'] = 64
        net_params['rnn_dense_activation'] = 'elu'
        net_params['rnn_dense_l2'] = .01
        net_params['rnn_agg'] = 'attention'
    if args.net_mode == 'cnn' or args.net_mode == 'rnncnn':
        net_params['cnn_filters'] = 16
        net_params['cnn_filter_sizes'] = [1, 2, 3, 4]
        net_params['cnn_activation'] = 'elu'
        net_params['cnn_l2'] = .01
    paragraph_dropout = args.paragraph_dropout
    agg_params = dict()
    if args.agg_mode == 'maxavg':
        pass
    elif args.agg_mode == 'max':
        pass
    elif args.agg_mode == 'avg':
        pass
    else:  # args.agg_mode == 'rnn':
        agg_params['rnn'] = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
        agg_params['rnn_units'] = 64
        agg_params['rnn_l2'] = .01
    normal_agg = False
    book_dense_units = 128
    book_dense_activation = tf.keras.layers.LeakyReLU(alpha=.1)
    book_dense_l2 = .01
    bag_params = dict()
    if args.bag_mode:
        bag_params['max_words'] = max_words
        bag_params['dense_1_units'] = 256
        bag_params['dense_1_activation'] = tf.keras.layers.LeakyReLU(alpha=.1)
        bag_params['dense_1_l2'] = .01
        bag_params['dense_2_units'] = 256
        bag_params['dense_2_activation'] = tf.keras.layers.LeakyReLU(alpha=.1)
        bag_params['dense_2_l2'] = .01
    book_dropout = args.book_dropout
    model = create_model(
        n_tokens, embedding_matrix, embedding_trainable,
        args.net_mode, net_params,
        paragraph_dropout, args.agg_mode, agg_params, normal_agg,
        book_dense_units, book_dense_activation, book_dense_l2,
        args.bag_mode, bag_params,
        book_dropout, category_k, categories, args.label_mode)
    lr = 2**-16
    optimizer = Adam(lr=lr)
    if args.label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        loss = 'binary_crossentropy'
        metric = 'binary_accuracy'
    elif args.label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        loss = 'categorical_crossentropy'
        metric = 'categorical_accuracy'
    else:  # args.label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        loss = 'mse'
        metric = 'accuracy'
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
    if X_w is not None:
        X_w_train, X_w_test = train_test_split(X_w, test_size=test_size, random_state=test_random_state)
        X_w_train, X_w_val = train_test_split(X_w_train, test_size=val_size, random_state=val_random_state)
    else:
        X_w_train, X_w_val, X_w_test = None, None, None
    Y_train = Y_train_T.transpose()  # (c, n * (1 - b) * (1 - v))
    Y_val = Y_val_T.transpose()  # (c, n * (1 - b) * v)
    Y_test = Y_test_T.transpose()  # (c, n * b)

    # Transform labels based on the label mode.
    Y_train = shared_parameters.transform_labels(Y_train, category_k, args.label_mode)
    Y_val = shared_parameters.transform_labels(Y_val, category_k, args.label_mode)

    # Calculate class weights.
    class_weight_f = 'square inverse'
    if args.use_class_weights:
        category_class_weights = shared_parameters.get_category_class_weights(Y_train, args.label_mode, f=class_weight_f)
    else:
        category_class_weights = None

    # Create generators.
    train_generator = SingleInstanceBatchGenerator(X_train, Y_train, X_w=X_w_train, shuffle=True)
    val_generator = SingleInstanceBatchGenerator(X_val, Y_val, X_w=X_w_val, shuffle=False)
    test_generator = SingleInstanceBatchGenerator(X_test, Y_test, X_w=X_w_test, shuffle=False)

    # Train.
    plateau_monitor = 'val_loss'
    plateau_factor = .5
    early_stopping_monitor = 'val_loss'
    early_stopping_min_delta = 2**-10
    if args.net_mode == 'rnn' or args.net_mode == 'rnncnn':
        plateau_patience = 3
        early_stopping_patience = 6
    else:  # args.net_mode == 'cnn':
        plateau_patience = 6
        early_stopping_patience = 12
    callbacks = [
        ReduceLROnPlateau(monitor=plateau_monitor, factor=plateau_factor, patience=plateau_patience),
        EarlyStopping(monitor=early_stopping_monitor, min_delta=early_stopping_min_delta, patience=early_stopping_patience)
    ]
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=args.steps_per_epoch if args.steps_per_epoch > 0 else None,
                                  epochs=args.epochs,
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
    if args.category_index != -1:
        Y_pred = [Y_pred]
    if args.label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        Y_pred = [ordinal.from_multi_hot_ordinal(y, threshold=.5) for y in Y_pred]
    elif args.label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        Y_pred = [np.argmax(y, axis=1) for y in Y_pred]
    else:  # args.label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        Y_pred = [np.maximum(0, np.minimum(k - 1, np.round(Y_pred[i] * k))) for i, k in enumerate(category_k)]
    print('Done.')

    # Save model.
    save_model = False
    if save_model:
        models_path = os.path.join(folders.MODELS_PATH, classifier_name)
        model_path = os.path.join(models_path, '{}.h5'.format(base_fname))
        print('Saving model to `{}`...'.format(model_path))
        if not os.path.exists(folders.MODELS_PATH):
            os.mkdir(folders.MODELS_PATH)
        if not os.path.exists(models_path):
            os.mkdir(models_path)
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
        if args.note is not None:
            fd.write('{}\n\n'.format(args.note))
        fd.write('PARAMETERS\n\n')
        fd.write('steps_per_epoch={:d}\n'.format(args.steps_per_epoch))
        fd.write('epochs={:d}\n'.format(args.epochs))
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
        fd.write('\nResampling\n')
        fd.write('balance_mode={}\n'.format(str(args.balance_mode)))
        if args.balance_mode == 'reduce majority':
            fd.write('balance_majority_tolerance={:d}\n'.format(balance_majority_tolerance))
        elif args.balance_mode == 'sample union':
            fd.write('balance_split_seed={:d}\n'.format(balance_sample_seed))
        fd.write('\nTokenization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('n_tokens={:d}\n'.format(n_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        if args.net_mode == 'rnn' or args.net_mode == 'rnncnn':
            fd.write('rnn={}\n'.format(net_params['rnn'].__name__))
            fd.write('rnn_units={:d}\n'.format(net_params['rnn_units']))
            fd.write('rnn_l2={}\n'.format(str(net_params['rnn_l2'])))
            fd.write('rnn_dense_units={:d}\n'.format(net_params['rnn_dense_units']))
            fd.write('rnn_dense_activation=\'{}\'\n'.format(net_params['rnn_dense_activation']))
            fd.write('rnn_dense_l2={}\n'.format(str(net_params['rnn_dense_l2'])))
            fd.write('rnn_agg={}\n'.format(net_params['rnn_agg']))
        if args.net_mode == 'cnn' or args.net_mode == 'rnncnn':
            fd.write('cnn_filters={:d}\n'.format(net_params['cnn_filters']))
            fd.write('cnn_filter_sizes={}\n'.format(str(net_params['cnn_filter_sizes'])))
            fd.write('cnn_activation=\'{}\'\n'.format(net_params['cnn_activation']))
            fd.write('cnn_l2={}\n'.format(str(net_params['cnn_l2'])))
        fd.write('paragraph_dropout={}\n'.format(str(paragraph_dropout)))
        if args.agg_mode == 'maxavg':
            pass
        elif args.agg_mode == 'max':
            pass
        elif args.agg_mode == 'avg':
            pass
        elif args.agg_mode == 'rnn':
            fd.write('agg_rnn={}\n'.format(agg_params['rnn'].__name__))
            fd.write('agg_rnn_units={:d}\n'.format(agg_params['rnn_units']))
            fd.write('agg_rnn_l2={}\n'.format(str(agg_params['rnn_l2'])))
        fd.write('normal_agg={}'.format(normal_agg))
        fd.write('book_dense_units={:d}\n'.format(book_dense_units))
        fd.write('book_dense_activation={} {}\n'.format(book_dense_activation.__class__.__name__,
                                                        book_dense_activation.__dict__))
        fd.write('book_dense_l2={}\n'.format(str(book_dense_l2)))
        fd.write('bag_mode={}\n'.format(args.bag_mode))
        if args.bag_mode:
            fd.write('dense_1_units={:d}\n'.format(bag_params['dense_1_units']))
            fd.write('dense_1_activation={} {}\n'.format(bag_params['dense_1_activation'].__class__.__name__,
                                                         bag_params['dense_1_activation'].__dict__))
            fd.write('dense_1_l2={}\n'.format(str(bag_params['dense_2_l2'])))
            fd.write('dense_2_units={:d}\n'.format(bag_params['dense_2_units']))
            fd.write('dense_2_activation={} {}\n'.format(bag_params['dense_2_activation'].__class__.__name__,
                                                         bag_params['dense_2_activation'].__dict__))
            fd.write('dense_2_l2={}\n'.format(str(bag_params['dense_1_l2'])))
        fd.write('book_dropout={}\n'.format(str(book_dropout)))
        model.summary(print_fn=lambda x: fd.write('{}\n'.format(x)))
        fd.write('\nTraining\n')
        fd.write('optimizer={}\n'.format(optimizer.__class__.__name__))
        fd.write('lr={}\n'.format(str(lr)))
        fd.write('loss=\'{}\'\n'.format(loss))
        fd.write('metric=\'{}\'\n'.format(metric))
        fd.write('test_size={}\n'.format(str(test_size)))
        fd.write('test_random_state={:d}\n'.format(test_random_state))
        fd.write('val_size={}\n'.format(str(val_size)))
        fd.write('val_random_state={:d}\n'.format(val_random_state))
        fd.write('use_class_weights={}\n'.format(args.use_class_weights))
        if args.use_class_weights:
            fd.write('class_weight_f={}\n'.format(class_weight_f))
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
        overall_last = args.category_index == -1 and return_overall
        evaluation.write_confusion_and_metrics(Y_test, Y_pred, fd, categories, overall_last=overall_last)

    if not os.path.exists(folders.PREDICTIONS_PATH):
        os.mkdir(folders.PREDICTIONS_PATH)
    predictions_path = os.path.join(folders.PREDICTIONS_PATH, classifier_name)
    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
    with open(os.path.join(predictions_path, '{}.txt'.format(base_fname)), 'w') as fd:
        evaluation.write_predictions(Y_test, Y_pred, fd, categories)

    print('Done.')


def identity(v):
    return v


def create_model(
        n_tokens, embedding_matrix, embedding_trainable,
        net_mode, net_params,
        paragraph_dropout, agg_mode, agg_params, normal_agg,
        book_dense_units, book_dense_activation, book_dense_l2,
        bag_mode, bag_params,
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
    else:  # net_mode == 'rnncnn':
        x_p = Concatenate()([
            create_source_rnn(net_params, x_p),
            create_source_cnn(n_tokens, d, net_params, x_p)
        ])
    source_encoder = Model(input_p, x_p)  # (m_p); constant per configuration

    # Consider signals among all sources of books.
    input_b = Input(shape=(None, n_tokens), dtype='float32')  # (P, T); P is not constant per instance!
    x_b = Dropout(paragraph_dropout, noise_shape=(1, tf.shape(input_b)[1], 1))(input_b)  # (P, T)
    x_b = TimeDistributed(source_encoder)(x_b)  # (P, m_p)
    if agg_mode == 'maxavg':
        x_b = Concatenate()([
            GlobalMaxPooling1D()(x_b),
            GlobalAveragePooling1D()(x_b)
        ])  # (2m_p)
    elif agg_mode == 'max':
        x_b = GlobalMaxPooling1D()(x_b)  # (m_p)
    elif agg_mode == 'avg':
        x_b = GlobalAveragePooling1D()(x_b)  # (m_p)
    else:  # agg_mode == 'rnn':
        agg_rnn = agg_params['rnn']
        agg_rnn_units = agg_params['rnn_units']
        agg_rnn_l2 = regularizers.l2(agg_params['rnn_l2']) if agg_params['rnn_l2'] is not None else None
        x_b = Bidirectional(agg_rnn(agg_rnn_units,
                                    kernel_regularizer=agg_rnn_l2,
                                    return_sequences=False),
                            merge_mode='concat')(x_b)  # (2h_b)
    if normal_agg:
        x_b = Activation('softmax')(x_b)
    if book_dense_l2 is not None:
        book_dense_l2 = regularizers.l2(book_dense_l2)
    x_b = Dense(book_dense_units,
                kernel_regularizer=book_dense_l2)(x_b)  # (c_b)
    x_b = book_dense_activation(x_b)  # (c_b)
    if bag_mode:
        input_w, x_w = create_bag_mlp(bag_params)
        x_b = Concatenate()([x_b, x_w])
        inputs = [input_b, input_w]
    else:
        inputs = [input_b]
    x_b = Dropout(book_dropout)(x_b)  # (c_b)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        outputs = [Dense(k - 1, activation='sigmoid', name=output_names[i])(x_b) for i, k in enumerate(output_k)]
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        outputs = [Dense(k, activation='softmax', name=output_names[i])(x_b) for i, k in enumerate(output_k)]
    else:  # label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        outputs = [Dense(1, activation='linear', name=output_names[i])(x_b) for i in range(len(output_k))]
    return Model(inputs, outputs)


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
    else:  # rnn_agg == 'max':
        x_p = GlobalMaxPooling1D()(x_p)  # (c_p)
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


def create_bag_mlp(bag_params):
    max_words = bag_params['max_words']
    dense_1_l2 = bag_params['dense_1_l2'] if bag_params['dense_1_l2'] is not None else None
    dense_1_units = bag_params['dense_1_units']
    dense_1_activation = bag_params['dense_1_activation']
    dense_2_l2 = bag_params['dense_2_l2'] if bag_params['dense_2_l2'] is not None else None
    dense_2_units = bag_params['dense_2_units']
    dense_2_activation = bag_params['dense_2_activation']
    input_w = Input(shape=(max_words,), dtype='float32')
    if dense_1_l2 is not None:
        dense_1_l2 = regularizers.l2(dense_1_l2)
    x_w = Dense(dense_1_units,
                kernel_regularizer=dense_1_l2)(input_w)
    x_w = dense_1_activation(x_w)  # (c_w)
    if dense_2_l2 is not None:
        dense_2_l2 = regularizers.l2(dense_2_l2)
    x_w = Dense(dense_2_units,
                kernel_regularizer=dense_2_l2)(x_w)
    x_w = dense_2_activation(x_w)  # (c_w)
    return input_w, x_w


if __name__ == '__main__':
    main()
