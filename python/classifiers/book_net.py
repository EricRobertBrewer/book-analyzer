from argparse import ArgumentParser, RawTextHelpFormatter
import os
import time

# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Bidirectional, Concatenate, Conv2D, CuDNNGRU, Dense, Dropout, Embedding, Flatten, \
    GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, Input, LeakyReLU, MaxPool2D, Reshape, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split

from python import folders
from python.sites.bookcave import bookcave
from python.text import load_embeddings
from python.util import evaluation, shared_parameters
from python.util import ordinal
from python.util.net.batch_generators import SingleInstanceBatchGenerator, TransformBalancedBatchGenerator


def main():
    parser = ArgumentParser(
        description='Classify the maturity level of a book by its text.',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('category_index',
                        type=int,
                        help='The category index.\n  {}'.format(
                            '\n  '.join(['{:d} {}'.format(j, bookcave.CATEGORY_NAMES[category])
                                         for j, category in enumerate(bookcave.CATEGORIES)]
                        )))
    parser.add_argument('--source_mode',
                        default='paragraph',
                        choices=['paragraph', 'sentence'],
                        help='The source of text. Default is `paragraph`.')
    parser.add_argument('--remove_stopwords',
                        action='store_true',
                        help='Remove stop-words from text. Default is False.')
    parser.add_argument('--agg_mode',
                        default='maxavg',
                        choices=['max', 'avg', 'maxavg', 'rnn'],
                        help='The way the network will aggregate paragraphs or sentences. Default is `maxavg`.')
    parser.add_argument('--label_mode',
                        default=shared_parameters.LABEL_MODE_ORDINAL,
                        choices=[shared_parameters.LABEL_MODE_ORDINAL,
                                 shared_parameters.LABEL_MODE_CATEGORICAL,
                                 shared_parameters.LABEL_MODE_REGRESSION],
                        help='The way that labels will be interpreted. '
                             'Default is `{}`.'.format(shared_parameters.LABEL_MODE_ORDINAL))
    parser.add_argument('--remove_classes',
                        type=str,
                        help='Remove classes altogether. Can be used when the minority class is severely tiny. '
                             'Like `<class1>[,<class2>,...]` as in `3` or `3,0`. Optional.')
    parser.add_argument('--class_weight_p',
                        default=2,
                        type=int,
                        help='Power with which to scale class weights. Default is 2.')
    parser.add_argument('--book_dense_units',
                        default='128',
                        help='The number of neurons in the final fully-connected layers, comma separated. '
                             'Default is `128`.')
    parser.add_argument('--book_dropout',
                        default=0.5,
                        type=float,
                        help='Dropout probability before final classification layer. Default is 0.5.')
    parser.add_argument('--plateau_patience',
                        default=16,
                        type=int,
                        help='Number of epochs to wait before dividing the learning rate by 2. Default is 16.')
    parser.add_argument('--early_stopping_patience',
                        default=32,
                        type=int,
                        help='Number of epochs to wait before dividing the learning rate by 2. Default is 32.')
    parser.add_argument('--epochs',
                        default=1,
                        type=int,
                        help='Epochs. Default is 1.')
    parser.add_argument('--note',
                        help='An optional note that will be appended to the names of generated files.')
    args = parser.parse_args()

    classifier_name = '{}_{}_{}'.format(args.source_mode, args.agg_mode, args.label_mode)

    start_time = int(time.time())
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
    else:
        stamp = start_time
    print('Time stamp: {:d}'.format(stamp))
    if args.note is not None:
        print('Note: {}'.format(args.note))
        base_fname = '{:d}_{}_{:d}'.format(stamp, args.note, args.category_index)
    else:
        base_fname = '{:d}_{:d}'.format(stamp, args.category_index)

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
                          remove_stopwords=args.remove_stopwords,
                          categories_mode=categories_mode,
                          return_overall=return_overall)
    text_source_tokens = list(zip(*inputs[source]))[0]
    print('Retrieved {:d} texts.'.format(len(text_source_tokens)))

    # Reduce labels to the specified category.
    y = Y[args.category_index]
    category = categories[args.category_index]
    levels = category_levels[args.category_index]
    k = len(levels)
    k_train = k

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

    # Convert to sequences.
    print('Converting texts to sequences...')
    if args.source_mode == 'paragraph':
        if not args.remove_stopwords:
            n_tokens = shared_parameters.TEXT_N_PARAGRAPH_TOKENS
        else:
            n_tokens = shared_parameters.TEXT_N_PARAGRAPH_TOKENS_NO_STOPWORDS
    else:  # args.source_mode == 'sentence':
        n_tokens = shared_parameters.TEXT_N_SENTENCE_TOKENS
    padding = shared_parameters.TEXT_PADDING
    truncating = shared_parameters.TEXT_TRUNCATING
    X = [np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in source_tokens]),
                                maxlen=n_tokens,
                                padding=padding,
                                truncating=truncating))
         for source_tokens in text_source_tokens]

    # Load embedding.
    print('Loading embedding matrix...')
    embedding_path = folders.EMBEDDING_GLOVE_300_PATH
    embedding_matrix = load_embeddings.load_embedding(tokenizer, embedding_path, max_words)

    # Create model.
    print('Creating model...')
    embedding_trainable = False
    cnn_filters = 16
    cnn_filter_sizes = [1, 2, 3, 4]
    cnn_activation = 'elu'
    cnn_l2 = .001
    agg_params = dict()
    if args.agg_mode == 'rnn':
        agg_params['rnn'] = CuDNNGRU if tf.test.is_gpu_available(cuda_only=True) else GRU
        agg_params['rnn_units'] = 64
        agg_params['rnn_l2'] = .001
    book_dense_units = [int(units) for units in args.book_dense_units.split(',')]
    book_dense_activation = LeakyReLU(alpha=.1)
    book_dense_l2 = .001
    book_dropout = args.book_dropout
    model = create_model(
        n_tokens, embedding_matrix, embedding_trainable,
        cnn_filters, cnn_filter_sizes, cnn_activation, cnn_l2,
        args.agg_mode, agg_params,
        book_dense_units, book_dense_activation, book_dense_l2,
        book_dropout, k, category, args.label_mode)
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

    # Split data set.
    print('Splitting data set...')
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    val_size = shared_parameters.EVAL_VAL_SIZE  # v
    val_random_state = shared_parameters.EVAL_VAL_RANDOM_STATE
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=test_random_state)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=val_size, random_state=val_random_state)
    y_val_transform = shared_parameters.transform_labels(y_val, k, args.label_mode)
    y_test_transform = shared_parameters.transform_labels(y_test, k, args.label_mode)

    # Remove classes from training set, if specified.
    if args.remove_classes is not None:
        remove_classes = sorted(list(map(int, args.remove_classes.strip().split(','))),
                                reverse=True)
        for class_ in remove_classes:
            y_train[y_train >= class_] -= 1
            k_train -= 1

    # Create generators.
    print('Creating data generators...')
    train_generator = TransformBalancedBatchGenerator(np.arange(len(X_train)).reshape((len(X_train), 1)),
                                                      y_train,
                                                      transform_X=transform_X,
                                                      transform_y=transform_y,
                                                      batch_size=1,
                                                      X_data=[np.array([x]) for x in X_train],
                                                      k=k,
                                                      label_mode=args.label_mode)
    val_generator = SingleInstanceBatchGenerator(X_val, y_val_transform, shuffle=False)
    test_generator = SingleInstanceBatchGenerator(X_test, y_test_transform, shuffle=False)

    # Get class weight.
    class_weight = shared_parameters.get_class_weight(k_train, args.label_mode, p=args.class_weight_p)

    # Train.
    print('Training for up to {:d} epoch{}...'.format(args.epochs, 's' if args.epochs != 1 else ''))
    plateau_monitor = 'val_loss'
    plateau_factor = .5
    early_stopping_monitor = 'val_loss'
    early_stopping_min_delta = 2**-10
    plateau_patience = args.plateau_patience
    early_stopping_patience = args.early_stopping_patience
    callbacks = [
        ReduceLROnPlateau(monitor=plateau_monitor,
                          factor=plateau_factor,
                          patience=plateau_patience),
        EarlyStopping(monitor=early_stopping_monitor,
                      min_delta=early_stopping_min_delta,
                      patience=early_stopping_patience)
    ]
    history = model.fit_generator(train_generator,
                                  epochs=args.epochs,
                                  verbose=0,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  class_weight=class_weight)
    epochs_complete = len(history.history.get('val_loss'))

    # Save the history to visualize loss over time.
    print('Saving training history...')
    history_path = folders.ensure(os.path.join(folders.HISTORY_PATH, classifier_name))
    with open(os.path.join(history_path, '{}.txt'.format(base_fname)), 'w') as fd:
        for key in history.history.keys():
            values = history.history.get(key)
            fd.write('{} {}\n'.format(key, ' '.join(str(value) for value in values)))

    # Predict test instances.
    print('Predicting test instances...')
    y_pred_transform = model.predict_generator(test_generator)
    if args.label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        y_pred = ordinal.from_multi_hot_ordinal(y_pred_transform, threshold=.5)
    elif args.label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        y_pred = np.argmax(y_pred_transform, axis=1)
    else:  # args.label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        y_pred = np.maximum(0, np.minimum(k - 1, np.round(y_pred_transform * k)))

    # Save model.
    save_model = True
    if save_model:
        models_path = folders.ensure(os.path.join(folders.MODELS_PATH, classifier_name))
        model_path = os.path.join(models_path, '{}.h5'.format(base_fname))
        print('Saving model to `{}`...'.format(model_path))
        model.save(model_path)
    else:
        model_path = None

    # Calculate elapsed time.
    end_time = int(time.time())
    elapsed_s = end_time - start_time
    elapsed_m, elapsed_s = elapsed_s // 60, elapsed_s % 60
    elapsed_h, elapsed_m = elapsed_m // 60, elapsed_m % 60

    # Write results.
    print('Writing results...')
    logs_path = folders.ensure(os.path.join(folders.LOGS_PATH, classifier_name))
    with open(os.path.join(logs_path, '{}.txt'.format(base_fname)), 'w') as fd:
        if args.note is not None:
            fd.write('{}\n\n'.format(args.note))
        fd.write('PARAMETERS\n\n')
        fd.write('category_index={:d}\n'.format(args.category_index))
        fd.write('epochs={:d}\n'.format(args.epochs))
        fd.write('\nHYPERPARAMETERS\n')
        fd.write('\nText\n')
        fd.write('subset_ratio={}\n'.format(str(subset_ratio)))
        fd.write('subset_seed={}\n'.format(str(subset_seed)))
        fd.write('min_len={:d}\n'.format(min_len))
        fd.write('max_len={:d}\n'.format(max_len))
        fd.write('min_tokens={:d}\n'.format(min_tokens))
        fd.write('remove_stopwords={}\n'.format(args.remove_stopwords))
        fd.write('\nLabels\n')
        fd.write('categories_mode=\'{}\'\n'.format(categories_mode))
        fd.write('return_overall={}\n'.format(return_overall))
        if args.remove_classes is not None:
            fd.write('remove_classes={}\n'.format(args.remove_classes))
        else:
            fd.write('No classes removed.\n')
        fd.write('class_weight_p={:d}\n'.format(args.class_weight_p))
        fd.write('\nTokenization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('n_tokens={:d}\n'.format(n_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nWord Embedding\n')
        fd.write('embedding_path=\'{}\'\n'.format(embedding_path))
        fd.write('embedding_trainable={}\n'.format(embedding_trainable))
        fd.write('\nModel\n')
        fd.write('cnn_filters={:d}\n'.format(cnn_filters))
        fd.write('cnn_filter_sizes={}\n'.format(str(cnn_filter_sizes)))
        fd.write('cnn_activation=\'{}\'\n'.format(cnn_activation))
        fd.write('cnn_l2={}\n'.format(str(cnn_l2)))
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
        fd.write('book_dense_units={}\n'.format(args.book_dense_units))
        fd.write('book_dense_activation={} {}\n'.format(book_dense_activation.__class__.__name__,
                                                        book_dense_activation.__dict__))
        fd.write('book_dense_l2={}\n'.format(str(book_dense_l2)))
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
        fd.write('Epochs completed: {:d}\n'.format(epochs_complete))
        fd.write('Time elapsed: {:d}h {:d}m {:d}s\n\n'.format(elapsed_h, elapsed_m, elapsed_s))
        evaluation.write_confusion_and_metrics(y_test, y_pred, fd, category)

    # Write predictions.
    print('Writing predictions...')
    predictions_path = folders.ensure(os.path.join(folders.PREDICTIONS_PATH, classifier_name))
    with open(os.path.join(predictions_path, '{}.txt'.format(base_fname)), 'w') as fd:
        evaluation.write_predictions(y_test, y_pred, fd, category)

    print('Done.')


def identity(v):
    return v


def create_model(
        n_tokens, embedding_matrix, embedding_trainable,
        cnn_filters, cnn_filter_sizes, cnn_activation, cnn_l2,
        agg_mode, agg_params,
        book_dense_units, book_dense_activation, book_dense_l2,
        book_dropout, k, output_name, label_mode):
    # Source encoder.
    input_p = Input(shape=(n_tokens,), dtype='float32')  # (T)
    max_words, d = embedding_matrix.shape
    x_p = Embedding(max_words,
                    d,
                    weights=[embedding_matrix],
                    trainable=embedding_trainable)(input_p)  # (T, d)
    if cnn_l2 is not None:
        cnn_l2 = regularizers.l2(cnn_l2)
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
    else:  # agg_mode == 'rnn':
        agg_rnn = agg_params['rnn']
        agg_rnn_units = agg_params['rnn_units']
        agg_rnn_l2 = regularizers.l2(agg_params['rnn_l2']) if agg_params['rnn_l2'] is not None else None
        x_b = Bidirectional(agg_rnn(agg_rnn_units,
                                    kernel_regularizer=agg_rnn_l2,
                                    return_sequences=False),
                            merge_mode='concat')(x_b)  # (2h_b)
    if book_dense_l2 is not None:
        book_dense_l2 = regularizers.l2(book_dense_l2)
    for units in book_dense_units:
        x_b = Dense(units,
                    kernel_regularizer=book_dense_l2)(x_b)  # (c_b)
        x_b = book_dense_activation(x_b)  # (c_b)
    x_b = Dropout(book_dropout)(x_b)  # (c_b)
    if label_mode == shared_parameters.LABEL_MODE_ORDINAL:
        output = Dense(k - 1, activation='sigmoid', name=output_name)(x_b)
    elif label_mode == shared_parameters.LABEL_MODE_CATEGORICAL:
        output = Dense(k, activation='softmax', name=output_name)(x_b)
    else:  # label_mode == shared_parameters.LABEL_MODE_REGRESSION:
        output = Dense(1, activation='linear', name=output_name)(x_b)
    return Model(input_b, output)


def transform_X(X, **kwargs):
    indices = np.array([x[0] for x in X])
    X_data = kwargs['X_data']
    return [X_data[i] for i in indices]


def transform_y(y, **kwargs):
    k = kwargs['k']
    label_mode = kwargs['label_mode']
    return shared_parameters.transform_labels(y, k, label_mode)


if __name__ == '__main__':
    main()
