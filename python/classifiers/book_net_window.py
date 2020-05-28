from argparse import ArgumentParser, RawTextHelpFormatter
import os
import time

# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

from python import folders
from python.sites.bookcave import bookcave
from python.text import tokenizers
from python.util import evaluation, shared_parameters
from python.util import ordinal


def main():
    parser = ArgumentParser(
        description='Classify the maturity level of a book by its paragraphs.',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('model_file_name',
                        help='The file name of the model to load.')
    parser.add_argument('window',
                        type=int,
                        help='The paragraph window size.')
    args = parser.parse_args()
    source_mode = 'paragraph'
    remove_stopwords = False

    start_time = int(time.time())
    classifier_name = 'paragraph_max_ordinal'
    model_file_base_name = args.model_file_name[:args.model_file_name.rindex('.')]
    category_index = int(model_file_base_name[-1])
    base_fname = '{}_{:d}w'.format(model_file_base_name, args.window)

    # Load data.
    print('Retrieving texts...')
    source = 'paragraph_tokens'
    min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
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
                          remove_stopwords=remove_stopwords,
                          categories_mode=categories_mode,
                          return_overall=return_overall)
    text_source_tokens = list(zip(*inputs[source]))[0]
    print('Retrieved {:d} texts.'.format(len(text_source_tokens)))

    # Reduce labels to the specified category.
    y = Y[category_index]
    category = categories[category_index]

    # Tokenize.
    print('Tokenizing...')
    max_words = shared_parameters.TEXT_MAX_WORDS
    split = '\t'
    tokenizer = tokenizers.get_tokenizer_or_fit(max_words,
                                                source_mode,
                                                remove_stopwords,
                                                text_source_tokens=text_source_tokens)

    # Convert to sequences.
    print('Converting texts to sequences...')
    n_tokens = shared_parameters.TEXT_N_PARAGRAPH_TOKENS
    padding = shared_parameters.TEXT_PADDING
    truncating = shared_parameters.TEXT_TRUNCATING
    X = [np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in source_tokens]),
                                maxlen=n_tokens,
                                padding=padding,
                                truncating=truncating))
         for source_tokens in text_source_tokens]

    # Load model.
    print('Loading model...')
    model_path = os.path.join(folders.MODELS_PATH, classifier_name, args.model_file_name)
    model = tf.keras.models.load_model(model_path)

    # Split data set.
    print('Splitting data set...')
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    _, X_test, _, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=test_random_state)

    # Predict instances.
    print('Predicting labels...')
    y_pred = np.zeros((len(X_test),), dtype=np.int32)
    for i, x in enumerate(X_test):
        P = np.zeros((len(x) - args.window + 1, args.window, *x.shape[1:]))
        for w in range(len(P)):
            P[w] = x[w:w + args.window]
        q_pred_transform = model.predict(P)
        q_pred = ordinal.from_multi_hot_ordinal(q_pred_transform, threshold=.5)
        label_pred = max(q_pred)
        y_pred[i] = label_pred

    # Calculate elapsed time.
    end_time = int(time.time())
    elapsed_s = end_time - start_time
    elapsed_m, elapsed_s = elapsed_s // 60, elapsed_s % 60
    elapsed_h, elapsed_m = elapsed_m // 60, elapsed_m % 60

    # Write results.
    print('Writing results...')
    logs_path = folders.ensure(os.path.join(folders.LOGS_PATH, classifier_name))
    with open(os.path.join(logs_path, '{}.txt'.format(base_fname)), 'w') as fd:
        fd.write('PARAMETERS\n\n')
        fd.write('model_file_name={}\n'.format(args.model_file_name))
        fd.write('window={:d}\n'.format(args.window))
        fd.write('\nHYPERPARAMETERS\n')
        fd.write('\nText\n')
        fd.write('subset_ratio={}\n'.format(str(subset_ratio)))
        fd.write('subset_seed={}\n'.format(str(subset_seed)))
        fd.write('min_len={:d}\n'.format(min_len))
        fd.write('max_len={:d}\n'.format(max_len))
        fd.write('min_tokens={:d}\n'.format(min_tokens))
        fd.write('remove_stopwords={}\n'.format(remove_stopwords))
        fd.write('\nLabels\n')
        fd.write('categories_mode=\'{}\'\n'.format(categories_mode))
        fd.write('return_overall={}\n'.format(return_overall))
        if args.remove_classes is not None:
            fd.write('remove_classes={}\n'.format(args.remove_classes))
        else:
            fd.write('No classes removed.\n')
        fd.write('\nTokenization\n')
        fd.write('max_words={:d}\n'.format(max_words))
        fd.write('n_tokens={:d}\n'.format(n_tokens))
        fd.write('padding=\'{}\'\n'.format(padding))
        fd.write('truncating=\'{}\'\n'.format(truncating))
        fd.write('\nRESULTS\n\n')
        fd.write('Data size: {:d}\n'.format(len(X)))
        fd.write('Test size: {:d}\n'.format(len(X_test)))
        fd.write('Time elapsed: {:d}h {:d}m {:d}s\n\n'.format(elapsed_h, elapsed_m, elapsed_s))
        evaluation.write_confusion_and_metrics(y_test, y_pred, fd, category)

    # Write predictions.
    print('Writing predictions...')
    predictions_path = folders.ensure(os.path.join(folders.PREDICTIONS_PATH, classifier_name))
    with open(os.path.join(predictions_path, '{}.txt'.format(base_fname)), 'w') as fd:
        evaluation.write_predictions(y_test, y_pred, fd, category)

    print('Done.')


if __name__ == '__main__':
    main()
