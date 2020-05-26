import argparse
import os
import pickle

# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from python.classifiers import baselines as base
from python import folders
from python.sites.bookcave import bookcave
from python.text import tokenizers
from python.util import evaluation, shared_parameters


def main():
    parser = argparse.ArgumentParser(
        description='Run fitted baseline classifiers over paragraphs.'
    )
    parser.add_argument('model_name',
                        help='Name of the algorithm.')
    parser.add_argument('stamp',
                        help='Time stamp of saved models.')
    parser.add_argument('window',
                        type=int,
                        help='The paragraph window size.')
    args = parser.parse_args()

    max_words = shared_parameters.TEXT_MAX_WORDS

    # Load data.
    print('Retrieving texts...')
    source = 'paragraph_tokens'
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    remove_stopwords = False
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

    # Create vectorized representations of the book texts.
    print('Loading vectorizer...')
    vectorizer = tokenizers.get_vectorizer_or_fit(max_words, remove_stopwords)

    # Split data set.
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    Y_T = Y.transpose()  # (n, c)
    _, text_source_tokens_test, _, Y_test_T = \
      train_test_split(text_source_tokens, Y_T, test_size=test_size, random_state=test_random_state)
    Y_test = Y_test_T.transpose()  # (c, n * b)

    # Separate books into paragraph windows.
    print('Creating paragraph windows...')
    text_P = list()
    for source_tokens in text_source_tokens_test:
        token_windows = list()
        for i in range(len(source_tokens) - args.window + 1):
            token_window = list()
            for tokens in source_tokens[i:i + args.window]:
                token_window.extend(tokens)
            token_windows.append(token_window)
        P = vectorizer.transform(token_windows)
        text_P.append(P)

    # Load classifiers.
    print('Loading classifiers...')
    category_classifiers = list()
    for j, levels in enumerate(category_levels):
        classifiers = list()
        category_part = '{}_{:d}'.format(args.stamp, j)
        for k in range(len(levels) - 1):
            path = os.path.join(folders.MODELS_PATH, args.model_name, category_part, 'model{:d}.pickle'.format(k))
            with open(path, 'rb') as fd:
                model = pickle.load(fd)
            classifiers.append(model)
        category_classifiers.append(classifiers)

    # Infer from paragraphs.
    for j, y_test in enumerate(Y_test):
        category = categories[j]
        print('Predicting category `{}`...'.format(category))
        k = len(category_levels[j])
        models = category_classifiers[j]
        y_pred = np.zeros((len(y_test),), dtype=np.int32)
        for i in range(len(y_test)):
            P = text_P[i]
            q_pred = base.predict_ordinal(models, P, k)
            label_pred = max(q_pred)
            y_pred[i] = label_pred

        base_fname = '{}_{:d}_{:d}w'.format(args.stamp, j, args.window)
        logs_path = folders.ensure(os.path.join(folders.LOGS_PATH, args.model_name))
        with open(os.path.join(logs_path, '{}.txt'.format(base_fname)), 'w') as fd:
            fd.write('HYPERPARAMETERS\n')
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
            fd.write('vectorizer={}\n'.format(vectorizer.__class__.__name__))
            fd.write('\nTraining\n')
            fd.write('test_size={}\n'.format(str(test_size)))
            fd.write('test_random_state={:d}\n'.format(test_random_state))
            fd.write('\nRESULTS\n\n')
            fd.write('Data size: {:d}\n'.format(len(text_source_tokens)))
            fd.write('Test size: {:d}\n\n'.format(len(text_source_tokens_test)))
            evaluation.write_confusion_and_metrics(y_test, y_pred, fd, category)

        predictions_path = folders.ensure(os.path.join(folders.PREDICTIONS_PATH, args.model_name))
        with open(os.path.join(predictions_path, '{}.txt'.format(base_fname)), 'w') as fd:
            evaluation.write_predictions(y_test, y_pred, fd, category)

    print('Done.')


if __name__ == '__main__':
    main()
