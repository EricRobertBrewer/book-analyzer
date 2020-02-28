import os
import time

import numpy as np
# Weird "`GLIBCXX_...' not found" error occurs on rc.byu.edu if `sklearn` is imported before `tensorflow`.
import tensorflow as tf
from sklearn.model_selection import train_test_split

from python.util import evaluation, shared_parameters
from python import folders
from python.sites.bookcave import bookcave


def main():
    script_name = os.path.basename(__file__)
    classifier_name = script_name[:script_name.rindex('.')]

    stamp = int(time.time())
    base_fname = format(stamp, 'd')

    # Load data.
    print('Retrieving labels...')
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    return_overall = shared_parameters.DATA_RETURN_OVERALL
    _, Y, categories, category_levels = \
        bookcave.get_data({'paragraph_tokens'},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          categories_mode=categories_mode,
                          return_overall=return_overall)
    print('Retrieved {:d} labels.'.format(Y.shape[1]))

    # Split data set.
    test_size = shared_parameters.EVAL_TEST_SIZE  # b
    test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
    Y_T = Y.transpose()  # (n, c)
    Y_train_T, Y_test_T = train_test_split(Y_T, test_size=test_size, random_state=test_random_state)
    Y_train = Y_train_T.transpose()  # (c, n * (1 - b))
    Y_test = Y_test_T.transpose()  # (c, n * b)

    # Predict the most common class seen in the training data for each category.
    Y_pred = [[np.argmax(np.bincount(Y_train[j], minlength=len(category_levels[j])))]*Y_test.shape[1]
              for j in range(len(categories))]

    print('Writing results...')

    logs_path = folders.ensure(os.path.join(folders.LOGS_PATH, classifier_name))
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
        fd.write('\nTraining\n')
        fd.write('test_size={:.2f}\n'.format(test_size))
        fd.write('test_random_state={:d}\n'.format(test_random_state))
        fd.write('\nRESULTS\n\n')
        fd.write('Data size: {:d}\n'.format(Y.shape[1]))
        fd.write('Train size: {:d}\n'.format(Y_train.shape[1]))
        fd.write('Test size: {:d}\n'.format(Y_test.shape[1]))
        fd.write('\n')
        evaluation.write_confusion_and_metrics(Y_test, Y_pred, fd, categories, overall_last=return_overall)

    predictions_path = folders.ensure(os.path.join(folders.PREDICTIONS_PATH, classifier_name))
    with open(os.path.join(predictions_path, '{}.txt'.format(base_fname)), 'w') as fd:
        evaluation.write_predictions(Y_test, Y_pred, fd, categories)

    print('Done.')


if __name__ == '__main__':
    main()
