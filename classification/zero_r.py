import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from classification import evaluation, shared_parameters
import folders
from sites.bookcave import bookcave, bookcave_ids
from text import generate_data


def main():
    script_name = os.path.basename(__file__)
    classifier_name = script_name[:script_name.rindex('.')]

    start_time = int(time.time())
    if 'SLURM_JOB_ID' in os.environ:
        stamp = int(os.environ['SLURM_JOB_ID'])
    else:
        stamp = start_time
    print('Time stamp: {:d}'.format(stamp))
    base_fname = format(stamp, 'd')

    # Load data.
    print('Retrieving texts...')
    ids_fname = bookcave_ids.get_ids_fname()
    categories_mode = 'soft'
    Y = generate_data.load_Y(categories_mode, ids_fname)
    categories = bookcave.CATEGORIES
    category_levels = bookcave.CATEGORY_LEVELS[categories_mode]
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
    if not os.path.exists(folders.LOGS_PATH):
        os.mkdir(folders.LOGS_PATH)
    logs_path = os.path.join(folders.LOGS_PATH, classifier_name)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    with open(os.path.join(logs_path, '{}.txt'.format(base_fname)), 'w') as fd:
        fd.write('HYPERPARAMETERS\n')
        fd.write('\nText\n')
        fd.write('ids_fname={}\n'.format(bookcave_ids.get_ids_fname()))
        fd.write('\nLabels\n')
        fd.write('categories_mode=\'{}\'\n'.format(categories_mode))
        fd.write('\nTraining\n')
        fd.write('test_size={:.2f}\n'.format(test_size))
        fd.write('test_random_state={:d}\n'.format(test_random_state))
        fd.write('\nRESULTS\n\n')
        fd.write('Data size: {:d}\n'.format(Y.shape[1]))
        fd.write('Train size: {:d}\n'.format(Y_train.shape[1]))
        fd.write('Test size: {:d}\n'.format(Y_test.shape[1]))
        fd.write('\n')
        evaluation.write_confusion_and_metrics(Y_test, Y_pred, fd, categories)
    print('Done.')


if __name__ == '__main__':
    main()
