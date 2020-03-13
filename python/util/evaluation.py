import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score


METRIC_NAMES = [
    'accuracy',
    'precision_macro',
    'recall_macro',
    'f1_macro',
    'precision_weighted',
    'recall_weighted',
    'f1_weighted',
    'mean_absolute_error',
    'mean_squared_error'
]
METRIC_ABBREVIATIONS = [
    'ACC',
    'P_MAC',
    'R_MAC',
    'F_MAC',
    'P_WTD',
    'R_WTD',
    'F_WTD',
    'MAE',
    'MSE'
]


def get_confusion_and_metrics(y_true, y_pred, ignore_warnings=True):
    confusion = confusion_matrix(y_true, y_pred)
    if ignore_warnings:
        # https://stackoverflow.com/a/47749756/1559071
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
    metrics = [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='macro'),
        precision_score(y_true, y_pred, average='weighted'),
        recall_score(y_true, y_pred, average='weighted'),
        f1_score(y_true, y_pred, average='weighted'),
        mean_absolute_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred)
    ]
    if ignore_warnings:
        warnings.filterwarnings('default', category=UndefinedMetricWarning)
    return confusion, metrics


def write_confusion_and_metrics(y_true, y_pred, fd, category):
    # Calculate statistics for predictions.
    confusion, metrics = get_confusion_and_metrics(y_true, y_pred)

    category_width = len(category)

    # Metric abbreviations.
    fd.write('{:>{w}}'.format('Metric', w=category_width))
    for abbreviation in METRIC_ABBREVIATIONS:
        fd.write(' | {:^7}'.format(abbreviation))
    fd.write(' |\n')

    # Horizontal line.
    fd.write('{:>{w}}'.format('', w=category_width))
    for _ in range(len(metrics)):
        fd.write('-+-{}'.format('-' * 7))
    fd.write('-+\n')

    # Metrics per category.
    fd.write('{:>{w}}'.format(category, w=category_width))
    for value in metrics:
        fd.write(' | {:.4f} '.format(value))
    fd.write(' |\n')

    # Confusion matrices.
    fd.write('\n`{}`\n'.format(category))
    fd.write(np.array2string(confusion))
    fd.write('\n')


def write_predictions(y_true, y_pred, fd, category):
    fd.write('{}_true\t{}_pred\n'.format(category, category))
    for i in range(len(y_true)):
        fd.write('{:d}\t{:d}\n'.format(y_true[i], y_pred[i]))
