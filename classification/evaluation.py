import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score


METRIC_NAMES = [
    'accuracy',
    'precision_macro',
    'recall_macro',
    'f1_macro',
    'precision_weighted',
    'recall_weighted',
    'f1_weighted',
    'MSE'
]
METRIC_ABBREVIATIONS = [
    'ACC',
    'P_MAC',
    'R_MAC',
    'F_MAC',
    'P_WTD',
    'R_WTD',
    'F_WTD',
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
        mean_squared_error(y_true, y_pred)
    ]
    if ignore_warnings:
        warnings.filterwarnings('default', category=UndefinedMetricWarning)
    return confusion, metrics


def write_confusion_and_metrics(Y_true, Y_pred, fd, categories, overall_last=True):
    # Calculate statistics for predictions.
    category_confusion, category_metrics = zip(*[get_confusion_and_metrics(Y_true[j], Y_pred[j])
                                                 for j in range(len(Y_true))])
    if overall_last:
        n_average = len(category_metrics) - 1
    else:
        n_average = len(category_metrics)
    averages = [sum([metrics[metric_i] for metrics in category_metrics[:n_average]])/n_average
                for metric_i in range(len(category_metrics[0]))]

    category_width = max(7, max([len(category) for category in categories]))

    # Metric abbreviations.
    fd.write('{:>{w}}'.format('Metric', w=category_width))
    for abbreviation in METRIC_ABBREVIATIONS:
        fd.write(' | {:^7}'.format(abbreviation))
    fd.write(' |\n')

    # Horizontal line.
    fd.write('{:>{w}}'.format('', w=category_width))
    for _ in range(len(category_metrics)):
        fd.write('-+-{}'.format('-' * 7))
    fd.write('-+\n')

    # Metrics per category.
    for j, metrics in enumerate(category_metrics):
        fd.write('{:>{w}}'.format(categories[j], w=category_width))
        for value in metrics:
            fd.write(' | {:.5f}'.format(value))
        fd.write(' |\n')

    # Horizontal line.
    fd.write('{:>{w}}'.format('', w=category_width))
    for _ in range(len(category_metrics)):
        fd.write('-+-{}'.format('-' * 7))
    fd.write('-+\n')

    # Average metrics.
    fd.write('{:>{w}}'.format('Average', w=category_width))
    for value in averages:
        fd.write(' | {:.5f}'.format(value))
    fd.write(' |\n')

    # Confusion matrices.
    for j, category in enumerate(categories):
        fd.write('\n`{}`\n'.format(category))
        confusion = category_confusion[j]
        fd.write(np.array2string(confusion))
        fd.write('\n')


def write_predictions(Y_true, Y_pred, fd, categories):
    for j, category in enumerate(categories):
        if j > 0:
            fd.write('\t')
        fd.write('{}_true\t{}_pred'.format(category, category))
    fd.write('\n')
    for i in range(len(Y_true[0])):
        for j in range(len(Y_true)):
            if j > 0:
                fd.write('\t')
            fd.write('{:d}\t{:d}'.format(Y_true[j][i], Y_pred[j][i]))
        fd.write('\n')
