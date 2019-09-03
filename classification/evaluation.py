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


def get_confusion_and_metrics(y_true, y_pred):
    return confusion_matrix(y_true, y_pred), [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='macro'),
        precision_score(y_true, y_pred, average='weighted'),
        recall_score(y_true, y_pred, average='weighted'),
        f1_score(y_true, y_pred, average='weighted'),
        mean_squared_error(y_true, y_pred)
    ]
