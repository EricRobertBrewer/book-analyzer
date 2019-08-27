from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score


def get_metrics(y_true, y_pred):
    return confusion_matrix(y_true, y_pred), [
        ('accuracy', accuracy_score(y_true, y_pred)),
        ('precision_macro', precision_score(y_true, y_pred, average='macro')),
        ('recall_macro', recall_score(y_true, y_pred, average='macro')),
        ('f1_macro', f1_score(y_true, y_pred, average='macro')),
        ('precision_weighted', precision_score(y_true, y_pred, average='weighted')),
        ('recall_weighted', recall_score(y_true, y_pred, average='weighted')),
        ('f1_weighted', f1_score(y_true, y_pred, average='weighted')),
        ('MSE', mean_squared_error(y_true, y_pred))
    ]
