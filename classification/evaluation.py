import sys

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score


def print_metrics(y_true, y_pred, fd=sys.stdout):
    confusion = confusion_matrix(y_true, y_pred)
    fd.write(confusion)
    fd.write('\n')

    accuracy = accuracy_score(y_true, y_pred)
    fd.write('Accuracy: {:.4f}\n'.format(accuracy))

    precision_macro = precision_score(y_true, y_pred, average='macro')
    fd.write('precision_macro={:.4f}\n'.format(precision_macro))
    recall_macro = recall_score(y_true, y_pred, average='macro')
    fd.write('recall_macro={:.4f}\n'.format(recall_macro))
    f1_macro = f1_score(y_true, y_pred, average='macro')
    fd.write('f1_macro={:.4f}\n'.format(f1_macro))

    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    fd.write('precision_weighted={:.4f}\n'.format(precision_weighted))
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    fd.write('recall_weighted={:.4f}\n'.format(recall_weighted))
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    fd.write('f1_weighted={:.4f}\n'.format(f1_weighted))

    mse = mean_squared_error(y_true, y_pred)
    fd.write('MSE={:.4f}\n'.format(mse))
