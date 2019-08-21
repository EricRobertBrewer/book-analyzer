from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score


def print_metrics(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)

    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: {:.4f}'.format(accuracy))

    precision_macro = precision_score(y_true, y_pred, average='macro')
    print('precision_macro={:.4f}'.format(precision_macro))
    recall_macro = recall_score(y_true, y_pred, average='macro')
    print('recall_macro={:.4f}'.format(recall_macro))
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print('f1_macro={:.4f}'.format(f1_macro))

    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    print('precision_weighted={:.4f}'.format(precision_weighted))
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    print('recall_weighted={:.4f}'.format(recall_weighted))
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print('f1_weighted={:.4f}'.format(f1_weighted))

    mse = mean_squared_error(y_true, y_pred)
    print('MSE={:.4f}'.format(mse))
