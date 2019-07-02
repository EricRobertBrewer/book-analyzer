# Math.
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score

from sites.bookcave import bookcave


def main():
    min_len, max_len = 250, 7500
    _, Y, categories, category_levels =\
        bookcave.get_data({'paragraphs'},
                          min_len=min_len,
                          max_len=max_len)
    print(Y.shape)

    seed = 1
    np.random.seed(seed)
    permutation = np.random.permutation(Y.shape[1])
    test_size = .25
    pivot = int(test_size * Y.shape[1])
    test_indices, train_indices = permutation[:pivot], permutation[pivot:]

    for category_index, category in enumerate(categories):
        levels = category_levels[category_index]
        labels = [i for i in range(len(levels))]

        y = Y[category_index]
        y_test, y_train = y[test_indices], y[train_indices]
        bincount = np.bincount(y_train, minlength=len(levels))
        argmax = np.argmax(bincount)
        y_pred = [argmax]*len(y_test)

        print()
        print(category)
        print('majority=`{}` ({:d}/{:d})'.format(levels[argmax], bincount[argmax], len(y_train)))
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy={:.4f}'.format(accuracy))

        print(confusion_matrix(y_test, y_pred, labels=labels))

        precision_micro = precision_score(y_test, y_pred, average='micro')
        print('precision_micro={:.4f}'.format(precision_micro))
        recall_micro = recall_score(y_test, y_pred, average='micro')
        print('recall_micro={:.4f}'.format(recall_micro))
        f1_micro = f1_score(y_test, y_pred, average='micro')
        print('f1_micro={:.4f}'.format(f1_micro))

        precision_macro = precision_score(y_test, y_pred, average='macro')
        print('precision_macro={:.4f}'.format(precision_macro))
        recall_macro = recall_score(y_test, y_pred, average='macro')
        print('recall_macro={:.4f}'.format(recall_macro))
        f1_macro = f1_score(y_test, y_pred, average='macro')
        print('f1_macro={:.4f}'.format(f1_macro))

        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        print('precision_weighted={:.4f}'.format(precision_weighted))
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        print('recall_weighted={:.4f}'.format(recall_weighted))
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        print('f1_weighted={:.4f}'.format(f1_weighted))


if __name__ == '__main__':
    main()
