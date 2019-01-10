# Math.
import numpy as np
# Learning.
from sklearn.metrics import accuracy_score

import bookcave as bc


def cross_validate(folds, book_ids, book_id_to_text, category_names, category_sizes, level_names, y, seed=None):
    # Start cross-validation.
    num_correct_totals = np.zeros(len(category_names))
    # Start looping through folds before categories because text vectorization is the most time-consuming operation.
    for fold in range(folds):
        # Split data into train and test sets for this fold.
        _, _, y_train_all, y_test_all = bc.get_train_test_split(book_ids, y, fold, folds, seed=seed)
        for category_index, category_name in enumerate(category_names):
            category_size = category_sizes[category_name]
            y_train = y_train_all[:, category_index]
            counts = [0 for _ in range(category_size)]
            for value in y_train:
                counts[value] += 1
            p_max = np.argmax(counts)

            y_test = y_test_all[:, category_index]
            y_pred = [p_max for _ in range(len(y_test))]
            num_correct = accuracy_score(y_test, y_pred, normalize=False)
            num_correct_totals[category_index] += num_correct
    accuracies = num_correct_totals/len(y)
    for i, accuracy in enumerate(accuracies):
        print('`{}` overall accuracy: {:.3%}'.format(category_names[i], accuracy))


def main():
    # book_ids, book_id_to_text, category_names, category_sizes, y = bc.get_data('book.txt')
    book_ids, book_id_to_text, category_names, category_sizes, level_names, y = bc.get_data('text.txt', kindle=True)
    seed = 1
    cross_validate(5, book_ids, book_id_to_text, category_names, category_sizes, level_names, y, seed=seed)


if __name__ == '__main__':
    main()
