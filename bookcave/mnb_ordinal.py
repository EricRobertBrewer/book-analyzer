# Math.
import numpy as np
# Learning.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
# Data.
import bookcave.data as bcdata


def get_classifier():
    return MultinomialNB()


def get_vectorizer():
    return TfidfVectorizer()


def get_train_test_split(book_ids, book_id_to_preview, y, category_index, perm, fold, folds):
    # Cross validate...
    test_start = len(y) * fold // folds
    test_end = len(y) * (fold + 1) // folds
    perm_train = np.concatenate((perm[:test_start], perm[test_end:]))
    perm_test = perm[test_start:test_end]
    previews_train = [book_id_to_preview[book_id] for book_id in book_ids[perm_train]]
    previews_test = [book_id_to_preview[book_id] for book_id in book_ids[perm_test]]
    y_train = y[perm_train, category_index]
    y_test = y[perm_test, category_index]
    return previews_train, previews_test, y_train, y_test


def to_ordinal(y, ordinal_index):
    # and use ordinal classification as explained in
    # `Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification."
    # European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2001.`.
    return np.array([1 if level > ordinal_index else 0 for level in y])


def cross_validate(folds, book_ids, book_id_to_preview, category_names, category_sizes, y, perm):
    for category_index, category_name in enumerate(category_names):
        print('Evaluating category `{}`...'.format(category_name))
        category_size = category_sizes[category_name]
        # Start cross-validation.
        num_correct_total = 0
        for fold in range(folds):
            print('Starting fold {}...'.format(fold + 1))
            # Split data into train and test sets for this fold.
            previews_train, previews_test, y_train, y_test = get_train_test_split(book_ids, book_id_to_preview, y,
                                                                                  category_index, perm, fold, folds)
            # Create vectorized representations of the book previews.
            vectorizer = get_vectorizer()
            vectorizer.fit(previews_train)  # Be fair, as if we were only allowed to model the training data.
            x_train = vectorizer.transform(previews_train)
            x_test = vectorizer.transform(previews_test)
            # Perform ordinal classification.
            # Get probabilities for binarized ordinal labels.
            ordinal_ps = np.zeros((len(y_test), category_size - 1))
            for ordinal_index in range(category_size - 1):
                # Find P(Target > Class_k) for 0..(k-1)
                classifier = get_classifier()
                y_train_ordinal = to_ordinal(y_train, ordinal_index)
                classifier.fit(x_train, y_train_ordinal)
                ordinal_ps[:, ordinal_index] = classifier.predict(x_test)
            # Calculate the actual class label probabilities.
            ps = np.zeros((len(y_test), category_size))
            for level_index in range(category_size):
                if level_index == 0:
                    ps[:, level_index] = 1 - ordinal_ps[:, 0]
                elif level_index == category_size - 1:
                    ps[:, level_index] = ordinal_ps[:, level_index - 1]
                else:
                    ps[:, level_index] = ordinal_ps[:, level_index - 1] - ordinal_ps[:, level_index]
            # Choose the most likely class label.
            y_pred = np.argmax(ps, axis=1)
            num_correct = accuracy_score(y_test, y_pred, normalize=False)
            num_correct_total += num_correct
        accuracy = num_correct_total/len(y)
        print('Accuracy: {:.4%}'.format(accuracy))


def main():
    book_ids, book_id_to_preview, category_names, category_sizes, y = bcdata.get_data()
    # Generate a random permutation in order to process the data set in a random order.
    # np.random.seed(1)
    perm = np.random.permutation(len(y))
    cross_validate(5, book_ids, book_id_to_preview, category_names, category_sizes, y, perm=perm)


if __name__ == '__main__':
    main()
