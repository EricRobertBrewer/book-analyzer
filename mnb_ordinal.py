# Math.
import numpy as np
# Learning.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
# Data.
import bookcave as bc


def get_classifier():
    return MultinomialNB()


def get_vectorizer():
    return TfidfVectorizer(sublinear_tf=True,
                           min_df=2,
                           norm='l2',
                           encoding='latin-1',
                           ngram_range=(1, 2),
                           stop_words='english')


def to_ordinal(y, ordinal_index):
    # and use ordinal classification as explained in:
    # `Frank, Eibe, and Mark Hall. "A simple approach to ordinal classification."
    # European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2001.`.
    return np.array([1 if level > ordinal_index else 0 for level in y])


def cross_validate(folds, book_ids, book_id_to_text, category_names, category_sizes, level_names, y, seed=None):
    # Start cross-validation.
    num_correct_totals = np.zeros(len(category_names))

    # Start looping through folds before categories because text vectorization is the most time-consuming operation.
    for fold in range(folds):
        print('Starting fold {}...'.format(fold + 1))

        # Split data into train and test sets for this fold.
        book_ids_train, book_ids_test, y_train_all, y_test_all = bc.get_train_test_split(book_ids, y, fold, folds, seed=seed)
        texts_train = [book_id_to_text[book_id] for book_id in book_ids_train]
        texts_test = [book_id_to_text[book_id] for book_id in book_ids_test]

        # Create vectorized representations of the book texts.
        print('Vectorizing text...')
        vectorizer = get_vectorizer()
        vectorizer.fit(texts_train)  # Be fair, as if we were only allowed to model the training data.
        x_train = vectorizer.transform(texts_train)
        x_test = vectorizer.transform(texts_test)

        for category_index, category_name in enumerate(category_names):
            # Perform ordinal classification.
            category_size = category_sizes[category_name]
            y_train = y_train_all[:, category_index]
            y_test = y_test_all[:, category_index]

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
            num_correct_totals[category_index] += num_correct

    accuracies = num_correct_totals/len(y)
    for i, accuracy in enumerate(accuracies):
        print('`{}` overall accuracy: {:.4%}'.format(category_names[i], accuracy))


def main():
    # book_ids, book_id_to_text, category_names, category_sizes, y = bcdata.get_data('book.txt')
    book_ids, book_id_to_text, category_names, category_sizes, level_names, y = bc.get_data('text.txt', kindle=True)
    seed = 1
    cross_validate(5, book_ids, book_id_to_text, category_names, category_sizes, level_names, y, seed=seed)


if __name__ == '__main__':
    main()
