import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sites.bookcave import bookcave
from classification import ordinal, shared_parameters


def identity(v):
    return v


def create_mnb():
    return MultinomialNB(fit_prior=True)


def create_lr():
    return LogisticRegression(solver='lbfgs')


def create_rf():
    return RandomForestClassifier(n_estimators=6)


def create_svm():
    return LinearSVC()


def main():
    # Load data.
    print('Retrieving texts...')
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_MIN_LEN
    max_len = shared_parameters.DATA_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'paragraph_tokens'},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens)
    text_paragraph_tokens, _ = zip(*inputs['paragraph_tokens'])
    print('Retrieved {:d} texts.'.format(len(text_paragraph_tokens)))

    # Create vectorized representations of the book texts.
    print('Vectorizing text...')
    max_words = 4096
    vectorizer = TfidfVectorizer(
        preprocessor=identity,
        tokenizer=identity,
        analyzer='word',
        token_pattern=None,
        max_features=max_words,
        norm='l2',
        sublinear_tf=True)
    text_tokens = []
    for paragraph_tokens in text_paragraph_tokens:
        all_tokens = []
        for tokens in paragraph_tokens:
            all_tokens.extend(tokens)
        text_tokens.append(all_tokens)
    X = vectorizer.fit_transform(text_tokens)
    print('Vectorized text with {:d} unique words.'.format(len(vectorizer.get_feature_names())))

    # Split data set.
    test_size = .25  # b
    test_random_state = 1
    Y_T = Y.transpose()  # (n, c)
    X_train, X_test, Y_train_T, Y_test_T = train_test_split(X, Y_T, test_size=test_size, random_state=test_random_state)
    Y_train = Y_train_T.transpose()  # (c, n * (1 - b))
    Y_test = Y_test_T.transpose()  # (c, n * b)

    create_models = [create_mnb, create_lr, create_rf, create_svm]
    model_names = ['mnb', 'lr', 'rf', 'svm']
    for m, create_model in enumerate(create_models):
        print()
        print('='*72)
        print()
        print(model_names[m])
        for j, category in enumerate(categories):
            print()
            print('Classifying category `{}`...'.format(category))

            k = len(category_levels[j])
            y_train = Y_train[j]  # (n * (1 - b))
            y_test = Y_test[j]  # (n * b)
            # Calculate probabilities for derived data sets.
            y_train_ordinal = ordinal.to_multi_hot_ordinal(y_train, k=k)  # (n * (1 - b), k - 1)
            classifiers = [create_model() for _ in range(k - 1)]
            ordinal_p = np.zeros((len(y_test), k - 1))  # (n * b, k - 1)
            for i, classifier in enumerate(classifiers):
                classifier.fit(X_train, y_train_ordinal[:, i])
                ordinal_p[:, i] = classifier.predict(X_test)
            # Calculate the actual class label probabilities.
            p = np.zeros((len(y_test), k))  # (n * b, k)
            for i in range(k):
                if i == 0:
                    p[:, i] = 1 - ordinal_p[:, 0]
                elif i == k - 1:
                    p[:, i] = ordinal_p[:, i - 1]
                else:
                    p[:, i] = ordinal_p[:, i - 1] - ordinal_p[:, i]
            # Choose the most likely class label.
            y_pred = np.argmax(p, axis=1)  # (n * b)

            confusion = confusion_matrix(y_test, y_pred)
            print(confusion)
            num_correct = accuracy_score(y_test, y_pred, normalize=False)
            print('Accuracy: {:.4f} ({:d}/{:d})'.format(num_correct / len(y_test), int(num_correct), len(y_test)))


if __name__ == '__main__':
    main()
