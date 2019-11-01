import os

import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

from classification import shared_parameters
import folders
from sites.bookcave import bookcave


def write_formatted_term_scores(category, term_scores, size, min_gram, max_gram, max_features, force=False):
    fname = folders.CORRELATED_WORDS_FNAME_FORMAT.format(category,
                                                         size,
                                                         min_gram,
                                                         max_gram,
                                                         max_features)
    path = os.path.join(folders.CORRELATED_WORDS_PATH, fname)
    if os.path.exists(path):
        if force:
            os.remove(path)
        else:
            return

    with open(path, 'w', encoding='utf-8') as fd:
        fd.write('{:d}\n'.format(len(term_scores)))
        for term, score in term_scores:
            fd.write('{}\t{}\n'.format(term, score))


def read_formatted_term_scores(category, size, min_gram, max_gram, max_features, top_n):
    fname = folders.CORRELATED_WORDS_FNAME_FORMAT.format(category,
                                                         size,
                                                         min_gram,
                                                         max_gram,
                                                         max_features,
                                                         top_n)
    path = os.path.join(folders.CORRELATED_WORDS_PATH, fname)
    if not os.path.exists(path):
        return None

    term_scores = []
    with open(path, 'r', encoding='utf-8') as fd:
        n = int(fd.readline()[:-1])
        for _ in range(min(top_n, n)):
            term_and_score = fd.readline()[:-1].split('\t')
            term = term_and_score[0]
            score = float(term_and_score[1])
            term_scores.append((term, score))
    return term_scores


def main(min_len=shared_parameters.DATA_PARAGRAPH_MIN_LEN,
         max_len=shared_parameters.DATA_PARAGRAPH_MAX_LEN,
         min_tokens=shared_parameters.DATA_MIN_TOKENS,
         categories_mode=shared_parameters.DATA_CATEGORIES_MODE,
         return_overall=shared_parameters.DATA_RETURN_OVERALL,
         min_gram=1,
         max_gram=1,
         max_features=8192,
         top_n=256,
         force=False):
    # Get data.
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'paragraph_tokens'},
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          categories_mode=categories_mode,
                          return_overall=return_overall)
    text_paragraph_tokens = [paragraph_tokens for paragraph_tokens, _ in inputs['paragraph_tokens']]
    text_all_tokens = []
    for paragraph_tokens in text_paragraph_tokens:
        all_tokens = []
        for tokens in paragraph_tokens:
            all_tokens.extend(tokens)
        text_all_tokens.append(all_tokens)

    # Vectorize.
    def identity(v):
        return v

    vectorizer = TfidfVectorizer(
        preprocessor=identity,
        tokenizer=identity,
        analyzer='word',
        token_pattern=None,
        ngram_range=(min_gram, max_gram),
        max_features=max_features,
        norm='l2',
        sublinear_tf=True)
    X = vectorizer.fit_transform(text_all_tokens)
    features = vectorizer.get_feature_names()

    # See Multi Class Text Classification article:
    # https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
    category_term_scores = []
    for category_i, category in enumerate(categories):
        y = Y[category_i]
        scores, pvals = chi2(X, y)
        indices = np.argsort(scores)
        term_scores = [(features[indices[-1 - i]], scores[indices[-1 - i]]) for i in range(top_n)]
        category_term_scores.append(term_scores)

    # Save.
    if not os.path.exists(folders.LOGS_PATH):
        os.mkdir(folders.LOGS_PATH)
    if not os.path.exists(folders.CORRELATED_WORDS_PATH):
        os.mkdir(folders.CORRELATED_WORDS_PATH)
    size = len(text_all_tokens)
    for category_i, category in enumerate(categories):
        term_scores = category_term_scores[category_i]
        write_formatted_term_scores(category, term_scores, size, min_gram, max_gram, max_features, force=force)
    return size


if __name__ == '__main__':
    print(main())
