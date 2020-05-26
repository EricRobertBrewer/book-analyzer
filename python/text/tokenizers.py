import os
import pickle

from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from python import folders

SPLIT = '\t'


def get_tokenizer_or_fit(max_words, source_mode, remove_stopwords, text_source_tokens=None):
    tokenizer_file_name = 'tokenizer-{:d}-{}{}.pickle'.format(max_words,
                                                              source_mode,
                                                              '-nostop' if remove_stopwords else '')
    tokenizer_path = os.path.join(folders.TOKENIZERS_PATH, tokenizer_file_name)
    if not os.path.exists(tokenizer_path):
        if text_source_tokens is None:
            raise ValueError('Tokenizer at \'' + tokenizer_path +'\' does not exist. `text_source_tokens` is required.')
        tokenizer = Tokenizer(num_words=max_words, split=SPLIT)
        all_sources = []
        for source_tokens in text_source_tokens:
            for tokens in source_tokens:
                all_sources.append(SPLIT.join(tokens))
        tokenizer.fit_on_texts(all_sources)
        with open(tokenizer_path, 'wb') as fd:
            pickle.dump(tokenizer, fd, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(tokenizer_path, 'rb') as fd:
            tokenizer = pickle.load(fd)

    return tokenizer


def get_vectorizer_or_fit(max_words, remove_stopwords, text_tokens=None):
    vectorizer_file_name = 'vectorizer-{:d}{}.pickle'.format(max_words,
                                                             '-nostop' if remove_stopwords else '')
    vectorizer_path = os.path.join(folders.TOKENIZERS_PATH, vectorizer_file_name)
    if not os.path.exists(vectorizer_path):
        if text_tokens is None:
            raise ValueError('Vectorizer at \'' + vectorizer_path +'\' does not exist. `text_tokens` is required.')
        vectorizer = TfidfVectorizer(
            preprocessor=identity,
            tokenizer=identity,
            analyzer='word',
            token_pattern=None,
            max_features=max_words,
            norm='l2',
            sublinear_tf=True)
        vectorizer.fit(text_tokens)
        with open(vectorizer_path, 'wb') as fd:
            pickle.dump(vectorizer, fd, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(vectorizer_path, 'rb') as fd:
            vectorizer = pickle.load(fd)

    return vectorizer


def identity(x):
    return x
