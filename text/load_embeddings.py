from io import open

import numpy as np


def get_coefs(word, *arr):
    """
    Read the GloVe word vectors.
    """
    return word, np.asarray(arr, dtype='float32')


def get_embedding(tokenizer, fname, max_words=10000):
    word_index = tokenizer.word_index
    word_count = min(max_words, len(word_index))
    with open(fname, 'r', encoding='utf-8') as fd:
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in fd)
    all_embeddings = np.stack(list(embeddings_index.values()))
    embed_size = all_embeddings.shape[1]
    mean, std = all_embeddings.mean(), all_embeddings.std()
    # Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe.
    # We'll use the same mean and standard deviation of embeddings that GloVe has when generating the random init.
    embedding_matrix = np.random.normal(mean, std, (word_count, embed_size))
    for word, i in word_index.items():
        if i >= word_count:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
