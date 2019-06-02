import numpy as np

from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, GlobalMaxPool1D, GRU, Input
from keras.models import Model
import tensorflow as tf

from sites.bookcave import bookcave


def get_embedding(tokenizer, fname, max_words=10000):
    word_index = tokenizer.word_index
    word_count = min(max_words, len(word_index))
    with open(fname, 'r', encoding='utf-8') as fd:
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in fd)
    all_embeddings = np.stack(embeddings_index.values())
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
    return embed_size, embedding_matrix


def get_coefs(word, *arr):
    """
    Read the GloVe word vectors.
    """
    return word, np.asarray(arr, dtype='float32')


def create_model(category, n_classes, n_tokens, embedding_matrix, hidden_size, dense_size, train_emb=True):
    inp = Input(shape=(n_tokens,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=train_emb)(inp)
    x = Bidirectional(GRU(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Each model outputs ordinal labels; thus, one less than the number of classes.
    x = Dense(n_classes - 1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inp, x)

    weights_fname = 'model_{:d}t_{:d}v_{:d}d_{:d}h_{:d}f{}_{}.h5'.format(n_tokens,
                                                                         embedding_matrix.shape[0],
                                                                         embedding_matrix.shape[1],
                                                                         hidden_size,
                                                                         dense_size,
                                                                         '' if train_emb else '_static',
                                                                         category)

    return model, weights_fname


def main():
    print('TensorFlow version: {}'.format(tf.__version__))

    label_inputs, Y, categories, labels =\
        bookcave.get_data({'text'},
                          text_source='labels',
                          only_categories={bookcave.CATEGORY_INDEX_DRUG_ALCOHOL_TOBACCO_USE,
                                           bookcave.CATEGORY_INDEX_SEX_AND_INTIMACY,
                                           bookcave.CATEGORY_INDEX_VIOLENCE_AND_HORROR})
    labels = label_inputs['text']
    print(len(labels))


if __name__ == '__main__':
    main()
