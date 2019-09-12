import os
import sys

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import folders
from sites.bookcave import bookcave, bookcave_ids
from text import load_embeddings


def load_sentence_data(max_words, n_tokens, padding='pre', truncating='pre', categories_mode='soft', embedding_path=None):
    ids_fname = bookcave_ids.get_fname()
    ids_base_name = ids_fname[:ids_fname.rindex('.')]
    ids_path = os.path.join(folders.GENERATED_PATH, 'sentences', ids_base_name)
    if not os.path.exists(ids_path):
        raise ValueError('Data for the given parameters `(max_words={:d}, n_tokens={:d}` do not exist. Quitting.'.format(max_words, n_tokens))

    X_parameters_path = os.path.join(ids_path, 'X', '{:d}v_{:d}l_{}-pad_{}-tru'.format(max_words, n_tokens, padding, truncating))
    fnames = os.listdir(X_parameters_path)
    X = [np.load(os.path.join(X_parameters_path, fname)) for fname in fnames]

    Y_categories_mode_path = os.path.join(ids_path, 'Y', '{}.npy'.format(categories_mode))
    Y = np.load(Y_categories_mode_path)

    if embedding_path is None:
        return X, Y

    matrices_vocabulary_path = os.path.join(ids_path, 'embedding_matrices', '{:d}v'.format(max_words))
    embedding_fname = os.path.basename(embedding_path)
    embedding_base_name = embedding_fname[:embedding_fname.rindex('.')]
    matrix_path = os.path.join(matrices_vocabulary_path, '{}.npy'.format(embedding_base_name))
    matrix = np.load(matrix_path)
    return X, Y, matrix


def main(argv):
    if len(argv) < 2 or len(argv) > 5:
        print('Usage: <max_words> <n_tokens> [padding] [truncating] [categories_mode]')
    max_words = int(argv[0])  # The maximum size of the vocabulary.
    n_tokens = int(argv[1])  # The maximum number of tokens to process in each sentence.
    padding = 'pre'
    if len(argv) > 2:
        padding = argv[2]
    truncating = 'pre'
    if len(argv) > 3:
        truncating = argv[3]
    categories_mode = 'soft'
    if len(argv) > 4:
        categories_mode = argv[4]

    # Load data.
    print('Retrieving texts...')
    ids_fname = bookcave_ids.get_fname()
    with open(os.path.join(folders.BOOKCAVE_IDS_PATH, ids_fname), 'r', encoding='utf-8') as fd:
        n_books = int(fd.readline()[:-1])
        book_ids = []
        for _ in range(n_books):
            book_ids.append(fd.readline()[:-1])
    inputs, Y, categories, category_levels = bookcave.get_data({'sentence_tokens'},
                                                               only_ids=book_ids,
                                                               categories_mode=categories_mode)
    text_sentence_tokens, _, _ = zip(*inputs['sentence_tokens'])
    print('Retrieved {:d} texts.'.format(len(text_sentence_tokens)))

    # Tokenize.
    print('Tokenizing...')
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_sentences = []
    for sentence_tokens in text_sentence_tokens:
        for tokens in sentence_tokens:
            all_sentences.append(split.join(tokens))
    tokenizer.fit_on_texts(all_sentences)
    print('Done.')

    # Create directories.
    if not os.path.exists(folders.GENERATED_PATH):
        os.mkdir(folders.GENERATED_PATH)
    sentences_path = os.path.join(folders.GENERATED_PATH, 'sentences')
    if not os.path.exists(sentences_path):
        os.mkdir(sentences_path)
    ids_base_name = ids_fname[:ids_fname.rindex('.')]
    ids_path = os.path.join(sentences_path, ids_base_name)
    if not os.path.exists(ids_path):
        os.mkdir(ids_path)

    # Save X to files.
    X_path = os.path.join(ids_path, 'X')
    if not os.path.exists(X_path):
        os.mkdir(X_path)
    X_parameters_path = os.path.join(X_path, '{:d}v_{:d}l_{}-pad_{}-tru'.format(max_words, n_tokens, padding, truncating))
    if not os.path.exists(X_parameters_path):
        # Convert to sequences.
        print('Converting texts to sequences...')
        X = [np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in sentence_tokens]),
                                    maxlen=n_tokens,
                                    padding=padding,
                                    truncating=truncating), dtype=np.int32)
             for sentence_tokens in text_sentence_tokens]  # [text_i][sentence_i][token_i]
        print('Saving arrays to {}...'.format(X_parameters_path))
        os.mkdir(X_parameters_path)
        digits = len(str(len(X)))
        for text_i, x in enumerate(X):
            x_path = os.path.join(X_parameters_path, '{:0>{digits}}.npy'.format(str(text_i), digits=digits))
            np.save(x_path, x)
        print('Done')
    else:
        print('Tensors for X already exist. Skipping.')

    # Save Y to file.
    Y_path = os.path.join(ids_path, 'Y')
    if not os.path.exists(Y_path):
        os.mkdir(Y_path)
    Y_categories_mode_path = os.path.join(Y_path, '{}.npy'.format(categories_mode))
    if not os.path.exists(Y_categories_mode_path):
        os.mkdir(Y_categories_mode_path)
        np.save(Y_categories_mode_path, Y)
        print('Saved Y to `{}`.'.format(Y_categories_mode_path))
    else:
        print('File for Y already exists. Skipping.')

    # Load and save embeddings.
    matrices_path = os.path.join(ids_path, 'embedding_matrices')
    if not os.path.exists(matrices_path):
        os.mkdir(matrices_path)
    matrices_vocabulary_path = os.path.join(matrices_path, '{:d}v'.format(max_words))
    if not os.path.exists(matrices_vocabulary_path):
        os.mkdir(matrices_vocabulary_path)
    embedding_paths = [
        folders.EMBEDDING_FASTTEXT_CRAWL_300_PATH,
        folders.EMBEDDING_GLOVE_100_PATH,
        folders.EMBEDDING_GLOVE_200_PATH,
        folders.EMBEDDING_GLOVE_300_PATH
    ]
    print('Loading and saving embedding matrices...')
    for embedding_path in embedding_paths:
        embedding_fname = os.path.basename(embedding_path)
        embedding_base_name = embedding_fname[:embedding_fname.rindex('.')]
        matrix_path = os.path.join(matrices_vocabulary_path, '{}.npy'.format(embedding_base_name))
        if not os.path.exists(matrix_path):
            print('Loading embedding matrix `{}`...'.format(embedding_path))
            header = embedding_path.endswith('.vec')
            matrix = load_embeddings.get_embedding_matrix(tokenizer, embedding_path, max_words, header=header)
            np.save(matrix_path, matrix)
            print('Saved to `{}`.'.format(matrix_path))
    print('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])
