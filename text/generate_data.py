import os
import sys

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import folders
from sites.bookcave import bookcave, bookcave_ids
from text import load_embeddings


def load_X_sentences(max_words, n_tokens, padding='pre', truncating='pre', ids_fname=bookcave_ids.get_ids_fname()):
    ids_base_name = ids_fname[:ids_fname.rindex('.')]
    sentences_path = os.path.join(folders.GENERATED_PATH, ids_base_name, 'X', 'sentences')
    sentence_parameters = '{:d}v_{:d}t_{}-pad_{}-tru'.format(max_words, n_tokens, padding, truncating)
    sentence_parameters_path = os.path.join(sentences_path, sentence_parameters)
    fnames = os.listdir(sentence_parameters_path)
    X = [np.load(os.path.join(sentence_parameters_path, fname)) for fname in fnames]
    return X


def load_X_paragraph_sentences(max_words, n_sentences, n_tokens, padding='pre', truncating='pre', ids_fname=bookcave_ids.get_ids_fname()):
    ids_base_name = ids_fname[:ids_fname.rindex('.')]
    paragraph_sentences_path = os.path.join(folders.GENERATED_PATH, ids_base_name, 'X', 'paragraph_sentences')
    paragraph_sentence_parameters = \
        '{:d}v_{:d}s_{:d}t_{}-pad_{}-tru'.format(max_words, n_sentences, n_tokens, padding, truncating)
    paragraph_sentence_parameters_path = os.path.join(paragraph_sentences_path, paragraph_sentence_parameters)
    fnames = os.listdir(paragraph_sentence_parameters_path)
    X = [np.load(os.path.join(paragraph_sentence_parameters_path, fname)) for fname in fnames]
    return X


def load_Y(categories_mode='soft', ids_fname=bookcave_ids.get_ids_fname()):
    ids_base_name = ids_fname[:ids_fname.rindex('.')]
    Y_categories_mode_path = os.path.join(folders.GENERATED_PATH, ids_base_name, 'Y', '{}.npy'.format(categories_mode))
    Y = np.load(Y_categories_mode_path)
    return Y


def load_embedding_matrix(max_words, embedding_path, ids_fname=bookcave_ids.get_ids_fname()):
    ids_base_name = ids_fname[:ids_fname.rindex('.')]
    matrices_path = os.path.join(folders.GENERATED_PATH, ids_base_name, 'embedding_matrices')
    matrices_vocabulary_path = os.path.join(matrices_path, '{:d}v'.format(max_words))
    embedding_fname = os.path.basename(embedding_path)
    embedding_base_name = embedding_fname[:embedding_fname.rindex('.')]
    matrix_path = os.path.join(matrices_vocabulary_path, '{}.npy'.format(embedding_base_name))
    matrix = np.load(matrix_path)
    return matrix


def main(argv):
    if len(argv) < 3 or len(argv) > 6:
        print('Usage: <max_words> <n_sentences> <n_tokens> [padding] [truncating] [categories_mode]')
    max_words = int(argv[0])  # The maximum size of the vocabulary.
    n_sentences = int(argv[1])  # The maximum number of sentences to process in each paragraph.
    n_tokens = int(argv[2])  # The maximum number of tokens to process in each sentence.
    padding = 'pre'
    if len(argv) > 3:
        padding = argv[3]
    truncating = 'pre'
    if len(argv) > 4:
        truncating = argv[4]
    categories_mode = 'soft'
    if len(argv) > 5:
        categories_mode = argv[5]

    # Load data.
    print('Retrieving texts...')
    ids_fname = bookcave_ids.get_ids_fname()
    with open(os.path.join(folders.BOOKCAVE_IDS_PATH, ids_fname), 'r', encoding='utf-8') as fd:
        n_books = int(fd.readline()[:-1])
        book_ids = []
        for _ in range(n_books):
            book_ids.append(fd.readline()[:-1])
    inputs, Y, categories, category_levels = bookcave.get_data({'sentence_tokens'},
                                                               only_ids=book_ids,
                                                               categories_mode=categories_mode)
    text_sentence_tokens, text_section_ids, text_paragraph_ids = zip(*inputs['sentence_tokens'])
    print('Retrieved {:d} texts.'.format(len(text_sentence_tokens)))

    # Create directories for all data types.
    if not os.path.exists(folders.GENERATED_PATH):
        os.mkdir(folders.GENERATED_PATH)
    ids_base_name = ids_fname[:ids_fname.rindex('.')]
    ids_path = os.path.join(folders.GENERATED_PATH, ids_base_name)
    if not os.path.exists(ids_path):
        os.mkdir(ids_path)

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

    # Create directories for X (input).
    X_path = os.path.join(ids_path, 'X')
    if not os.path.exists(X_path):
        os.mkdir(X_path)

    # Save X for sentences to files.
    sentences_path = os.path.join(X_path, 'sentences')
    if not os.path.exists(sentences_path):
        os.mkdir(sentences_path)
    sentence_parameters = '{:d}v_{:d}t_{}-pad_{}-tru'.format(max_words, n_tokens, padding, truncating)
    sentence_parameters_path = os.path.join(sentences_path, sentence_parameters)
    if not os.path.exists(sentence_parameters_path):
        os.mkdir(sentence_parameters_path)
        # Convert to sequences.
        print('Converting texts to sequences...')
        X = [np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in sentence_tokens]),
                                    maxlen=n_tokens,
                                    padding=padding,
                                    truncating=truncating), dtype=np.int32)
             for sentence_tokens in text_sentence_tokens]  # [text_i][sentence_i][token_i]
        print('Saving tokenized sentence arrays to {}...'.format(sentence_parameters_path))
        digits = len(str(len(X)))
        for text_i, x in enumerate(X):
            x_path = os.path.join(sentence_parameters_path, '{:0>{digits}}.npy'.format(str(text_i), digits=digits))
            np.save(x_path, x)
        print('Done')
    else:
        print('Tensors for sentences for X already exist. Skipping.')

    # Save X for paragraph sentences to files.
    paragraph_sentences_path = os.path.join(X_path, 'paragraph_sentences')
    if not os.path.exists(paragraph_sentences_path):
        os.mkdir(paragraph_sentences_path)
    paragraph_sentence_parameters = \
        '{:d}v_{:d}s_{:d}t_{}-pad_{}-tru'.format(max_words, n_sentences, n_tokens, padding, truncating)
    paragraph_sentence_parameters_path = os.path.join(paragraph_sentences_path, paragraph_sentence_parameters)
    if not os.path.exists(paragraph_sentence_parameters_path):
        os.mkdir(paragraph_sentence_parameters_path)
        # Convert to sequences.
        print('Converting texts to sequences...')
        text_sentence_tensors = [pad_sequences(tokenizer.texts_to_sequences([split.join(tokens)
                                                                             for tokens in sentence_tokens]),
                                               maxlen=n_tokens,
                                               padding=padding,
                                               truncating=truncating)
                                 for sentence_tokens in text_sentence_tokens]
        X = []  # [text_i][paragraph_i][sentence_i][token_i]
        for text_i, sentence_tensors in enumerate(text_sentence_tensors):
            section_ids = text_section_ids[text_i]
            paragraph_ids = text_paragraph_ids[text_i]
            paragraph_sentence_tensors = []  # [paragraph_i][sentence_i][token_i]
            paragraph_sentence_tensor = np.zeros((n_sentences, n_tokens), dtype=np.int32)
            sentence_i = 0
            last_section_paragraph_id = None
            for i, sentence_tensor in enumerate(sentence_tensors):
                section_paragraph_id = (section_ids[i], paragraph_ids[i])
                if last_section_paragraph_id is not None and section_paragraph_id != last_section_paragraph_id:
                    paragraph_sentence_tensors.append(paragraph_sentence_tensor)
                    paragraph_sentence_tensor = np.zeros((n_sentences, n_tokens), dtype=np.int32)
                    sentence_i = 0
                if sentence_i < len(paragraph_sentence_tensor):
                    paragraph_sentence_tensor[sentence_i] = sentence_tensor
                sentence_i += 1
                last_section_paragraph_id = section_paragraph_id
            paragraph_sentence_tensors.append(paragraph_sentence_tensor)
            X.append(np.array(paragraph_sentence_tensors))
        print('Saving tokenized paragraph sentence arrays to {}...'.format(paragraph_sentence_parameters_path))
        digits = len(str(len(X)))
        for text_i, x in enumerate(X):
            x_path = os.path.join(paragraph_sentence_parameters_path, '{:0>{digits}}.npy'.format(str(text_i), digits=digits))
            np.save(x_path, x)
        print('Done.')
    else:
        print('Tensors for paragraph sentences for X already exist. Skipping.')

    # Save Y to file.
    Y_path = os.path.join(ids_path, 'Y')
    if not os.path.exists(Y_path):
        os.mkdir(Y_path)
    Y_categories_mode_path = os.path.join(Y_path, '{}.npy'.format(categories_mode))
    if not os.path.exists(Y_categories_mode_path):
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
            matrix = load_embeddings.load_embedding(tokenizer, embedding_path, max_words, header=header)
            np.save(matrix_path, matrix)
            print('Saved to `{}`.'.format(matrix_path))
    print('Done.')


if __name__ == '__main__':
    main(sys.argv[1:])
