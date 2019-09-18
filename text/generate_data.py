import os
import sys

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import folders
from sites.bookcave import bookcave, bookcave_ids
from text import load_embeddings


def get_ids_base_name(ids_fname):
    return ids_fname[:ids_fname.rindex('.')]


def get_sentence_parameters_path(max_words, n_sentence_tokens, padding, truncating, ids_fname):
    ids_base_name = get_ids_base_name(ids_fname)
    parameters = '{:d}v_{:d}t_{}-pad_{}-tru'.format(max_words, n_sentence_tokens, padding, truncating)
    return os.path.join(folders.GENERATED_PATH, ids_base_name, 'X', 'sentences', parameters)


def get_paragraph_parameters_path(max_words, n_paragraph_tokens, padding, truncating, ids_fname):
    ids_base_name = get_ids_base_name(ids_fname)
    parameters = '{:d}v_{:d}t_{}-pad_{}-tru'.format(max_words, n_paragraph_tokens, padding, truncating)
    return os.path.join(folders.GENERATED_PATH, ids_base_name, 'X', 'paragraphs', parameters)


def get_sentence_paragraph_parameters_path(max_words, n_sentences, n_sentence_tokens, padding, truncating, ids_fname):
    ids_base_name = get_ids_base_name(ids_fname)
    parameters = \
        '{:d}v_{:d}s_{:d}t_{}-pad_{}-tru'.format(max_words, n_sentences, n_sentence_tokens, padding, truncating)
    return os.path.join(folders.GENERATED_PATH, ids_base_name, 'X', 'paragraph_sentences', parameters)


def load_X_sentences(max_words, n_sentence_tokens, padding='pre', truncating='pre', ids_fname=bookcave_ids.get_ids_fname()):
    sentence_parameters_path = \
        get_sentence_parameters_path(max_words, n_sentence_tokens, padding, truncating, ids_fname)
    fnames = os.listdir(sentence_parameters_path)
    return [np.load(os.path.join(sentence_parameters_path, fname)) for fname in fnames]


def load_X_paragraphs(max_words, n_paragraph_tokens, padding='pre', truncating='pre', ids_fname=bookcave_ids.get_ids_fname()):
    paragraph_parameters_path = \
        get_paragraph_parameters_path(max_words, n_paragraph_tokens, padding, truncating, ids_fname)
    fnames = os.listdir(paragraph_parameters_path)
    return [np.load(os.path.join(paragraph_parameters_path, fname)) for fname in fnames]


def load_X_paragraph_sentences(max_words, n_sentences, n_sentence_tokens, padding='pre', truncating='pre', ids_fname=bookcave_ids.get_ids_fname()):
    paragraph_sentence_parameters_path = \
        get_sentence_paragraph_parameters_path(max_words, n_sentences, n_sentence_tokens, padding, truncating, ids_fname)
    fnames = os.listdir(paragraph_sentence_parameters_path)
    return [np.load(os.path.join(paragraph_sentence_parameters_path, fname)) for fname in fnames]


def load_Y(categories_mode='soft', ids_fname=bookcave_ids.get_ids_fname()):
    ids_base_name = get_ids_base_name(ids_fname)
    Y_categories_mode_path = os.path.join(folders.GENERATED_PATH, ids_base_name, 'Y', '{}.npy'.format(categories_mode))
    return np.load(Y_categories_mode_path)


def load_embedding_matrix(max_words, embedding_path, ids_fname=bookcave_ids.get_ids_fname()):
    ids_base_name = get_ids_base_name(ids_fname)
    matrices_path = os.path.join(folders.GENERATED_PATH, ids_base_name, 'embedding_matrices')
    matrices_vocabulary_path = os.path.join(matrices_path, '{:d}v'.format(max_words))
    embedding_fname = os.path.basename(embedding_path)
    embedding_base_name = embedding_fname[:embedding_fname.rindex('.')]
    matrix_path = os.path.join(matrices_vocabulary_path, '{}.npy'.format(embedding_base_name))
    return np.load(matrix_path)


def main(argv):
    if len(argv) < 4 or len(argv) > 7:
        print('Usage: <max_words> <n_sentences> <n_sentence_tokens> <n_paragraph_tokens> [padding] [truncating] [categories_mode]')
    max_words = int(argv[0])  # The maximum size of the vocabulary.
    n_sentences = int(argv[1])  # The maximum number of sentences to process in each paragraph.
    n_sentence_tokens = int(argv[2])  # The maximum number of tokens to process in each sentence.
    n_paragraph_tokens = int(argv[3])    # The maximum number of tokens to process in each paragraph.
    padding = 'pre'
    if len(argv) > 4:
        padding = argv[4]
    truncating = 'pre'
    if len(argv) > 5:
        truncating = argv[5]
    categories_mode = 'soft'
    if len(argv) > 6:
        categories_mode = argv[6]

    # Load data.
    print('Retrieving texts...')
    ids_fname = bookcave_ids.get_ids_fname()
    book_ids = bookcave_ids.get_book_ids(ids_fname)
    inputs, Y, categories, category_levels, book_ids, _, _, _, _ = \
        bookcave.get_data({'sentence_tokens'},
                          only_ids=book_ids,
                          categories_mode=categories_mode,
                          return_meta=True)
    text_sentence_tokens, text_section_ids, text_paragraph_ids = zip(*inputs['sentence_tokens'])
    print('Retrieved {:d} texts.'.format(len(text_sentence_tokens)))

    # Create directories for all data types.
    if not os.path.exists(folders.GENERATED_PATH):
        os.mkdir(folders.GENERATED_PATH)
    ids_base_name = get_ids_base_name(ids_fname)
    ids_path = os.path.join(folders.GENERATED_PATH, ids_base_name)
    if not os.path.exists(ids_path):
        os.mkdir(ids_path)

    # Save Y to file.
    Y_path = os.path.join(ids_path, 'Y')
    if not os.path.exists(Y_path):
        os.mkdir(Y_path)
    Y_categories_mode_path = os.path.join(Y_path, '{}.npy'.format(categories_mode))
    if not os.path.exists(Y_categories_mode_path):
        print('Saved Y to `{}`.'.format(Y_categories_mode_path))
        np.save(Y_categories_mode_path, Y)

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
    sentence_parameters_path = \
        get_sentence_parameters_path(max_words, n_sentence_tokens, padding, truncating, ids_fname)
    if not os.path.exists(sentence_parameters_path):
        os.mkdir(sentence_parameters_path)

    # Save X for paragraphs to files.
    paragraphs_path = os.path.join(X_path, 'paragraphs')
    if not os.path.exists(paragraphs_path):
        os.mkdir(paragraphs_path)
    paragraph_parameters_path = \
        get_paragraph_parameters_path(max_words, n_paragraph_tokens, padding, truncating, ids_fname)
    if not os.path.exists(paragraph_parameters_path):
        os.mkdir(paragraph_parameters_path)

    # Save X for paragraph sentences to files.
    paragraph_sentences_path = os.path.join(X_path, 'paragraph_sentences')
    if not os.path.exists(paragraph_sentences_path):
        os.mkdir(paragraph_sentences_path)
    paragraph_sentence_parameters_path = \
        get_sentence_paragraph_parameters_path(max_words, n_sentences, n_sentence_tokens, padding, truncating, ids_fname)
    if not os.path.exists(paragraph_sentence_parameters_path):
        os.mkdir(paragraph_sentence_parameters_path)

    # Save X to files.
    digits = len(str(len(text_sentence_tokens)))
    for text_i, sentence_tokens in enumerate(text_sentence_tokens):
        fname = '{:0>{digits}}.npy'.format(str(text_i), digits=digits)
        x_sentences_path = os.path.join(sentence_parameters_path, fname)
        x_paragraphs_path = os.path.join(paragraph_parameters_path, fname)
        x_paragraph_sentences_path = os.path.join(paragraph_sentence_parameters_path, fname)
        if os.path.exists(x_sentences_path) and os.path.exists(x_paragraphs_path) and os.path.exists(x_paragraph_sentences_path):
            continue

        book_id = book_ids[text_i]
        sentence_sequences = pad_sequences(tokenizer.texts_to_sequences([split.join(tokens)
                                                                         for tokens in sentence_tokens]),
                                           maxlen=n_sentence_tokens,
                                           padding=padding,
                                           truncating=truncating)

        if not os.path.exists(x_sentences_path):
            print('Saving tokenized sentence arrays for {}...'.format(book_id))
            x_sentences = np.array(sentence_sequences, dtype=np.int32)  # [sentence_i][token_i]
            np.save(x_sentences_path, x_sentences)

        section_ids = text_section_ids[text_i]
        paragraph_ids = text_paragraph_ids[text_i]
        n_paragraphs = len(np.unique(list(zip(section_ids, paragraph_ids)), axis=0))

        if not os.path.exists(x_paragraphs_path):
            print('Saving tokenized paragraph arrays for {}...'.format(book_id))
            x_paragraphs = np.zeros((n_paragraphs, n_paragraph_tokens), dtype=np.int32)  # [paragraph_i][token_i]
            paragraph_i = 0
            paragraph_tokens = []
            last_section_paragraph_id = None
            for i, tokens in enumerate(sentence_tokens):
                section_paragraph_id = (section_ids[i], paragraph_ids[i])
                if last_section_paragraph_id is not None and section_paragraph_id != last_section_paragraph_id:
                    x_paragraphs[paragraph_i] = pad_sequences(tokenizer.texts_to_sequences([split.join(paragraph_tokens)]),
                                                              maxlen=n_paragraph_tokens,
                                                              padding=padding,
                                                              truncating=truncating)[0]
                    paragraph_i += 1
                    paragraph_tokens = []
                paragraph_tokens.extend(tokens)
                last_section_paragraph_id = section_paragraph_id
            x_paragraphs[paragraph_i] = pad_sequences(tokenizer.texts_to_sequences([split.join(paragraph_tokens)]),
                                                      maxlen=n_paragraph_tokens,
                                                      padding=padding,
                                                      truncating=truncating)[0]
            np.save(x_paragraphs_path, x_paragraphs)

        if not os.path.exists(x_paragraph_sentences_path):
            print('Saving tokenized paragraph sentence arrays for {}...'.format(book_id))
            x_paragraph_sentences = np.zeros((n_paragraphs, n_sentences, n_sentence_tokens), dtype=np.int32)  # [paragraph_i][sentence_i][token_i]
            paragraph_i = 0
            sentence_i = 0
            last_section_paragraph_id = None
            for i, sentence_sequence in enumerate(sentence_sequences):
                section_paragraph_id = (section_ids[i], paragraph_ids[i])
                if last_section_paragraph_id is not None and section_paragraph_id != last_section_paragraph_id:
                    paragraph_i += 1
                    sentence_i = 0
                if sentence_i < n_sentences:
                    x_paragraph_sentences[paragraph_i, sentence_i] = sentence_sequence
                sentence_i += 1
                last_section_paragraph_id = section_paragraph_id
            np.save(x_paragraph_sentences_path, x_paragraph_sentences)

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
