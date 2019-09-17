import os

import numpy as np

from classification import shared_parameters
import folders
from sites.bookcave import bookcave


def get_ids_fname(
        n_texts=7491,
        paragraph_min_len=shared_parameters.DATA_PARAGRAPH_MIN_LEN,
        paragraph_max_len=shared_parameters.DATA_PARAGRAPH_MAX_LEN,
        sentence_min_len=shared_parameters.DATA_SENTENCE_MIN_LEN,
        sentence_max_len=shared_parameters.DATA_SENTENCE_MAX_LEN,
        min_tokens=shared_parameters.DATA_MIN_TOKENS
):
    return 'ids_{:d}_{:d}-{:d}p_{:d}-{:d}s_{:d}t.txt'.format(n_texts,
                                                             paragraph_min_len,
                                                             paragraph_max_len,
                                                             sentence_min_len,
                                                             sentence_max_len,
                                                             min_tokens)


def main():
    # Find union between paragraph and sentence data sets with the given parameters.
    print('Retrieving sentence tokens...')
    paragraph_min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    paragraph_max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    sentence_min_len = shared_parameters.DATA_SENTENCE_MIN_LEN
    sentence_max_len = shared_parameters.DATA_SENTENCE_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    inputs, _, _, _, book_ids, _, _, _, _ = \
        bookcave.get_data({'sentence_tokens'},
                          min_len=sentence_min_len,
                          max_len=sentence_max_len,
                          min_tokens=min_tokens,
                          return_meta=True)
    print('Done.')

    sentence_token_inputs = inputs['sentence_tokens']
    text_sentence_tokens, text_section_ids, text_paragraph_ids = zip(*sentence_token_inputs)
    paragraph_counts = [len(np.unique(list(zip(text_section_ids[i], text_paragraph_ids[i])), axis=0))
                        for i in range(len(text_section_ids))]
    book_ids = sorted([book_id
                       for i, book_id in enumerate(book_ids)
                       if paragraph_min_len <= paragraph_counts[i] <= paragraph_max_len])

    # Write the union to a file.
    print('Writing the union of paragraph and sentence IDs...')
    script_name = os.path.basename(__file__)
    base_name = script_name[:script_name.rindex('.')]
    if not os.path.exists(folders.LOGS_PATH):
        os.mkdir(folders.LOGS_PATH)
    logs_path = os.path.join(folders.LOGS_PATH, base_name)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    fname = get_ids_fname(len(book_ids), paragraph_min_len, paragraph_max_len, sentence_min_len, sentence_max_len, min_tokens)
    path = os.path.join(logs_path, fname)
    with open(path, 'w', encoding='utf-8') as fd:
        fd.write('{:d}\n'.format(len(book_ids)))
        for book_id in book_ids:
            fd.write('{}\n'.format(book_id))
    print('Done.')


if __name__ == '__main__':
    main()
