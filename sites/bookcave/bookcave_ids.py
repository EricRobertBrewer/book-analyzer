import os

from classification import shared_parameters
import folders
from sites.bookcave import bookcave


def get_fname(
        paragraph_min_len=shared_parameters.DATA_PARAGRAPH_MIN_LEN,
        paragraph_max_len=shared_parameters.DATA_PARAGRAPH_MAX_LEN,
        sentence_min_len=shared_parameters.DATA_SENTENCE_MIN_LEN,
        sentence_max_len=shared_parameters.DATA_SENTENCE_MAX_LEN,
        min_tokens=shared_parameters.DATA_MIN_TOKENS
):
    return 'ids_{:d}-{:d}p_{:d}-{:d}s_{:d}t.txt'.format(paragraph_min_len,
                                                        paragraph_max_len,
                                                        sentence_min_len,
                                                        sentence_max_len,
                                                        min_tokens)


def main():
    # Find union between paragraph and sentence data sets with the given parameters.
    print('Retrieving paragraph tokens...')
    paragraph_min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    paragraph_max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    _, _, _, _, paragraph_book_ids, _, _, _, _ = \
        bookcave.get_data({'paragraph_tokens'},
                          min_len=paragraph_min_len,
                          max_len=paragraph_max_len,
                          min_tokens=min_tokens,
                          return_meta=True)
    print('Done.')
    print('Retrieving sentence tokens...')
    sentence_min_len = shared_parameters.DATA_SENTENCE_MIN_LEN
    sentence_max_len = shared_parameters.DATA_SENTENCE_MAX_LEN
    _, _, _, _, sentence_book_ids, _, _, _, _ = \
        bookcave.get_data({'sentence_tokens'},
                          min_len=sentence_min_len,
                          max_len=sentence_max_len,
                          min_tokens=min_tokens,
                          return_meta=True)
    print('Done.')
    sentence_book_id_set = set(sentence_book_ids)
    book_ids = sorted([book_id for book_id in paragraph_book_ids if book_id in sentence_book_id_set])

    # Write the union to a file.
    print('Writing the union of paragraph and sentence IDs...')
    script_name = os.path.basename(__file__)
    base_name = script_name[:script_name.rindex('.')]
    if not os.path.exists(folders.LOGS_PATH):
        os.mkdir(folders.LOGS_PATH)
    logs_path = os.path.join(folders.LOGS_PATH, base_name)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    fname = get_fname(paragraph_min_len, paragraph_max_len, sentence_min_len, sentence_max_len, min_tokens)
    path = os.path.join(logs_path, fname)
    with open(path, 'w', encoding='utf-8') as fd:
        fd.write('{:d}\n'.format(len(book_ids)))
        for book_id in book_ids:
            fd.write('{}\n'.format(book_id))
    print('Done.')


if __name__ == '__main__':
    main()
