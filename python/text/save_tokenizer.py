import os
import pickle
import sys

from keras.preprocessing.text import Tokenizer

from python import folders
from python.sites.bookcave import bookcave
from python.util import shared_parameters


def main(argv):
    source_mode = argv[0]
    remove_stopwords = bool(argv[1])

    # Load data.
    print('Retrieving texts...')
    if source_mode == 'paragraph':
        source = 'paragraph_tokens'
        min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
        max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    else:  # args.source_mode == 'sentence':
        source = 'sentence_tokens'
        min_len = shared_parameters.DATA_SENTENCE_MIN_LEN
        max_len = shared_parameters.DATA_SENTENCE_MAX_LEN
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    return_overall = shared_parameters.DATA_RETURN_OVERALL
    inputs, _, _, _ = \
        bookcave.get_data({source},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          remove_stopwords=remove_stopwords,
                          categories_mode=categories_mode,
                          return_overall=return_overall)
    text_source_tokens = list(zip(*inputs[source]))[0]
    print('Retrieved {:d} texts.'.format(len(text_source_tokens)))

    # Tokenize.
    print('Tokenizing...')
    max_words = shared_parameters.TEXT_MAX_WORDS
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_sources = []
    for source_tokens in text_source_tokens:
        for tokens in source_tokens:
            all_sources.append(split.join(tokens))
    tokenizer.fit_on_texts(all_sources)

    # Save.
    file_name = 'tokenizer-{}{}.pickle'.format(source_mode, '-nostop' if remove_stopwords else '')
    with open(os.path.join(folders.TOKENIZERS_PATH, file_name), 'wb') as fd:
        pickle.dump(tokenizer, fd, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(sys.argv[1:])
