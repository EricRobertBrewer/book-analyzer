import datetime
import os
import sys

from gensim.models import KeyedVectors, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import folders
from sites.bookcave import bookcave


class PrintCallback(CallbackAny2Vec):

    def __init__(self, epochs):
        super(PrintCallback, self).__init__()
        self.epochs = epochs
        self.epoch = 0

    def on_epoch_begin(self, model):
        now = datetime.datetime.now()
        self.epoch += 1
        print('{} Starting epoch {:d}/{:d}...'.format(str(now), self.epoch, self.epochs))


def load_vectors(fname):
    vectors_path = os.path.join(folders.VECTORS_PATH, fname)
    return KeyedVectors.load(vectors_path)


def main(argv):
    if len(argv) < 5 or len(argv) > 6:
        raise ValueError('Usage: <model_name> <vector_size> <max_vocab_size> <epochs> <window> [min_count]')
    model_name = argv[0]
    vector_size = int(argv[1])
    max_vocab_size = int(argv[2])  # The maximum size of the vocabulary.
    epochs = int(argv[3])
    window = int(argv[4])
    min_count = 4
    if len(argv) > 5:
        min_count = int(argv[5])

    # Load data.
    print('Retrieving texts...')
    subset_ratio = 1.
    subset_seed = 1
    min_len = 256
    max_len = 4096
    min_tokens = 6
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'tokens'},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens)
    text_paragraph_tokens, _ = zip(*inputs['tokens'])
    print('Retrieved {:d} texts.'.format(len(text_paragraph_tokens)))

    # Tokenize.
    print('Tokenizing...')
    all_paragraph_tokens = []
    for paragraph_tokens in text_paragraph_tokens:
        for tokens in paragraph_tokens:
            all_paragraph_tokens.append(tokens)
    print('Done.')

    # Create model.
    print('Creating model...')
    workers = 8
    if model_name == 'word2vec':
        model = Word2Vec(all_paragraph_tokens,
                         size=vector_size,
                         window=window,
                         min_count=min_count,
                         max_vocab_size=max_vocab_size,
                         workers=workers)
    else:
        raise ValueError('Unknown model name: `{}`'.format(model_name))
    print('Done.')

    # Train word vectors.
    print('Training model...')
    print_callback = PrintCallback(epochs)
    model.train(all_paragraph_tokens,
                total_examples=len(all_paragraph_tokens),
                epochs=epochs,
                callbacks=[print_callback])
    print('Done.')

    # Save.
    print('Saving vectors...')
    if not os.path.exists(folders.VECTORS_PATH):
        os.mkdir(folders.VECTORS_PATH)
    fname = '{}_{:d}_{:d}d_{:d}w_{:d}min_{:d}e.wv'.format(model_name,
                                                          len(text_paragraph_tokens),
                                                          vector_size,
                                                          window,
                                                          min_count,
                                                          epochs)
    vectors_path = os.path.join(folders.VECTORS_PATH, fname)
    model.wv.save(vectors_path)
    print('Saved vectors to `{}`.'.format(vectors_path))


if __name__ == '__main__':
    main(sys.argv[1:])
