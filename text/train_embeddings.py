import os
import gensim
import nltk

from sites.bookcave import bookcave
from text import preprocessing

EMBEDDINGS_FOLDER = 'embeddings'


def save_trained_vectors(documents, names, size=50, window=8, min_count=2, max_vocab_size=None, workers=8, epochs=1, verbose=False):
    if verbose:
        print('Creating model...')
    model = gensim.models.Word2Vec(documents,
                                   size=size,
                                   window=window,
                                   min_count=min_count,
                                   max_vocab_size=max_vocab_size,
                                   workers=workers)
    if verbose:
        print('Training model...')
    model.train(documents, total_examples=len(documents), epochs=epochs)
    
    if verbose:
        print('Saving vectors...')
    if isinstance(names, str):
        name = names
    else:
        name = '_'.join(names)
    fname = 'vectors_{}_{:d}d_{:d}w_{:d}min_{:d}e_{:d}.wv'.format(name, size, window, min_count, epochs, max_vocab_size)
    model.wv.save(os.path.join(EMBEDDINGS_FOLDER, fname))
    return fname


def load_vectors(fname):
    return gensim.models.KeyedVectors.load(os.path.join(EMBEDDINGS_FOLDER, fname))


def save_trained_doc_model(documents, names, vector_size=50, window=8, min_count=2, workers=8, epochs=1, verbose=False):
    if isinstance(names, str):
        name = names
    else:
        name = '_'.join(names)

    if verbose:
        print('Tagging documents...')
    tagged_docs = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    if verbose:
        print('Creating model...')
    model = gensim.models.Doc2Vec(tagged_docs,
                                  vector_size=vector_size,
                                  window=window,
                                  min_count=min_count,
                                  workers=workers)
    if verbose:
        print('Training model...')
    model.train(tagged_docs, total_examples=len(tagged_docs), epochs=epochs)
    if verbose:
        print('Saving entire model...')
    fname = 'docmodel_{}_{:d}d_{:d}w_{:d}min_{:d}e.model'.format(name, vector_size, window, min_count, epochs)
    model.save(os.path.join(EMBEDDINGS_FOLDER, fname))
    return fname


def load_doc_model(fname):
    return gensim.models.Doc2Vec.load(os.path.join(EMBEDDINGS_FOLDER, fname))


def main():
    print('Loading BookCave data...')
    inputs, _, _, _ = bookcave.get_data({'text'},
                                        text_source='book',
                                        text_min_len=6)
    texts = inputs['text']

    print('Splitting text files into lines...')
    text_lines = [text.split('\n') for text in texts]

    # Do pre-processing.
    tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    kwargs = {
        'lower': True,
        'endings': {'.', '?', ')', '!', ':', '-', '"', ';', ',', "'"},
        'min_len': 6,
        'normal': True
    }
    print('Pre-processing lines...')
    processed_lines = list()
    for lines in text_lines:
        processed_lines.extend(list(preprocessing.process_lines(tokenizer, lines, sentences=False, **kwargs)))
    print('Pre-processing sentences...')
    processed_sentences = list()
    for lines in text_lines:
        processed_sentences.extend(list(preprocessing.process_lines(tokenizer, lines, sentences=True, **kwargs)))

    # Hyper-parameters.
    data_size = str(len(texts))
    tokenizer_name = 'treebank'
    vector_size = 150
    max_vocab_size = 40000
    epochs = 32
    verbose = True

    # Train word vectors.
    line_fname = save_trained_vectors(processed_lines,
                                      ['line', data_size, tokenizer_name],
                                      size=vector_size,
                                      max_vocab_size=max_vocab_size,
                                      epochs=epochs,
                                      verbose=verbose)
    print('Saved `line` vectors to `{}`.'.format(line_fname))
    sentence_fname = save_trained_vectors(processed_sentences,
                                          ['sentence', data_size, tokenizer_name],
                                          size=vector_size,
                                          max_vocab_size=max_vocab_size,
                                          epochs=epochs,
                                          verbose=verbose)
    print('Saved `sentence` vectors to `{}`.'.format(sentence_fname))

    # Train doc2vec embeddings.
    line_doc_fname = save_trained_doc_model(processed_lines,
                                            ['line', data_size, tokenizer_name],
                                            vector_size=vector_size,
                                            epochs=epochs,
                                            verbose=verbose)
    print('Saved `line` doc2vec model to `{}`.'.format(line_doc_fname))
    sentence_doc_fname = save_trained_doc_model(processed_sentences,
                                                ['sentence', data_size, tokenizer_name],
                                                vector_size=vector_size,
                                                epochs=epochs,
                                                verbose=verbose)
    print('Saved `sentence` doc2vec model to `{}`.'.format(sentence_doc_fname))


if __name__ == '__main__':
    main()
