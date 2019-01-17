import gensim
import nltk
import bookcave


def get_documents(text_lines, sentences=False):
    for lines in text_lines:
        for line in lines:
            if sentences:
                for sentence in nltk.sent_tokenize(line):
                    yield gensim.utils.simple_preprocess(sentence)
            else:
                yield gensim.utils.simple_preprocess(line)


def save_trained_vectors(documents, name, size=150, window=8, min_count=2, workers=8, epochs=1, verbose=False):
    if verbose:
        print('Creating model...')
    model = gensim.models.Word2Vec(documents,
                                   size=size,
                                   window=window,
                                   min_count=min_count,
                                   workers=workers)
    if verbose:
        print('Training model...')
    model.train(documents, total_examples=len(documents), epochs=epochs)
    if verbose:
        print('Saving vectors...')
    fname = 'vectors_{}_d{}_w{}_min{}.wv'.format(name, size, window, min_count)
    model.wv.save(fname)
    return fname


def load_vectors(fname):
    return gensim.models.KeyedVectors.load(fname)


def save_trained_doc_model(documents, name, vector_size=150, window=8, min_count=2, workers=8, epochs=1, verbose=False):
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
    fname = 'docmodel_{}_d{}_w{}_min{}.model'.format(name, vector_size, window, min_count)
    model.save(fname)
    return fname


def load_doc_model(fname):
    return gensim.models.Doc2Vec.load(fname)


def main():
    texts, y, categories, levels = bookcave.get_data()
    text_lines = bookcave.get_text_lines(texts)
    lines = [line for line in get_documents(text_lines, sentences=False)]
    sentences = [sentence for sentence in get_documents(text_lines, sentences=True)]
    line_fname = save_trained_vectors(lines, 'line', epochs=10, verbose=True)
    print('Saved `line` vectors to `{}`.'.format(line_fname))
    sentence_fname = save_trained_vectors(sentences, 'sentence', epochs=10, verbose=True)
    print('Saved `sentence` vectors to `{}`.'.format(sentence_fname))
    line_doc_fname = save_trained_doc_model(lines, 'line', epochs=10, verbose=True)
    print('Saved `line` doc2vec model to `{}`.'.format(line_doc_fname))
    sentence_doc_fname = save_trained_doc_model(sentences, 'line', epochs=10, verbose=True)
    print('Saved `line` doc2vec model to `{}`.'.format(sentence_doc_fname))


if __name__ == '__main__':
    main()
