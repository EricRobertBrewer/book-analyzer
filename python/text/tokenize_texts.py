import os

import nltk

from python import folders
from python.sites.bookcave import bookcave
from python.text import paragraph_io


def main():
    inputs, _, _, _, _, books_df, _, _, _ = bookcave.get_data({'paragraphs'}, return_meta=True)
    texts = inputs['paragraphs']
    print('All texts: {:d}'.format(len(texts)))

    asins = list(books_df['asin'])
    print('ASINs: {:d}'.format(len(asins)))

    # First, start the Core NLP server:
    # ```
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    # ```
    st = nltk.parse.corenlp.CoreNLPParser()

    if not os.path.exists(folders.AMAZON_KINDLE_SENTENCE_TOKENS_PATH):
        os.mkdir(folders.AMAZON_KINDLE_SENTENCE_TOKENS_PATH)
    force = False
    for text_i, text in enumerate(texts):
        path = os.path.join(folders.AMAZON_KINDLE_SENTENCE_TOKENS_PATH, '{}.txt'.format(asins[text_i]))
        if os.path.exists(path):
            if force:
                os.remove(path)
            else:
                continue

        paragraphs, section_ids, sections = text
        paragraph_sentence_tokens = [[[token['originalText'] or token['word'] for token in sentence['tokens']]
                                      for sentence in st.api_call(paragraph.lower(),
                                                                  properties={
                                                                      'annotators': 'tokenize,ssplit'
                                                                  })['sentences']]
                                     for paragraph in paragraphs]
        section_paragraph_sentence_tokens = [[] for _ in range(len(sections))]
        for paragraph_i, sentence_tokens in enumerate(paragraph_sentence_tokens):
            section_id = section_ids[paragraph_i]
            section_paragraph_sentence_tokens[section_id].append(sentence_tokens)
        paragraph_io.write_formatted_section_paragraph_sentence_tokens(section_paragraph_sentence_tokens, path)


if __name__ == '__main__':
    main()
