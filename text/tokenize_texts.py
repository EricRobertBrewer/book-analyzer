import os
import nltk

import folders
from sites.bookcave import bookcave
from text import paragraph_io


def write_paragraphs_tokens(asins, texts, st, force=False):
    for i, text in enumerate(texts):
        sections, section_paragraphs = text
        path = os.path.join(folders.AMAZON_KINDLE_TEXT_PATH, asins[i], folders.FNAME_TEXT_PARAGRAPHS_TOKENS)
        if os.path.exists(path):
            if force:
                os.remove(path)
            else:
                continue

        section_paragraph_tokens = [[list(st.tokenize(paragraph.lower())) for paragraph in paragraphs]
                                    for paragraphs in section_paragraphs]
        paragraph_io.write_formatted_section_paragraph_tokens(section_paragraph_tokens, path)


def main():
    inputs, _, _, _,\
        _, books_df, _, _, _\
        = bookcave.get_data({'text'}, text_source='paragraphs', return_meta=True)
    texts = inputs['text']
    print('All texts: {:d}'.format(len(texts)))

    asins = list(books_df['asin'])
    print('ASINs: {:d}'.format(len(asins)))

    # First, start the Core NLP server:
    # ```
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    # ```
    st = nltk.parse.corenlp.CoreNLPParser()

    write_paragraphs_tokens(asins, texts, st, force=False)


if __name__ == '__main__':
    main()
