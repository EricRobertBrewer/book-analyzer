import os
import nltk

import folders
from sites.bookcave import bookcave
from text import paragraph_io


def is_valid(sections, section_paragraphs):
    if len(sections) == 0:
        return False
    section_sizes = [len(paragraphs) for paragraphs in section_paragraphs]
    sum_paragraphs = sum(section_sizes)
    if sum_paragraphs == 0:
        return False
    if len(sections) > sum_paragraphs:
        return False
    if max(section_sizes) > sum_paragraphs // 2:
        return False

    return True


def write_paragraphs_tokens(valid_asins, valid_texts, st, force=False):
    for i, text in enumerate(valid_texts):
        sections, section_paragraphs = text
        path = os.path.join(folders.AMAZON_KINDLE_TEXT_PATH, valid_asins[i], folders.FNAME_TEXT_PARAGRAPHS_TOKENS)
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

    valid_asins = []
    valid_texts = []
    for i, text in enumerate(texts):
        sections, section_paragraphs = text
        if is_valid(sections, section_paragraphs):
            valid_asins.append(books_df.iloc[i]['asin'])
            valid_texts.append(text)
    print('Valid texts: {:d}'.format(len(valid_texts)))

    # First, start the Core NLP server:
    # ```
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    # ```
    st = nltk.parse.corenlp.CoreNLPParser()

    write_paragraphs_tokens(valid_asins, valid_texts, st, force=False)


if __name__ == '__main__':
    main()
