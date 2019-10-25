from io import open
import os

import numpy as np
import pandas as pd

import folders
from text import paragraph_io


CATEGORY_INDEX_CRUDE_HUMOR_LANGUAGE = 0
CATEGORY_INDEX_DRUG_ALCOHOL_TOBACCO_USE = 1
CATEGORY_INDEX_KISSING = 2
CATEGORY_INDEX_PROFANITY = 3
CATEGORY_INDEX_NUDITY = 4
CATEGORY_INDEX_SEX_AND_INTIMACY = 5
CATEGORY_INDEX_VIOLENCE_AND_HORROR = 6
CATEGORY_INDEX_GAY_LESBIAN_CHARACTERS = 7
CATEGORY_INDEX_OVERALL = 8
CATEGORIES = [
    'crude_humor_language',
    'drug_alcohol_tobacco_use',
    'kissing',
    'profanity',
    'nudity',
    'sex_and_intimacy',
    'violence_and_horror',
    'gay_lesbian_characters',
    'overall'
]
CATEGORY_NAMES = {
    'crude_humor_language': 'Crude Humor/Language',
    'drug_alcohol_tobacco_use': 'Drug, Alcohol & Tobacco Use',
    'kissing': 'Kissing',
    'profanity': 'Profanity',
    'nudity': 'Nudity',
    'sex_and_intimacy': 'Sex and Intimacy',
    'violence_and_horror': 'Violence and Horror',
    'gay_lesbian_characters': 'Gay/Lesbian Characters',
    'overall': 'Overall'
}
CATEGORY_LEVELS = {
    'soft': [[
        'None',
        'Mild crude humor',
        'Moderate crude humor/language|Significant crude humor/language',
        'Extensive crude humor/language'
    ], [
        'None',
        'Mild substance use|Some substance use',
        'Moderate substance use by adults and/or some use by minors|Significant substance use',
        'Extensive substance abuse'
    ], [
        'None',
        'Mild kissing|Passionate kissing'
    ], [
        'None',
        'Mild language',
        'Some profanity (6 to 40)|Moderate profanity (41 to 100)',
        'Significant profanity (101 to 200)|Significant profanity (201 to 500)|Extensive profanity (501+)'
    ], [
        'None',
        'Brief (nonsexual) nudity|Brief nudity',
        'Some nudity',
        'Extensive nudity'
    ], [
        'None',
        'Mild sensuality',
        'Non-graphic sexual references|Non-detailed fade-out sensuality|Fade-out intimacy with details or significant sexual discussion',
        'Semi-detailed onscreen love scenes|Detailed onscreen love scenes|Repeated graphic sex|Menage or BDSM sex'
    ], [
        'None',
        'Mild (nonsexual) violence or horror|Some violence or horror',
        'Moderate violence or horror',
        'Graphic violence or horror|Extended gruesome and depraved violence or horror'
    ], [
        'None',
        'Minor gay/lesbian characters or elements',
        'Prominent gay/lesbian character(s)'
    ], [
        'All Ages',
        'Mild',
        'Moderate',
        'Adult'
    ]],
    'medium': [[
        'None',
        'Mild crude humor',
        'Moderate crude humor/language',
        'Significant crude humor/language',
        'Extensive crude humor/language'
    ], [
        'None',
        'Mild substance use',
        'Some substance use',
        'Moderate substance use by adults and/or some use by minors',
        'Significant substance use',
        'Extensive substance abuse'
    ], [
        'None',
        'Mild kissing',
        'Passionate kissing'
    ], [
        'None',
        'Mild language',
        'Some profanity (6 to 40)',
        'Moderate profanity (41 to 100)',
        'Significant profanity (101 to 200)|Significant profanity (201 to 500)',
        'Extensive profanity (501+)'
    ], [
        'None',
        'Brief (nonsexual) nudity',
        'Brief nudity',
        'Some nudity',
        'Extensive nudity'
    ], [
        'None',
        'Mild sensuality',
        'Non-graphic sexual references|Non-detailed fade-out sensuality',
        'Fade-out intimacy with details or significant sexual discussion',
        'Semi-detailed onscreen love scenes|Detailed onscreen love scenes',
        'Repeated graphic sex|Menage or BDSM sex'
    ], [
        'None',
        'Mild (nonsexual) violence or horror',
        'Some violence or horror',
        'Moderate violence or horror',
        'Graphic violence or horror',
        'Extended gruesome and depraved violence or horror'
    ], [
        'None',
        'Minor gay/lesbian characters or elements',
        'Prominent gay/lesbian character(s)'
    ], [
        'All Ages',
        'Mild',
        'Mild+',
        'Moderate',
        'Moderate+',
        'Adult'
        'Adult+'
    ]],
    'hard': [[
        'None',
        'Mild crude humor',
        'Moderate crude humor/language',
        'Significant crude humor/language',
        'Extensive crude humor/language'
    ], [
        'None',
        'Mild substance use',
        'Some substance use',
        'Moderate substance use by adults and/or some use by minors',
        'Significant substance use',
        'Extensive substance abuse'
    ], [
        'None',
        'Mild kissing',
        'Passionate kissing'
    ], [
        'None',
        'Mild language',
        'Some profanity (6 to 40)',
        'Moderate profanity (41 to 100)',
        'Significant profanity (101 to 200)',
        'Significant profanity (201 to 500)',
        'Extensive profanity (501+)'
    ], [
        'None',
        'Brief (nonsexual) nudity',
        'Brief nudity',
        'Some nudity',
        'Extensive nudity'
    ], [
        'None',
        'Mild sensuality',
        'Non-graphic sexual references',
        'Non-detailed fade-out sensuality',
        'Fade-out intimacy with details or significant sexual discussion',
        'Semi-detailed onscreen love scenes',
        'Detailed onscreen love scenes',
        'Repeated graphic sex',
        'Menage or BDSM sex'
    ], [
        'None',
        'Mild (nonsexual) violence or horror',
        'Some violence or horror',
        'Moderate violence or horror',
        'Graphic violence or horror',
        'Extended gruesome and depraved violence or horror'
    ], [
        'None',
        'Minor gay/lesbian characters or elements',
        'Prominent gay/lesbian character(s)'
    ], [
        'All Ages',
        'Mild',
        'Mild+',
        'Moderate',
        'Moderate+',
        'Adult'
        'Adult+'
    ]]
}
LEVEL_SPLIT = '|'


def is_between(value, _min=None, _max=None):
    if _min is not None and value < _min:
        return False
    if _max is not None and value > _max:
        return False
    return True


def get_book(asin, min_len=None, max_len=None):
    path = os.path.join(folders.AMAZON_KINDLE_BOOK_PATH, '{}.txt'.format(asin))
    if not os.path.exists(path):
        return None

    # Validate file length.
    with open(path, 'r', encoding='utf-8') as fd:
        book = fd.read()
    if not is_between(len(book), min_len, max_len):
        return None
    return book


def get_preview(asin, min_len=None, max_len=None):
    path = os.path.join(folders.AMAZON_KINDLE_PREVIEW_PATH, '{}.txt'.format(asin))
    if not os.path.exists(path):
        return None

    # Validate file length.
    with open(path, 'r', encoding='utf-8') as fd:
        preview = fd.read()
    if not is_between(len(preview), min_len, max_len):
        return None
    return preview


def get_paragraphs(asin, min_len=None, max_len=None):
    path = os.path.join(folders.AMAZON_KINDLE_PARAGRAPHS_PATH, '{}.txt'.format(asin))
    if not os.path.exists(path):
        return None

    sections, section_paragraphs = paragraph_io.read_formatted_section_paragraphs(path)
    all_paragraphs, section_ids = [], []
    for section_i, paragraphs in enumerate(section_paragraphs):
        for paragraph_i, paragraph in enumerate(paragraphs):
            all_paragraphs.append(paragraph)
            section_ids.append(section_i)
    if not is_between(len(all_paragraphs), min_len, max_len):
        return None
    return all_paragraphs, section_ids, sections


def get_paragraph_tokens(asin, min_len=None, max_len=None, min_tokens=None, max_tokens=None):
    path = os.path.join(folders.AMAZON_KINDLE_PARAGRAPH_TOKENS_PATH, '{}.txt'.format(asin))
    if not os.path.exists(path):
        return None

    section_paragraph_tokens = paragraph_io.read_formatted_section_paragraph_tokens(path)
    all_paragraph_tokens, section_ids = [], []
    for section_i, paragraph_tokens in enumerate(section_paragraph_tokens):
        for paragraph_i, tokens in enumerate(paragraph_tokens):
            if not is_between(len(tokens), min_tokens, max_tokens):
                continue
            all_paragraph_tokens.append(tokens)
            section_ids.append(section_i)
    if len(all_paragraph_tokens) == 0 or not is_between(len(all_paragraph_tokens), min_len, max_len):
        return None
    return all_paragraph_tokens, section_ids


def get_sentence_tokens(asin, min_len=None, max_len=None, min_tokens=None, max_tokens=None):
    path = os.path.join(folders.AMAZON_KINDLE_SENTENCE_TOKENS_PATH, '{}.txt'.format(asin))
    if not os.path.exists(path):
        return None

    section_paragraph_sentence_tokens = paragraph_io.read_formatted_section_paragraph_sentence_tokens(path)
    all_sentence_tokens, section_ids, paragraph_ids = [], [], []
    for section_i, paragraph_sentence_tokens in enumerate(section_paragraph_sentence_tokens):
        for paragraph_i, sentence_tokens in enumerate(paragraph_sentence_tokens):
            if len(sentence_tokens) == 1 and not is_between(len(sentence_tokens[0]), min_tokens, max_tokens):
                continue
            for tokens in sentence_tokens:
                all_sentence_tokens.append(tokens)
                section_ids.append(section_i)
                paragraph_ids.append(paragraph_i)
    if len(all_sentence_tokens) == 0 or not is_between(len(all_sentence_tokens), min_len, max_len):
        return None
    return all_sentence_tokens, section_ids, paragraph_ids


def get_input(source, asin, min_len=None, max_len=None, min_tokens=None, max_tokens=None):
    if source == 'book':
        return get_book(asin, min_len, max_len)
    if source == 'preview':
        return get_preview(asin, min_len, max_len)
    if source == 'paragraphs':
        return get_paragraphs(asin, min_len, max_len)
    if source == 'paragraph_tokens':
        return get_paragraph_tokens(asin, min_len, max_len, min_tokens, max_tokens)
    if source == 'sentence_tokens':
        return get_sentence_tokens(asin, min_len, max_len, min_tokens, max_tokens)
    raise ValueError('Unknown source: `{}`.'.format(source))


def get_data(
        sources,
        only_ids=None,
        subset_ratio=None,
        subset_seed=None,
        min_len=None,
        max_len=None,
        min_tokens=None,
        max_tokens=None,
        categories_mode='soft',
        return_overall=True,
        combine_ratings='max',
        return_meta=False,
        verbose=False):
    """
    Retrieve text with corresponding labels for books in the BookCave database.
    :param sources: set of str {'book', 'preview', 'paragraphs', 'paragraph_tokens', 'sentence_tokens'}
        The type(s) of text to be retrieved.
        When 'book', the entire raw book texts will be returned.
        When 'preview', the first few chapters of books will be returned.
        When 'paragraphs', the sections and paragraphs will be returned (as tuples).
        When 'paragraph_tokens', the tokens for each paragraph will be returned.
        When 'sentence_tokens', the tokens for each sentence for each paragraph will be returned.
    :param only_ids: iterable of str, optional
        Filter the returned books by a set
    :param subset_ratio: float, optional
        Used to specify that only a subset of the data should be returned.
        Ignored when `only_ids` is provided.
    :param subset_seed: integer, optional
        Used to seed the random subset.
    :param min_len: int, optional
        When `source` is 'book' or 'preview', this is the minimum file length of text files that will be returned.
        When `source` is 'paragraphs' or 'paragraph_tokens', this is the minimum number of paragraphs in each text.
        When `source` is 'sentence_tokens', this is the minimum number of sentences in each text.
    :param max_len: int, optional
        When `source` is 'book' or 'preview', this is the maximum file length of text files that will be returned.
        When `source` is 'paragraphs' or 'paragraph_tokens', this is the maximum number of paragraphs in each text.
        When `source` is 'sentence_tokens', this is the maximum number of sentences in each text.
    :param min_tokens: int, optional
        The minimum number of tokens in each paragraph. Paragraphs with fewer tokens will not be returned.
        Only applied when `source` is 'paragraph_tokens' or 'sentence_tokens'.
    :param max_tokens: int, optional
        The maximum number of tokens in each paragraph. Paragraphs with more tokens will not be returned.
        Only applied when `source` is 'tokens'.
    :param categories_mode: string {'soft' (default), 'medium', 'hard'}
        The flexibility of rating levels within categories.
        When 'soft', all levels which would yield the same base overall rating (without a '+') will be merged.
        When 'medium', all levels which would yield the same overall rating will be merged.
        When 'hard', no levels will be merged.
    :param return_overall: boolean, default True
        When `True`, each book's overall rating will be returned as an additional category.
    :param combine_ratings: string {'max' (default), 'avg ceil', 'avg floor'}
        The method by which multiple ratings for a single book will be combined.
        When `'max'`, the maximum among all rating levels per category per book will be returned.
        When `'avg ceil'`, the ceiling of the average of all rating levels per category per book will be returned.
        When `'avg floor'`, the floor of the average of all rating levels per category per book will be returned.
    :param return_meta: boolean, default False
        When `True`, all meta data will be returned.
    :param verbose: boolean, default False
        When `True`, function progress will be printed to the console.
    :return:
        Always:
            inputs (dict):                  A dict containing raw texts, section/section-paragraph tuples,
                                                section-paragraph-token lists,
                                                or section-paragraph-sentence-token lists of books.
            Y (np.ndarray):                 Level (label) for the corresponding text/images in up to 8 categories.
            categories ([str]):             Names of categories.
            category_levels ([[str]]):      Names of levels per category.
        Only when `return_meta` is set to `True`:
            book_ids (np.array)             Alphabetically-sorted array of book IDs parallel with `inputs`.
            books_df (pd.DataFrame):        Metadata for books which have been rated AND have text.
            ratings_df (pd.DataFrame):      Metadata for book ratings (unused).
            levels_df (pd.DataFrame):       Metadata for book rating levels.
            categories_df (pd.DataFrame):   Metadata for categories (contains a description for each rating level).
    """
    # Read all of the data from the BookCave database.
    all_books_df = pd.read_csv(folders.CONTENT_BOOKCAVE_BOOKS_CSV_PATH, encoding='utf-8')
    all_ratings_df = pd.read_csv(folders.CONTENT_BOOKCAVE_BOOK_RATINGS_CSV_PATH, encoding='utf-8')
    all_levels_df = pd.read_csv(folders.CONTENT_BOOKCAVE_BOOK_RATING_LEVELS_CSV_PATH, encoding='utf-8')

    # Consider only books which have at least one rating.
    rated_books_df = all_books_df[all_books_df['community_ratings_count'] > 0]

    # Create a subset, if needed.
    if only_ids is None and subset_ratio:
        np.random.seed(subset_seed)
        only_ids = np.random.choice(rated_books_df['id'], int(subset_ratio*len(rated_books_df)), replace=False)

    # Filter by `only_ids`, if present.
    if only_ids is not None:
        rated_books_df = rated_books_df[rated_books_df['id'].isin(set(only_ids))]

    # Determine which books will be retrieved.
    if verbose:
        print('Collecting inputs...')
    inputs = dict()
    for source in sources:
        inputs[source] = []
    book_id_set = set()
    for _, rated_book_row in rated_books_df.iterrows():
        # Skip books without a known ASIN.
        asin = rated_book_row['asin']
        if asin is None:
            continue

        # Ensure that all selected sources exist.
        has_all_sources = True
        book_inputs = dict()
        for source in sources:
            book_input = get_input(source, asin, min_len, max_len, min_tokens, max_tokens)
            if book_input is None:
                has_all_sources = False
                break
            book_inputs[source] = book_input

        if has_all_sources:
            for source, book_input in book_inputs.items():
                inputs[source].append(book_input)
            book_id_set.add(rated_book_row['id'])

    # Consider only books for which text has been collected.
    books_df = rated_books_df[rated_books_df['id'].isin(book_id_set)]
    books_df = books_df.sort_values(by='id')

    # Map book IDs to indices.
    book_ids = list(books_df['id'])
    book_id_to_index = {book_id: i for i, book_id in enumerate(book_ids)}

    # Get smaller ratings and levels DataFrames.
    ratings_df = all_ratings_df[all_ratings_df['book_id'].isin(book_id_to_index.keys())]
    levels_df = all_levels_df[all_levels_df['book_id'].isin(book_id_to_index.keys())]

    # Extract category data.
    if categories_mode not in {'hard', 'medium', 'soft'}:
        raise ValueError('Unknown value for `categories_mode`: `{}`'.format(categories_mode))
    categories_path = os.path.join(folders.BOOKCAVE_CATEGORIES_PATH, '{}.tsv'.format(categories_mode))
    categories_df = pd.read_csv(categories_path, sep='\t')

    # Determine category names and levels.
    if return_overall:
        categories = CATEGORIES
        category_levels = CATEGORY_LEVELS[categories_mode]
    else:
        categories = CATEGORIES[:CATEGORY_INDEX_OVERALL]
        category_levels = CATEGORY_LEVELS[categories_mode][:CATEGORY_INDEX_OVERALL]

    # Map each level name (and parts of the level name, i.e., between bars `|`) to its index within its own category.
    # The full level name (including bars `|`) are necessary when iterating through the category DataFrame.
    # The parts are necessary when iterating through the rating levels DataFrame.
    category_level_to_index = []
    for j in range(len(categories)):
        level_to_index = dict()
        for index, level in enumerate(category_levels[j]):
            level_to_index[level] = index
            for level_part in level.split(LEVEL_SPLIT):
                level_to_index[level_part] = index
        category_level_to_index.append(level_to_index)

    # Map each level name part to its category index.
    # The full level name isn't necessary because we don't use this map when iterating through the category DataFrame,
    # since the category is already included there.
    level_to_category_index = dict()
    for j, levels in enumerate(category_levels):
        if j == CATEGORY_INDEX_OVERALL:
            continue
        for level in levels:
            if level == 'None':
                continue
            for level_part in level.split(LEVEL_SPLIT):
                level_to_category_index[level_part] = j

    # Populate Y by reading from the rating levels DataFrame.
    if verbose:
        print('Extracting labels for {} books...'.format(len(book_ids)))
    if combine_ratings == 'max':
        # For each book, take the maximum rating level in each category.
        Y = np.zeros((len(categories), len(book_ids)), dtype=np.int32)
        for _, level_row in levels_df.iterrows():
            book_index = book_id_to_index[level_row['book_id']]
            level = level_row['title']
            j = level_to_category_index[level]
            level_index = category_level_to_index[j][level]
            Y[j, book_index] = max(Y[j, book_index], level_index)
    elif combine_ratings == 'avg ceil' or combine_ratings == 'avg floor':
        # For each category, calculate the average rating for each book.
        Y_cont = np.zeros((len(categories), len(book_ids)), dtype=np.float32)
        # First, add all levels together for each book.
        for _, level_row in levels_df.iterrows():
            book_index = book_id_to_index[level_row['book_id']]
            level = level_row['title']
            j = level_to_category_index[level]
            level_index = category_level_to_index[j][level]
            Y_cont[j, book_index] += level_index * level_row['count']
        # Then calculate the average level for each book by dividing by the number of ratings for that book.
        for _, book_row in books_df.iterrows():
            book_index = book_id_to_index[book_row['id']]
            Y_cont[:, book_index] /= book_row['community_ratings_count']

        if combine_ratings == 'avg ceil':
            Y = np.ceil(Y_cont).astype(np.int32)
        else:  # combine_ratings == 'avg floor':
            Y = np.floor(Y_cont).astype(np.int32)
    else:
        raise ValueError('Unknown value for `combine_ratings`: `{}`'.format(combine_ratings))

    # Populate the `Overall` column, if needed.
    if return_overall:
        category_to_index = {category: index for index, category in enumerate(categories)}
        rating_to_index = {rating: index for index, rating in enumerate(category_levels[CATEGORY_INDEX_OVERALL])}
        category_level_index_to_rating_index = [dict() for _ in range(CATEGORY_INDEX_GAY_LESBIAN_CHARACTERS)]
        for _, category_row in categories_df.iterrows():
            rating = category_row['rating']
            if pd.isna(rating):
                continue
            category = category_row['category']
            j = category_to_index[category]
            level = category_row['level']
            level_index = category_level_to_index[j][level]
            rating_index = rating_to_index[rating]
            category_level_index_to_rating_index[j][level_index] = rating_index
        Y[CATEGORY_INDEX_OVERALL] = [max([category_level_index_to_rating_index[j][Y[j, i]]
                                          for j in range(len(category_level_index_to_rating_index))])
                                     for i in range(Y.shape[1])]

    if return_meta:
        return inputs, Y, categories, category_levels, \
               np.array(book_ids), books_df, ratings_df, levels_df, categories_df

    return inputs, Y, categories, category_levels


def is_image_file(fname):
    return fname.endswith('.jpg') or \
           fname.endswith('.png') or \
           fname.endswith('.gif') or \
           fname.endswith('.svg') or \
           fname.endswith('.bmp')


def get_images(
        asin,
        source='cover',
        size=None,
        min_size=None,
        max_size=None):
    # Skip books without a known ASIN.
    if asin is None:
        return None

    # Skip books whose content has not yet been scraped.
    folder = os.path.join(folders.AMAZON_KINDLE_IMAGES_PATH, asin)
    if not os.path.exists(folder):
        return None

    if source == 'cover':
        if size is None:
            path = os.path.join(folder, 'cover.jpg')
        else:
            path = os.path.join(folder, 'cover-' + str(size[0]) + 'x' + str(size[1]) + '.jpg')
        if os.path.exists(path):
            size = os.path.getsize(path)
            if not is_between(size, min_size, max_size):
                return None
            return [path]

        # Fail when looking for the cover image by exact name.
        return None

    if source == 'all':
        images = []
        fnames = os.listdir(folder)
        for fname in fnames:
            if not is_image_file(fname):
                continue
            path = os.path.join(folder, fname)
            size = os.path.getsize(path)
            if not is_between(size, min_size, max_size):
                continue
            images.append(path)

        if len(images) == 0:
            return None
        return images

    raise ValueError('Unknown value for `source`: `{}`'.format(source))


def get_labels(asin, category):
    fname = folders.AMAZON_KINDLE_LABELS_FNAME_FORMAT.format(category)
    path = os.path.join(folders.AMAZON_KINDLE_LABELS_PATH, asin, fname)
    if not os.path.exists(path):
        return None
    section_paragraph_labels = paragraph_io.read_formatted_section_paragraph_labels(path)
    labels = []
    for section_i in range(len(section_paragraph_labels)):
        for paragraph_i in range(len(section_paragraph_labels[section_i])):
            labels.append(section_paragraph_labels[section_i][paragraph_i])
    return labels


def save_labels(asin, category, sections, section_ids, labels, force=False, verbose=0):
    section_paragraph_labels = [[] for _ in range(len(sections))]
    for paragraph_i, section_i in enumerate(section_ids):
        label = labels[paragraph_i]
        section_paragraph_labels[section_i].append(label)
    fname = folders.AMAZON_KINDLE_LABELS_FNAME_FORMAT.format(category)
    asin_path = os.path.join(folders.AMAZON_KINDLE_LABELS_PATH, asin)
    if not os.path.exists(asin_path):
        os.mkdir(asin_path)
    path = os.path.join(asin_path, fname)
    paragraph_io.write_formatted_section_paragraph_labels(section_paragraph_labels, path, force=force, verbose=verbose)
