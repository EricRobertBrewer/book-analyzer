# Math.
import numpy as np
# Data manipulation.
import sqlite3
import pandas as pd
# File I/O.
import os

import folders
from text import paragraph_io

# Declare file path constants.

# Category indices.
CATEGORY_INDEX_CRUDE_HUMOR_LANGUAGE = 0
CATEGORY_INDEX_DRUG_ALCOHOL_TOBACCO_USE = 1
CATEGORY_INDEX_KISSING = 2
CATEGORY_INDEX_PROFANITY = 3
CATEGORY_INDEX_NUDITY = 4
CATEGORY_INDEX_SEX_AND_INTIMACY = 5
CATEGORY_INDEX_VIOLENCE_AND_HORROR = 6
CATEGORY_INDEX_GAY_LESBIAN_CHARACTERS = 7
CATEGORIES = [
    'crude_humor_language',
    'drug_alcohol_tobacco_use',
    'kissing',
    'profanity',
    'nudity',
    'sex_and_intimacy',
    'violence_and_horror',
    'gay_lesbian_characters'
]
CATEGORY_NAMES = [
    'Crude Humor/Language',
    'Drug, Alcohol & Tobacco Use',
    'Kissing',
    'Profanity',
    'Nudity',
    'Sex and Intimacy',
    'Violence and Horror',
    'Gay/Lesbian Characters'
]


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
    paragraphs, section_ids = [], []
    for section_i in range(len(section_paragraphs)):
        for paragraph_i in range(len(section_paragraphs[section_i])):
            paragraphs.append(section_paragraphs[section_i][paragraph_i])
            section_ids.append(section_i)
    if not is_between(len(paragraphs), min_len, max_len):
        return None
    return paragraphs, section_ids, sections


def get_paragraph_tokens(asin, min_len=None, max_len=None):
    path = os.path.join(folders.AMAZON_KINDLE_TOKENS_PATH, '{}.txt'.format(asin))
    if not os.path.exists(path):
        return None

    section_paragraphs_tokens = paragraph_io.read_formatted_section_paragraph_tokens(path)
    paragraph_tokens, section_ids = [], []
    for section_i in range(len(section_paragraphs_tokens)):
        for paragraph_i in range(len(section_paragraphs_tokens[section_i])):
            paragraph_tokens.append(section_paragraphs_tokens[section_i][paragraph_i])
            section_ids.append(section_i)
    if not is_between(len(paragraph_tokens), min_len, max_len):
        return None
    return paragraph_tokens, section_ids


def get_input(source, asin, min_len=None, max_len=None):
    if source == 'book':
        return get_book(asin, min_len, max_len)
    if source == 'preview':
        return get_preview(asin, min_len, max_len)
    if source == 'paragraphs':
        return get_paragraphs(asin, min_len, max_len)
    if source == 'tokens':
        return get_paragraph_tokens(asin, min_len, max_len)
    raise ValueError('Unknown source: `{}`.'.format(source))


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


def get_data(
        sources,
        min_len=None,
        max_len=None,
        categories_mode='soft',
        only_categories=None,
        combine_ratings='max',
        return_meta=False,
        verbose=False):
    """
    Retrieve text with corresponding labels for books in the BookCave database.
    :param sources: set of str {'book', 'preview', 'paragraphs' (default), 'tokens'}
        The type(s) of media to be retrieved.
        When 'book', the entire raw book texts will be returned.
        When 'preview', the first few chapters of books will be returned.
        When 'paragraphs', the sections and paragraphs will be returned (as tuples).
        When 'tokens', the space-separated tokens for each paragraph will be returned.
        When `None`, paragraphs will be returned.
    :param min_len: int, optional
        The minimum length of texts that will be returned.
    :param max_len: int, optional
        The maximum length of texts that will be returned.
    :param categories_mode: string {'soft' (default), 'medium', 'hard'}
        The flexibility of rating levels within categories.
        When 'soft', all levels which would yield the same base overall rating (without a '+') will be merged.
        When 'medium', all levels which would yield the same overall rating will be merged.
        When 'hard', no levels will be merged.
    :param only_categories: set of int, optional
        Filter the returned labels (and meta data when `return_meta` is True) only to specific maturity categories.
        When not provided, all category labels will be returned.
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
            inputs (dict):                  A dict containing file paths, raw texts, section/section-paragraph tuples,
                                                or section-paragraph-token lists of books and/or images.
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
    # Validate `media`.
    if sources is None:
        sources = {'paragraphs'}

    # Read all of the data from the BookCave database.
    conn = sqlite3.connect(os.path.join(folders.CONTENT_PATH, 'contents.db'))
    all_books_df = pd.read_sql_query('SELECT * FROM BookCaveBooks;', conn)
    all_books_df.sort_values('id', inplace=True)
    all_ratings_df = pd.read_sql_query('SELECT * FROM BookCaveBookRatings;', conn)
    all_levels_df = pd.read_sql_query('SELECT * FROM BookCaveBookRatingLevels;', conn)
    conn.close()

    # Consider only books which have at least one rating.
    rated_books_df = all_books_df[all_books_df['community_ratings_count'] > 0]

    # Determine which books will be retrieved.
    if verbose:
        print('Collecting inputs...')
    inputs = dict()
    for source in sources:
        inputs[source] = []
    book_ids = []
    for _, rated_book_row in rated_books_df.iterrows():
        # Skip books without a known ASIN.
        asin = rated_book_row['asin']
        if asin is None:
            continue

        # Ensure that all selected sources exist.
        has_all_sources = True
        book_inputs = dict()
        for source in sources:
            book_input = get_input(source, asin, min_len, max_len)
            if book_input is None:
                has_all_sources = False
                break
            book_inputs[source] = book_input

        if has_all_sources:
            for source, book_input in book_inputs.items():
                inputs[source].append(book_input)
            book_ids.append(rated_book_row['id'])

    # Consider only books for which text has been collected.
    books_df = rated_books_df[rated_books_df['id'].isin(set(book_ids))]

    # Map book IDs to indices.
    book_id_to_index = {book_id: i for i, book_id in enumerate(book_ids)}

    # Extract category data.
    if categories_mode not in {'hard', 'medium', 'soft'}:
        raise ValueError('Unknown value for `categories_mode`: `{}`'.format(categories_mode))
    categories_path = os.path.join(folders.BOOKCAVE_CATEGORIES_PATH, 'categories_{}.tsv'.format(categories_mode))
    all_categories_df = pd.read_csv(categories_path, sep='\t')

    # Enumerate category names.
    categories = list(all_categories_df['category'].unique())
    if only_categories is not None:
        categories = [category for i, category in enumerate(categories) if i in only_categories]

    # Map category names to their indices.
    category_to_index = {category: i for i, category in enumerate(categories)}

    # Get smaller ratings and categories DataFrames.
    ratings_df = all_ratings_df[all_ratings_df['book_id'].isin(book_id_to_index.keys())]
    categories_df = all_categories_df[all_categories_df['category'].isin(category_to_index.keys())]

    # Map each level name to its index within its own category.
    level_to_index = dict()
    for category in categories:
        category_rows = categories_df[categories_df['category'].str.match(category)]
        category_levels = category_rows['level']
        category_level_to_index = dict()
        for j, level in enumerate(category_levels):
            category_level_to_index[level] = j
            for level_part in level.split('|'):
                category_level_to_index[level_part] = j
        level_to_index.update(category_level_to_index)

    # Map and enumerate level names.
    level_to_category_index = dict()
    category_levels = [['None'] for _ in range(len(categories))]
    for _, category_row in categories_df.iterrows():
        level = category_row['level']
        if level == 'None':
            continue
        # Map each level to its category index.
        category_index = category_to_index[category_row['category']]
        level_to_category_index[level] = category_index
        for level_part in level.split('|'):
            level_to_category_index[level_part] = category_index
        # Enumerate the level names per category.
        category_levels[category_index].append(level)

    # Get a smaller levels DataFrame.
    levels_df = all_levels_df[all_levels_df['book_id'].isin(book_id_to_index.keys()) &
                              all_levels_df['title'].isin(level_to_index.keys())]

    if verbose:
        print('Extracting labels for {} books...'.format(len(book_ids)))
    if combine_ratings == 'max':
        # For each book, take the maximum rating level in each category.
        Y = np.zeros((len(categories), len(book_ids)), dtype=np.int32)
        for _, level_row in levels_df.iterrows():
            book_id = level_row['book_id']
            book_index = book_id_to_index[book_id]
            level = level_row['title']
            category_index = level_to_category_index[level]
            level_index = level_to_index[level]
            Y[category_index, book_index] = max(Y[category_index, book_index], level_index)
    elif combine_ratings == 'avg ceil' or combine_ratings == 'avg floor':
        # For each category, calculate the average rating for each book.
        Y_cont = np.zeros((len(categories), len(book_ids)), dtype=np.float32)
        # First, add all levels together for each book.
        for _, level_row in levels_df.iterrows():
            book_id = level_row['book_id']
            book_index = book_id_to_index[book_id]
            level = level_row['title']
            category_index = level_to_category_index[level]
            level_index = level_to_index[level]
            Y_cont[category_index, book_index] += level_index * level_row['count']
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

    if return_meta:
        return inputs, Y, categories, category_levels, \
               np.array(book_ids), books_df, ratings_df, levels_df, categories_df

    return inputs, Y, categories, category_levels


def get_labels(asin, category):
    fname = folders.FNAME_LABELS_FORMAT.format(category)
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
    fname = folders.FNAME_LABELS_FORMAT.format(category)
    asin_path = os.path.join(folders.AMAZON_KINDLE_LABELS_PATH, asin)
    if not os.path.exists(asin_path):
        os.mkdir(asin_path)
    path = os.path.join(asin_path, fname)
    paragraph_io.write_formatted_section_paragraph_labels(section_paragraph_labels, path, force=force, verbose=verbose)
