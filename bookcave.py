# Math.
import numpy as np
# Data manipulation.
import sqlite3
import pandas as pd
# File I/O.
import os
import sys

# Declare file path constants.
CONTENT_ROOT = os.path.join('..', 'content')
AMAZON_KINDLE_ROOT = os.path.join(CONTENT_ROOT, 'amazon_kindle')
BOOKCAVE_ROOT = os.path.join(CONTENT_ROOT, 'bookcave')


def get_data(
        input='content',
        categories_mode='hard',
        combine_ratings='average',
        return_meta=False,
        verbose=False):
    """
    :param input: string {'filename', 'content'}
    :param combine_ratings: string {'average', 'max'}
        When `'average'`, the average of all rating levels per book will be returned.
        When `'max'`, the maximum among all rating levels per book will be returned.
    :param categories_mode: string {'hard', 'soft'}
        When 'hard', all BookCave category levels will be returned.
        When 'soft', adjacent category levels which result in the same rating will be collapsed.
    :param return_meta: boolean, default False
        When `True`, all meta data will be returned.
    :param verbose: boolean, default False
        When `True`, function progress will be printed to the console.
    :return:
        Always:
            text (np.array):                Raw texts of books.
            y (np.ndarray):                 Averaged level (label) for the corresponding text in 8 categories.
            categories (list):              Names of categories.
            levels (list):                  Names of levels per category.
        Only when `return_meta` is set to `True`:
            all_books_df (pd.DataFrame):    Metadata for all books collected from BookCave.
            rated_books_df (pd.DataFrame):  Metadata for books which have been rated.
            books_df (pd.DataFrame):        Metadata for books which have been rated AND have text.
            ratings_df (pd.DataFrame):      Metadata for book ratings (unused).
            levels_df (pd.DataFrame):       Metadata for book rating levels.
            categories_df (pd.DataFrame):   Metadata for categories (contains a description for each rating level).
    """
    # Read all of the data from the BookCave database.
    conn = sqlite3.connect(os.path.join(CONTENT_ROOT, 'contents.db'))
    all_books_df = pd.read_sql_query('SELECT * FROM BookCaveBooks;', conn)
    ratings_df = pd.read_sql_query('SELECT * FROM BookCaveBookRatings;', conn)
    levels_df = pd.read_sql_query('SELECT * FROM BookCaveBookRatingLevels;', conn)
    conn.close()

    # Consider only books which have at least one rating.
    rated_books_df = all_books_df[all_books_df['community_ratings_count'] > 0]

    if verbose:
        print('Reading text files...')

    # Count how many texts have been attempted to be collected.
    text_file = 'text.txt'
    # Extract raw book text contents.
    book_asin_to_input = dict()
    # Skip reading books into memory which do not have a rating or a ASIN.
    rated_book_asins = set(rated_books_df['asin'].dropna())
    for book_asin in rated_book_asins:
        if sys.platform == 'win32':
            # To overcome `FileNotFoundError`s for files with long names, use an extended-length path on Windows.
            # See `https://stackoverflow.com/questions/36219317/pathname-too-long-to-open/36219497`.
            path = u'\\\\?\\' + os.path.abspath(os.path.join(AMAZON_KINDLE_ROOT, book_asin, text_file))
        # elif sys.platform == 'darwin':
        else:
            path = os.path.join(AMAZON_KINDLE_ROOT, book_asin, text_file)
        if not os.path.exists(path):
            continue
        if input == 'content':
            with open(path, 'r', encoding='utf-8') as fd:
                text = fd.read()
            book_asin_to_input[book_asin] = text
        elif input == 'filename':
            book_asin_to_input[book_asin] = path

    # Consider only books for which text has been collected.
    books_df = rated_books_df[rated_books_df['asin'].isin(book_asin_to_input)]

    # Create a fancy-indexable array of book ASINs.
    book_asins = np.array([book_row['asin'] for _, book_row in books_df.iterrows()])

    # Map book ASINs to indices.
    book_asin_to_index = {book_asin: i for i, book_asin in enumerate(book_asins)}

    # Map each book ID to its corresponding ASIN.
    book_id_to_asin = {book_row['id']: book_row['asin'] for _, book_row in books_df.iterrows()}

    # Create an array of input.
    inputs = np.array([book_asin_to_input[book_asin] for book_asin in book_asins])

    # Extract category data.
    if categories_mode == 'hard':
        categories_df = pd.read_csv(os.path.join(BOOKCAVE_ROOT, 'categories_hard.tsv'), sep='\t')
    elif categories_mode == 'soft':
        categories_df = pd.read_csv(os.path.join(BOOKCAVE_ROOT, 'categories_soft.tsv'), sep='\t')
    else:
        raise ValueError('Unknown value for `categories_mode`: `{}`'.format(categories_mode))

    # Enumerate category names.
    categories = list(categories_df['category'].unique())

    # Map category names to their indices.
    category_to_index = {category: i for i, category in enumerate(categories)}

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
    levels = [['None'] for _ in range(len(categories))]
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
        levels[category_index].append(level)

    if verbose:
        print('Extracting labels for {} books...'.format(len(book_asins)))

    if combine_ratings == 'average':
        # For each category, calculate the average rating for each book.
        Y_cont = np.zeros((len(book_asins), len(categories)))
        # First, add all levels together for each book.
        for _, level_row in levels_df.iterrows():
            book_id = level_row['book_id']
            if book_id not in book_id_to_asin:
                continue
            book_asin = book_id_to_asin[book_id]
            if book_asin not in book_asin_to_index:
                continue
            # Add this rating level to its category for this book.
            book_index = book_asin_to_index[book_asin]
            category_index = level_to_category_index[level_row['title']]
            level_index = level_to_index[level_row['title']]
            Y_cont[book_index, category_index] += level_index * level_row['count']
        # Then calculate the average level for each book by dividing by the number of ratings for that book.
        for _, book_row in books_df.iterrows():
            book_id = book_row['id']
            book_asin = book_id_to_asin[book_id]
            book_index = book_asin_to_index[book_asin]
            Y_cont[book_index] /= book_row['community_ratings_count']

        # Since false negatives are less desirable than false positives, implement somewhat of a "harsh critic"
        # by taking the ceiling of the average ratings.
        Y = np.ceil(Y_cont).astype(np.int32)
    elif combine_ratings == 'max':
        # For each book, take the maximum rating level in each category.
        Y = np.zeros((len(book_asins), len(categories)), dtype=np.int32)
        for _, level_row in levels_df.iterrows():
            book_id = level_row['book_id']
            if book_id not in book_id_to_asin:
                continue
            book_asin = book_id_to_asin[book_id]
            if book_asin not in book_asin_to_index:
                continue
            book_index = book_asin_to_index[book_asin]
            category_index = level_to_category_index[level_row['title']]
            level_index = level_to_index[level_row['title']]
            Y[book_index, category_index] = max(Y[book_index, category_index], level_index)
    else:
        raise ValueError('Unknown value for `combine_ratings`: `{}`'.format(combine_ratings))

    if return_meta:
        return inputs, Y, categories, levels, all_books_df, rated_books_df, books_df, ratings_df, levels_df, categories_df

    return inputs, Y, categories, levels


def get_text_lines(texts):
    return np.array([text.split('\n') for text in texts])
