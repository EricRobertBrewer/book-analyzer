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
AMAZON_KINDLE_CONTENT_ROOT = os.path.join(CONTENT_ROOT, 'amazon_kindle')
BOOKCAVE_ROOT = 'bookcave'


def get_data(
        source='book',
        input='content',
        min_len=None,
        max_len=None,
        categories_mode='medium',
        combine_ratings='avgceil',
        return_meta=False,
        verbose=False):
    """
    :param source: string {'book' (default), 'preview', 'description'}
        The source of text to retrieve.
        When 'book', the entire book texts will be returned.
        When 'preview', the first few chapters of books will be returned.
        When 'description', only a few paragraphs describing the book will be returned.
    :param input: string {'content' (default), 'filename'}
        The medium by which text will be returned.
        When 'content', raw text content will be returned.
        When 'filename', file paths specifying the text contents will be returned.
    :param min_len: None or int, default None
        The minimum length of texts that will be returned.
    :param max_len: None or int, default None
        The maximum length of texts that will be returned.
    :param categories_mode: string {'hard', 'medium' (default), 'soft'}
        The flexibility of rating levels within categories.
        When 'hard', all BookCave category levels will be returned.
        When 'medium', adjacent category levels which result in the same overall rating will be collapsed.
        When 'soft', ratings are collapsed to their formo without a '+'.
        For example, all category levels which would yield a rating of 'Adult' or 'Adult+' are merged.
    :param combine_ratings: string {'avgceil' (default), 'avgfloor', 'max'}
        The method by which multiple ratings for a single book will be combined.
        When `'avgceil'`, the ceiling of the average of all rating levels per category per book will be returned.
        When `'avgfloor'`, the floor of the average of all rating levels per category per book will be returned.
        When `'max'`, the maximum among all rating levels per category per book will be returned.
    :param return_meta: boolean, default False
        When `True`, all meta data will be returned.
    :param verbose: boolean, default False
        When `True`, function progress will be printed to the console.
    :return:
        Always:
            inputs (np.array):              An array of either file paths or of raw texts of books.
            Y (np.ndarray):                 Level (label) for the corresponding text in 8 categories.
            categories (list):              Names of categories.
            levels (list):                  Names of levels per category.
        Only when `return_meta` is set to `True`:
            book_ids (np.array)             Alphabetically-sorted array of book IDs parallel with `inputs`.
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

    # Extract text contents.
    book_id_to_input = dict()
    if source == 'book' or source == 'preview':
        if verbose:
            print('Reading text files...')

        # Determine the type of texts that will be retrieved.
        if source == 'book':
            text_file = 'text.txt'
        else:  # source == 'preview':
            text_file = 'preview.txt'

        # Collect file contents or paths.
        for _, rated_book_row in rated_books_df.iterrows():
            # Skip books without a known ASIN.
            asin = rated_book_row['asin']
            if asin is None:
                continue

            if sys.platform == 'win32':
                # To overcome `FileNotFoundError`s for files with long names, use an extended-length path on Windows.
                # See `https://stackoverflow.com/questions/36219317/pathname-too-long-to-open/36219497`.
                path = u'\\\\?\\' + os.path.abspath(os.path.join(AMAZON_KINDLE_CONTENT_ROOT, asin, text_file))
            # elif sys.platform == 'darwin':
            else:
                path = os.path.join(AMAZON_KINDLE_CONTENT_ROOT, asin, text_file)
            if not os.path.exists(path):
                continue

            # Conditionally open the file.
            text = None
            if input == 'content' or min_len is not None or max_len is not None:
                with open(path, 'r', encoding='utf-8') as fd:
                    text = fd.read()

            # Validate file length.
            if min_len is not None:
                if len(text) < min_len:
                    continue
            if max_len is not None:
                if len(text) > max_len:
                    continue

            book_id = rated_book_row['id']
            if input == 'content':
                book_id_to_input[book_id] = text
            elif input == 'filename':
                book_id_to_input[book_id] = path
            else:
                raise ValueError('Unknown value for `input`: `{}`'.format(input))
    elif source == 'description':
        if verbose:
            print('Collecting book descriptions...')

        for _, rated_book_row in rated_books_df.iterrows():
            description = rated_book_row['description']
            if min_len is not None:
                if len(description) < min_len:
                    continue
            if max_len is not None:
                if len(description) > max_len:
                    continue
            book_id_to_input[rated_book_row['id']] = description
    else:
        raise ValueError('Unknown value for `source`: `{}`'.format(source))

    # Consider only books for which text has been collected.
    books_df = rated_books_df[rated_books_df['id'].isin(book_id_to_input)]

    # Create a fancy-indexable array of book IDs.
    book_ids = np.array(sorted([book_row['id'] for _, book_row in books_df.iterrows()]))

    # Map book IDs to indices.
    book_id_to_index = {book_id: i for i, book_id in enumerate(book_ids)}

    # Create an array of inputs.
    inputs = np.array([book_id_to_input[book_id] for book_id in book_ids])

    # Extract category data.
    if categories_mode == 'hard':
        categories_df = pd.read_csv(os.path.join(BOOKCAVE_ROOT, 'categories_hard.tsv'), sep='\t')
    elif categories_mode == 'medium':
        categories_df = pd.read_csv(os.path.join(BOOKCAVE_ROOT, 'categories_medium.tsv'), sep='\t')
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
        print('Extracting labels for {} texts...'.format(len(book_ids)))

    if combine_ratings == 'avgceil' or combine_ratings == 'avgfloor':
        # For each category, calculate the average rating for each book.
        Y_cont = np.zeros((len(book_ids), len(categories)))
        # First, add all levels together for each book.
        for _, level_row in levels_df.iterrows():
            book_id = level_row['book_id']
            if book_id not in book_id_to_index:
                continue
            # Add this rating level to its category for this book.
            book_index = book_id_to_index[book_id]
            category_index = level_to_category_index[level_row['title']]
            level_index = level_to_index[level_row['title']]
            Y_cont[book_index, category_index] += level_index * level_row['count']
        # Then calculate the average level for each book by dividing by the number of ratings for that book.
        for _, book_row in books_df.iterrows():
            book_index = book_id_to_index[book_row['id']]
            Y_cont[book_index] /= book_row['community_ratings_count']

        if combine_ratings == 'avgceil':
            Y = np.ceil(Y_cont).astype(np.int32)
        else:  # combine_ratings == 'avgfloor':
            Y = np.floor(Y_cont).astype(np.int32)
    elif combine_ratings == 'max':
        # For each book, take the maximum rating level in each category.
        Y = np.zeros((len(book_ids), len(categories)), dtype=np.int32)
        for _, level_row in levels_df.iterrows():
            book_id = level_row['book_id']
            if book_id not in book_id_to_index:
                continue
            book_index = book_id_to_index[book_id]
            category_index = level_to_category_index[level_row['title']]
            level_index = level_to_index[level_row['title']]
            Y[book_index, category_index] = max(Y[book_index, category_index], level_index)
    else:
        raise ValueError('Unknown value for `combine_ratings`: `{}`'.format(combine_ratings))

    if return_meta:
        return inputs, Y, categories, levels,\
               book_ids, all_books_df, rated_books_df, books_df, ratings_df, levels_df, categories_df

    return inputs, Y, categories, levels


def main():
    """
    Test each permutation of parameters.
    """
    for source in ['book', 'preview', 'description']:
        # for input in ['content', 'filename']:
        input = 'filename'
        for categories_mode in ['hard', 'medium', 'soft']:
            for combine_ratings in ['avgceil', 'avgfloor', 'max']:
                inputs, Y, categories, levels = get_data(source=source,
                                                         input=input,
                                                         min_len=None,
                                                         max_len=None,
                                                         categories_mode=categories_mode,
                                                         combine_ratings=combine_ratings)
                print('source={}, input={}, categories_modex={}, combine_ratings={}: len(inputs)={:d}, Y.shape={}'
                      .format(source, input, categories_mode, combine_ratings, len(inputs), Y.shape))


if __name__ == '__main__':
    main()
