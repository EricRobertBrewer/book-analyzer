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
BOOKCAVE_ROOT = os.path.join(CONTENT_ROOT, 'bookcave')
BOOKCAVE_AMAZON_KINDLE_ROOT = os.path.join(CONTENT_ROOT, 'bookcave_amazon_kindle')
BOOKCAVE_AMAZON_PREVIEW_ROOT = os.path.join(CONTENT_ROOT, 'bookcave_amazon_preview')


def get_data(text_file='text.txt', kindle=True, verbose=False):

    # Pull all of the data from the BookCave database.
    conn = sqlite3.connect(os.path.join(BOOKCAVE_ROOT, 'contents.db'))
    all_books_df = pd.read_sql_query('SELECT * FROM Books;', conn)
    # rating_df = pd.read_sql_query('SELECT * FROM BookRatings;', conn)
    levels_df = pd.read_sql_query('SELECT * FROM BookRatingLevels;', conn)
    conn.close()

    # Consider only books which have at least one rating.
    rated_books_df = all_books_df[all_books_df['community_ratings_count'] > 0]

    # Count how many texts have been attempted to be collected.
    text_root = BOOKCAVE_AMAZON_KINDLE_ROOT if kindle else BOOKCAVE_AMAZON_PREVIEW_ROOT
    text_book_ids = os.listdir(text_root)
    
    if verbose:
        print('Reading text files...')

    # Extract raw book text contents.
    book_id_to_text = dict()
    # Skip reading books into memory which do not have a rating.
    rated_book_ids = set(rated_books_df['id'])
    for text_book_id in text_book_ids:
        if text_book_id not in rated_book_ids:
            continue
        if sys.platform == 'win32':
            # To overcome `FileNotFoundError`s for files with long names, use an extended-length path on Windows.
            # See `https://stackoverflow.com/questions/36219317/pathname-too-long-to-open/36219497`.
            path = u'\\\\?\\' + os.path.abspath(os.path.join(text_root, text_book_id, text_file))
        # elif sys.platform == 'darwin':
        else:
            path = os.path.join(text_root, text_book_id, text_file)
        try:
            with open(path, 'r', encoding='utf-8') as fd:
                text = fd.read()
            book_id_to_text[text_book_id] = text
        except FileNotFoundError:
            pass
        except NotADirectoryError:
            pass

    # Consider only books for which text has been collected.
    books_df = rated_books_df[rated_books_df['id'].isin(book_id_to_text)]

    # Create a fancy-indexable array of book IDs.
    book_ids = np.array([book_row['id'] for _, book_row in books_df.iterrows()])

    # Map book IDs to indices.
    book_id_to_index = {book_id: i for i, book_id in enumerate(book_ids)}

    # Create an array of texts.
    texts = np.array([book_id_to_text[book_id] for book_id in book_ids])

    # Extract category data.
    categories_df = pd.read_csv(os.path.join(BOOKCAVE_ROOT, 'categories.tsv'), sep='\t')

    # Enumerate category names.
    categories = list(categories_df['category'].unique())

    # Map category names to their indices.
    category_to_index = {category: i for i, category in enumerate(categories)}

    # Map each level name to its index within its own category.
    level_to_index = dict()
    for category in categories:
        category_rows = categories_df[categories_df['category'].str.match(category)]
        category_levels = category_rows['level']
        category_level_to_index = {name: j for j, name in enumerate(category_levels)}
        level_to_index.update(category_level_to_index)

    # Map each level to its category index.
    level_to_category_index = dict()
    # Enumerate the level names per category.
    levels = [['None'] for _ in range(len(categories))]
    for _, category_row in categories_df.iterrows():
        level = category_row['level']
        if level == 'None':
            continue
        level_to_category_index[level] = category_to_index[category_row['category']]
        category_index = level_to_category_index[level]
        levels[category_index].append(level)

    if verbose:
        print('Extracting labels for {} books...'.format(len(book_ids)))

    # For each category, calculate the average rating for each book.
    y_cont = np.zeros((len(book_ids), len(categories)))
    # Add all levels together for each book.
    for _, level_row in levels_df.iterrows():
        book_id = level_row['book_id']
        # Skip books which have a rating (and rating levels), but no text.
        if book_id not in book_id_to_index:
            continue
        # Add this rating level to its category for this book.
        book_index = book_id_to_index[book_id]
        category_index = level_to_category_index[level_row['title']]
        level_index = level_to_index[level_row['title']]
        y_cont[book_index, category_index] += level_index * level_row['count']
    # Calculate the average level for each book by dividing by the number of ratings for that book.
    for _, book_row in books_df.iterrows():
        book_id = book_row['id']
        book_index = book_id_to_index[book_id]
        y_cont[book_index] /= book_row['community_ratings_count']

    # Since false negatives are less desirable than false positives, implement somewhat of a "harsh critic"
    # by taking the ceiling of the average ratings.
    y = np.ceil(y_cont).astype(np.int32)

    return texts, y, categories, levels


def get_train_test_split(x, y, fold, folds, seed=None):
    # Generate a random permutation in order to process the data set in a random order.
    if seed:
        np.random.seed(seed)
    perm = np.random.permutation(len(y))
    # Cross validate...
    test_start = len(y) * fold // folds
    test_end = len(y) * (fold + 1) // folds
    perm_train = np.concatenate((perm[:test_start], perm[test_end:]))
    perm_test = perm[test_start:test_end]
    x_train = x[perm_train]
    x_test = x[perm_test]
    y_train = y[perm_train]
    y_test = y[perm_test]
    return x_train, x_test, y_train, y_test
