# Math.
import numpy as np
# Data manipulation.
import sqlite3
import pandas as pd
# File I/O.
import os
import sys

# Declare file path constants.
CONTENT_ROOT = os.path.join('..', '..', 'content')
BOOKCAVE_ROOT = os.path.join(CONTENT_ROOT, 'bookcave')
BOOKCAVE_AMAZON_KINDLE_ROOT = os.path.join(CONTENT_ROOT, 'bookcave_amazon_kindle')
BOOKCAVE_AMAZON_PREVIEW_ROOT = os.path.join(CONTENT_ROOT, 'bookcave_amazon_preview')


def get_data(text_file_name, kindle=False, verbose=False):

    # Pull all of the data from the BookCave database.
    if verbose:
        print('Opening database...')
    conn = sqlite3.connect(os.path.join(BOOKCAVE_ROOT, 'contents.db'))
    all_books = pd.read_sql_query('SELECT * FROM Books;', conn)
    # ratings = pd.read_sql_query('SELECT * FROM BookRatings;', conn)
    levels = pd.read_sql_query('SELECT * FROM BookRatingLevels;', conn)
    conn.close()

    # Consider only books which have at least one rating.
    rated_books = all_books[all_books['community_ratings_count'] > 0]

    # Extract categories.
    categories = pd.read_csv(os.path.join(CONTENT_ROOT, 'bookcave', 'categories.tsv'), sep='\t')

    # Enumerate category names.
    category_names = categories['category'].unique()

    # Map category names to their indices.
    category_indices = dict()
    for i, category in enumerate(category_names):
        category_indices[category] = i

    # Map level names to their indices.
    level_indices = dict()
    for category in category_names:
        category_rows = categories[categories['category'].str.match(category)]
        category_level_names = category_rows['level']
        category_level_indices = {name: j for j, name in enumerate(category_level_names)}
        level_indices.update(category_level_indices)

    # Map each level to its category index.
    level_to_category_index = dict()
    for _, category_row in categories.iterrows():
        level = category_row['level']
        if level != 'None':
            level_to_category_index[level] = category_indices[category_row['category']]

    # Enumerate the level names per category.
    level_names = [['None'] for _ in range(len(category_names))]
    for _, category_row in categories.iterrows():
        level = category_row['level']
        if level == 'None':
            continue
        category_index = level_to_category_index[level]
        level_names[category_index].append(level)

    # Count the number of levels in each category.
    category_sizes = categories.groupby('category').size()

    # Count how many Amazon Kindle texts have been attempted to be collected.
    text_root = BOOKCAVE_AMAZON_KINDLE_ROOT if kindle else BOOKCAVE_AMAZON_PREVIEW_ROOT
    text_book_ids = os.listdir(text_root)

    # Extract raw book text contents.
    if verbose:
        print('Reading text files...')
    book_id_to_text = dict()
    for text_book_id in text_book_ids:
        if sys.platform == 'win32':
            # One book folder is named:
            # `diy-body-care-the-complete-body-care-guide-for-beginners-with-over-37-recipes-for-homemade-body-butters-body-scrubs-lotions-lip-balms-and-shampoos-body-care-essential-oils-organic-lotions`.
            # To overcome a `FileNotFoundError` for this file, use an extended-length path on Windows.
            # See `https://stackoverflow.com/questions/36219317/pathname-too-long-to-open/36219497`.
            path = u'\\\\?\\' + os.path.abspath(os.path.join(text_root, text_book_id, text_file_name))
        # elif sys.platform == 'darwin':
        else:
            path = os.path.join(text_root, text_book_id, text_file_name)
        try:
            with open(path, 'r', encoding='utf-8') as fd:
                contents = fd.read()
            # Skip empty text files.
            if len(contents) == 0:
                continue
            book_id_to_text[text_book_id] = contents
        except FileNotFoundError:
            pass
        except NotADirectoryError:
            pass

    # Count how many texts exist for books with at least one rating.
    books = rated_books[rated_books['id'].isin(book_id_to_text)]

    # Map book IDs to indices.
    book_indices = {book_id: i for i, book_id in enumerate(books['id'])}

    # Likewise, create a fancy-indexable array of book IDs.
    book_ids = np.array([book['id'] for _, book in books.iterrows()])

    # For each category, calculate the average rating for each book.
    if verbose:
        print('Extracting labels for {} books...'.format(len(book_indices)))
    y_cont = np.zeros((len(books), len(category_names)))
    # Add all levels together for each book.
    for _, level in levels.iterrows():
        book_id = level['book_id']
        # Skip books which have a rating (and rating levels), but no text.
        if book_id in book_indices:
            # Add this rating level to its category for this book.
            book_index = book_indices[book_id]
            category_index = level_to_category_index[level['title']]
            level_index = level_indices[level['title']]
            y_cont[book_index, category_index] += level_index * level['count']
    # Calculate the average level for each book by dividing by the number of ratings for that book.
    for _, book in books.iterrows():
        book_id = book['id']
        book_index = book_indices[book_id]
        y_cont[book_index] /= book['community_ratings_count']

    # Since false negatives are less desirable than false positives, implement somewhat of a "harsh critic"
    # by taking the ceiling of the average ratings.
    y = np.ceil(y_cont).astype(np.int32)

    return book_ids, book_id_to_text, category_names, category_sizes, level_names, y
