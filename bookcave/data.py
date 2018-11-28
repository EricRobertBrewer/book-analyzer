# Data manipulation.
import sqlite3
import pandas as pd
# File I/O.
import os
import sys
# Math.
import numpy as np

# Declare file path constants.
CONTENT_ROOT = os.path.join('..', 'content')
BOOKCAVE_ROOT = os.path.join(CONTENT_ROOT, 'bookcave')
BOOKCAVE_AMAZON_ROOT = os.path.join(CONTENT_ROOT, 'bookcave_amazon')


def get_data():
    # Pull all of the data from the BookCave database.
    conn = sqlite3.connect(os.path.join(BOOKCAVE_ROOT, 'contents.db'))
    all_books = pd.read_sql_query('SELECT * FROM Books;', conn)
    # ratings = pd.read_sql_query('SELECT * FROM BookRatings;', conn)
    levels = pd.read_sql_query('SELECT * FROM BookRatingLevels;', conn)
    conn.close()
    # Consider only books which have at least one rating.
    rated_books = all_books[all_books['community_ratings_count'] > 0]
    categories = pd.read_csv(os.path.join(CONTENT_ROOT, 'bookcave', 'categories.tsv'), sep='\t')
    category_names = categories['category'].unique()
    # Create index maps of categories and levels to speed up operations later.
    # Map category names to their indices.
    category_indices = dict()
    # Map level names to their indices.
    level_indices = dict()
    for book_index, category in enumerate(category_names):
        category_indices[category] = book_index
        category_rows = categories[categories['category'].str.match(category)]
        j = 0
        for _, row in category_rows.iterrows():
            level_indices[row['level']] = j
            j += 1
    # Map each level to its category index.
    level_to_category_index = dict()
    for _, category_row in categories.iterrows():
        level = category_row['level']
        if level != 'None':
            level_to_category_index[level] = category_indices[category_row['category']]
    # Count the number of levels in each category.
    category_sizes = categories.groupby('category').size()
    # Count how many Amazon Kindle previews have been attempted to be collected.
    preview_book_ids = os.listdir(BOOKCAVE_AMAZON_ROOT)
    # Extract raw book preview contents.
    book_id_to_preview = dict()
    for preview_book_id in preview_book_ids:
        if sys.platform == 'win32':
            # One book folder is named:
            # `diy-body-care-the-complete-body-care-guide-for-beginners-with-over-37-recipes-for-homemade-body-butters-body-scrubs-lotions-lip-balms-and-shampoos-body-care-essential-oils-organic-lotions`.
            # To overcome a `FileNotFoundError` for this file, use an extended-length path on Windows.
            # See `https://stackoverflow.com/questions/36219317/pathname-too-long-to-open/36219497`.
            path = u'\\\\?\\' + os.path.abspath(os.path.join(BOOKCAVE_AMAZON_ROOT, preview_book_id, 'book.txt'))
        # elif sys.platform == 'darwin':
        else:
            path = os.path.join(BOOKCAVE_AMAZON_ROOT, preview_book_id, 'book.txt')
        try:
            with open(path, 'r', encoding='utf-8') as fd:
                contents = fd.read()
                # Skip empty text files.
                if len(contents) == 0:
                    continue
                book_id_to_preview[preview_book_id] = contents
        except FileNotFoundError:
            pass
    # Count how many previews exist for books with at least one rating.
    books = rated_books[rated_books['id'].isin(book_id_to_preview)]
    # Map book IDs to indices.
    book_indices = dict()
    book_index = 0
    for _, book in books.iterrows():
        book_indices[book['id']] = book_index
        book_index += 1
    # Likewise, create a fancy-indexable array of book IDs.
    book_ids = np.array([book['id'] for _, book in books.iterrows()])
    # For each category, calculate the average rating for each book.
    y_cont = np.zeros((len(books), len(category_names)))
    # Add all levels together for each book.
    for _, level in levels.iterrows():
        book_id = level['book_id']
        # Skip books which have a rating (and rating levels), but no preview text.
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
    return book_ids, book_id_to_preview, category_names, category_sizes, y
