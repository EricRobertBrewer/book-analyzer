# Math.
import numpy as np
# Data manipulation.
import sqlite3
import pandas as pd
# File I/O.
import os

# Declare file path constants.
CONTENT_ROOT = os.path.join('..', 'content')
AMAZON_KINDLE_CONTENT_ROOT = os.path.join(CONTENT_ROOT, 'amazon_kindle')
BOOKCAVE_ROOT = 'bookcave'


def is_between(value, _min=None, _max=None):
    if _min is not None and value < _min:
        return False
    if _max is not None and value > _max:
        return False
    return True


def get_text(
        book_row,
        source='book',
        input='content',
        min_len=None,
        max_len=None):
    if source == 'book' or source == 'preview':
        # Skip books without a known ASIN.
        asin = book_row['asin']
        if asin is None:
            return None

        # Determine the type of texts that will be retrieved.
        if source == 'book':
            text_file = 'text.txt'
        else:  # source == 'preview':
            text_file = 'preview.txt'

        # Get the file path to the text.
        path = os.path.join(AMAZON_KINDLE_CONTENT_ROOT, asin, text_file)
        if not os.path.exists(path):
            return None

        # Conditionally open the file.
        text = None
        if input == 'content' or min_len is not None or max_len is not None:
            with open(path, 'r', encoding='utf-8') as fd:
                text = fd.read()

        # Validate file length.
        if not is_between(len(text), min_len, max_len):
            return None

        if input == 'content':
            return text
        elif input == 'filename':
            return path
        raise ValueError('Unknown value for `input`: `{}`'.format(input))
    elif source == 'description':
        description = book_row['description']
        if not is_between(len(description), min_len, max_len):
            return None
        return description
    raise ValueError('Unknown value for `source`: `{}`'.format(source))


def is_image_file(fname):
    return fname.endswith('.jpg') or \
                    fname.endswith('.png') or \
                    fname.endswith('.gif') or \
                    fname.endswith('.svg') or \
                    fname.endswith('.bmp')


def get_images(
        book_row,
        source='cover',
        min_size=None,
        max_size=None):
    # Skip books without a known ASIN.
    asin = book_row['asin']
    if asin is None:
        return None

    # Skip books whose content has not yet been scraped.
    folder = os.path.join(AMAZON_KINDLE_CONTENT_ROOT, asin)
    if not os.path.exists(folder):
        return None

    if source == 'cover soft' or source == 'cover':
        path = os.path.join(folder, 'cover.jpg')
        if os.path.exists(path):
            size = os.path.getsize(path)
            if not is_between(size, min_size, max_size):
                return None
            return [path]

        # Fail when looking for the cover image by exact name.
        if source == 'cover':
            return None

        # Find the largest image file; assume that it is the cover image.
        largest_image_path = None
        largest_image_size = None
        fnames = os.listdir(folder)
        for fname in fnames:
            if not is_image_file(fname):
                continue
            path = os.path.join(folder, fname)
            size = os.path.getsize(path)
            if not is_between(size, min_size, max_size):
                continue
            if largest_image_size is None or size > largest_image_size:
                largest_image_path = path
                largest_image_size = size

        if largest_image_path is None:
            return None
        return [largest_image_path]

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
        media,
        text_source='book',
        text_input='content',
        text_min_len=None,
        text_max_len=None,
        images_source='cover',
        images_min_size=None,
        images_max_size=None,
        categories_mode='soft',
        combine_ratings='max',
        return_meta=False,
        verbose=False):
    """
    Retrieve text and/or images with corresponding labels for books in the BookCave database.
    :param media: set of str {'text', 'images'}, default None
        The type of media to be retrieved.
        When 'text', only text and associated labels will be returned.
        When 'images', only images and associated labels will be returned.
        When 'both', text and images with associated labels will be returned.
    :param text_source: string {'book' (default), 'preview', 'description'}
        The source of text to retrieve.
        When 'book', the entire book texts will be returned.
        When 'preview', the first few chapters of books will be returned.
        When 'description', only a few paragraphs describing the book will be returned.
    :param text_input: string {'content' (default), 'filename'}
        The medium by which text will be returned.
        When 'content', raw text content will be returned.
        When 'filename', file paths specifying the text contents will be returned.
    :param text_min_len: int or None, default None
        The minimum length of texts that will be returned.
    :param text_max_len: int or None, default None
        The maximum length of texts that will be returned.
    :param images_source: str {'cover' (default), 'cover soft', 'all'}
        The quantity of images to retrieve.
        When 'cover', only the image named exactly `cover.jpg` will be returned for each book.
        When 'cover soft', `cover.jpg` will be looked for first.
            Otherwise, the largest image (by size in bytes) in the book folder will be returned.
        When 'all', all images in the book folder will be returned.
    :param images_min_size: int or None, default None
        The minimum file size (in bytes) of images that will be returned.
    :param images_max_size: int or None, default None
        The maximum file size (in bytes) of images that will be returned.
    :param categories_mode: string {'soft' (default), 'medium', 'hard'}
        The flexibility of rating levels within categories.
        When 'soft', all levels which would yield the same base overall rating (without a '+') will be collapsed.
        When 'medium', all levels which would yield the same overall rating will be collapsed.
        When 'hard', no levels will be collapsed.
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
            inputs (dict):                  A dict containing file paths or raw texts of books and/or images.
            Y (np.ndarray):                 Level (label) for the corresponding text/images in 8 categories.
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
    # Validate `media`.
    if media is None:
        raise ValueError('Parameter `media` is None. Should be a set of str.')
    if 'text' not in media and 'images' not in media:
        raise ValueError('Unknown values in `media`: `{}`. Should include `text` or `images`.'.format(media))

    # Read all of the data from the BookCave database.
    conn = sqlite3.connect(os.path.join(CONTENT_ROOT, 'contents.db'))
    all_books_df = pd.read_sql_query('SELECT * FROM BookCaveBooks;', conn)
    all_books_df.sort_values('id', inplace=True)
    ratings_df = pd.read_sql_query('SELECT * FROM BookCaveBookRatings;', conn)
    levels_df = pd.read_sql_query('SELECT * FROM BookCaveBookRatingLevels;', conn)
    conn.close()

    # Consider only books which have at least one rating.
    rated_books_df = all_books_df[all_books_df['community_ratings_count'] > 0]

    # Determine which books will be retrieved.
    if verbose:
        print('Collecting inputs...')
    book_id_set = set()
    book_id_list = []
    text_list = []
    images_list = []
    for _, rated_book_row in rated_books_df.iterrows():
        # Ensure that text AND images are available when `media`==`both`.
        text = None
        if 'text' in media:
            text = get_text(rated_book_row, text_source, text_input, text_min_len, text_max_len)
            if text is None:
                continue
        images = None
        if 'images' in media:
            images = get_images(rated_book_row, images_source, images_min_size, images_max_size)
            if images is None:
                continue
        book_id = rated_book_row['id']
        book_id_list.append(book_id)
        book_id_set.add(book_id)
        if 'text' in media:
            text_list.append(text)
        if 'images' in media:
            images_list.append(images)

    # Consider only books for which text has been collected.
    if 'text' in media:
        books_df = rated_books_df[rated_books_df['id'].isin(book_id_set)]
    # elif 'images' in media:
    else:
        books_df = rated_books_df[rated_books_df['id'].isin(book_id_set)]

    # Create a fancy-indexable array of book IDs.
    book_ids = np.array(books_df['id'].values)

    # Map book IDs to indices.
    book_id_to_index = {book_id: i for i, book_id in enumerate(book_ids)}

    # Create the return value.
    inputs = dict()
    if 'text' in media:
        inputs['text'] = text_list
    if 'images' in media:
        inputs['images'] = images_list

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
        print('Extracting labels for {} books...'.format(len(book_ids)))
    if combine_ratings == 'max':
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
    elif combine_ratings == 'avgceil' or combine_ratings == 'avgfloor':
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
    else:
        raise ValueError('Unknown value for `combine_ratings`: `{}`'.format(combine_ratings))

    if return_meta:
        return inputs, Y, categories, levels, \
               book_ids, all_books_df, rated_books_df, books_df, ratings_df, levels_df, categories_df

    return inputs, Y, categories, levels


def main():
    """
    Test each permutation of parameters.
    """
    for text_source in ['book', 'preview', 'description']:
        # for input in ['content', 'filename']:
        text_input = 'filename'
        for categories_mode in ['hard', 'medium', 'soft']:
            for combine_ratings in ['avgceil', 'avgfloor', 'max']:
                inputs, Y, categories, levels = get_data({'text'},
                                                         text_source=text_source,
                                                         text_input=text_input,
                                                         text_min_len=None,
                                                         text_max_len=None,
                                                         categories_mode=categories_mode,
                                                         combine_ratings=combine_ratings)
                print('text_source={}, '
                      'text_input={}, '
                      'categories_modex={}, '
                      'combine_ratings={}: '
                      'len(inputs)={:d},'
                      ' Y.shape={}'
                      .format(text_source,
                              input,
                              categories_mode,
                              combine_ratings,
                              len(inputs),
                              Y.shape))


if __name__ == '__main__':
    main()
