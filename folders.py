import os


CONTENT_PATH = os.path.join('..', 'content')
CONTENT_BOOKCAVE_BOOKS_CSV_PATH = os.path.join(CONTENT_PATH, 'bookcave_books.csv')
CONTENT_BOOKCAVE_BOOK_RATINGS_CSV_PATH = os.path.join(CONTENT_PATH, 'bookcave_book_ratings.csv')
CONTENT_BOOKCAVE_BOOK_RATING_LEVELS_CSV_PATH = os.path.join(CONTENT_PATH, 'bookcave_book_rating_levels.csv')
AMAZON_KINDLE_PATH = os.path.join(CONTENT_PATH, 'amazon_kindle')
AMAZON_KINDLE_BOOK_PATH = os.path.join(AMAZON_KINDLE_PATH, 'book')
AMAZON_KINDLE_PREVIEW_PATH = os.path.join(AMAZON_KINDLE_PATH, 'preview')
AMAZON_KINDLE_PARAGRAPHS_PATH = os.path.join(AMAZON_KINDLE_PATH, 'paragraphs')
AMAZON_KINDLE_TOKENS_PATH = os.path.join(AMAZON_KINDLE_PATH, 'tokens')
AMAZON_KINDLE_LABELS_PATH = os.path.join(AMAZON_KINDLE_PATH, 'labels')
AMAZON_KINDLE_LABELS_FNAME_FORMAT = 'paragraph_labels_{}.txt'
AMAZON_KINDLE_IMAGES_PATH = os.path.join(AMAZON_KINDLE_PATH, 'images')

SITES_PATH = 'sites'
BOOKCAVE_PATH = os.path.join(SITES_PATH, 'bookcave')
BOOKCAVE_CATEGORIES_PATH = os.path.join(BOOKCAVE_PATH, 'categories')

MODELS_PATH = 'models'

EMBEDDING_GLOVE_100_PATH = os.path.join('..', '..', 'embeddings', 'glove.6B.100d.txt')
EMBEDDING_GLOVE_200_PATH = os.path.join('..', '..', 'embeddings', 'glove.6B.200d.txt')
EMBEDDING_GLOVE_300_PATH = os.path.join('..', '..', 'embeddings', 'glove.6B.300d.txt')

CORRELATED_WORDS_PATH = os.path.join('logs', 'correlated_words')
CORRELATED_WORDS_FNAME_FORMAT = 'words-{}-{:d}-{:d}g{:d}-{:d}f.txt'
