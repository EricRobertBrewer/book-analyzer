import os


def ensure(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


CONTENT_PATH = ensure(os.path.join('..', 'content'))
CONTENT_BOOKCAVE_BOOKS_CSV_PATH = os.path.join(CONTENT_PATH, 'bookcave_books.csv')
CONTENT_BOOKCAVE_BOOK_RATINGS_CSV_PATH = os.path.join(CONTENT_PATH, 'bookcave_book_ratings.csv')
CONTENT_BOOKCAVE_BOOK_RATING_LEVELS_CSV_PATH = os.path.join(CONTENT_PATH, 'bookcave_book_rating_levels.csv')
AMAZON_KINDLE_PATH = os.path.join(CONTENT_PATH, 'amazon_kindle')
AMAZON_KINDLE_IMAGES_PATH = os.path.join(AMAZON_KINDLE_PATH, 'images')
AMAZON_KINDLE_LABELS_PATH = os.path.join(AMAZON_KINDLE_PATH, 'labels')
AMAZON_KINDLE_LABELS_FNAME_FORMAT = 'paragraph_labels_{}.txt'
AMAZON_KINDLE_PARAGRAPHS_PATH = os.path.join(AMAZON_KINDLE_PATH, 'paragraphs')
AMAZON_KINDLE_PARAGRAPH_TOKENS_PATH = os.path.join(AMAZON_KINDLE_PATH, 'paragraph_tokens')
AMAZON_KINDLE_SENTENCE_TOKENS_PATH = os.path.join(AMAZON_KINDLE_PATH, 'sentence_tokens')

EMBEDDINGS_PATH = os.path.join('..', '..', 'embeddings')
EMBEDDING_FASTTEXT_CRAWL_300_PATH = os.path.join(EMBEDDINGS_PATH, 'crawl-300d-2M.vec')  # Contains a header as (n, d).
EMBEDDING_GLOVE_100_PATH = os.path.join(EMBEDDINGS_PATH, 'glove.6B.100d.txt')
EMBEDDING_GLOVE_200_PATH = os.path.join(EMBEDDINGS_PATH, 'glove.6B.200d.txt')
EMBEDDING_GLOVE_300_PATH = os.path.join(EMBEDDINGS_PATH, 'glove.6B.300d.txt')

FIGURES_PATH = ensure(os.path.join('..', 'figures'))

GENERATED_PATH = ensure(os.path.join('..', 'generated'))

HISTORY_PATH = ensure('history')

INPUT_PATH = 'input'

LOGS_PATH = ensure('logs')
CORRELATED_WORDS_PATH = os.path.join(LOGS_PATH, 'correlated_words')
CORRELATED_WORDS_FNAME_FORMAT = 'words-{}-{:d}-{:d}g{:d}-{:d}f.txt'

MODELS_PATH = ensure('models')

PREDICTIONS_PATH = ensure('predictions')

SITES_PATH = os.path.join('python', 'sites')
BOOKCAVE_PATH = os.path.join(SITES_PATH, 'bookcave')
BOOKCAVE_CATEGORIES_PATH = os.path.join(BOOKCAVE_PATH, 'categories')

VECTORS_PATH = ensure('vectors')
