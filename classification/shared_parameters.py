DATA_SUBSET_RATIO = 1.
DATA_SUBSET_SEED = 1
DATA_PARAGRAPH_MIN_LEN = 256  # The minimum number of paragraphs in each text.
DATA_PARAGRAPH_MAX_LEN = 4096  # The maximum number of paragraphs in each text.
DATA_SENTENCE_MIN_LEN = 512  # The minimum number of sentences in each text.
DATA_SENTENCE_MAX_LEN = 16384  # The maximum number of sentences in each text.
DATA_MIN_TOKENS = 8  # The minimum number of tokens in each paragraph.
DATA_CATEGORIES_MODE = 'soft'

TEXT_MAX_WORDS = 8192  # The maximum size of the vocabulary.
TEXT_N_PARAGRAPH_TOKENS = 128  # The maximum number of tokens to process in each paragraph.
TEXT_N_SENTENCES = 16  # The maximum number of sentences to process in each paragraph.
TEXT_N_SENTENCE_TOKENS = 32  # The maximum number of tokens to process in each sentence.
TEXT_PADDING = 'pre'
TEXT_TRUNCATING = 'pre'

LABEL_MODE_ORDINAL = 'ordinal'
LABEL_MODE_CATEGORICAL = 'categorical'
LABEL_MODE_REGRESSION = 'regression'

EVAL_TEST_SIZE = .25
EVAL_TEST_RANDOM_STATE = 1
EVAL_VAL_SIZE = .1
EVAL_VAL_RANDOM_STATE = 1
