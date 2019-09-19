DATA_SUBSET_RATIO = None
DATA_SUBSET_SEED = None
DATA_PARAGRAPH_MIN_LEN = 256  # The minimum number of paragraphs in each text.
DATA_PARAGRAPH_MAX_LEN = 4096  # The maximum number of paragraphs in each text.
DATA_SENTENCE_MIN_LEN = 512  # The minimum number of sentences in each text.
DATA_SENTENCE_MAX_LEN = 16384  # The maximum number of sentences in each text.
DATA_MIN_TOKENS = 8  # The minimum number of tokens in each paragraph.

LABEL_MODE_ORDINAL = 'ordinal'
LABEL_MODE_CATEGORICAL = 'categorical'
LABEL_MODE_REGRESSION = 'regression'

EVAL_TEST_SIZE = .25
EVAL_TEST_RANDOM_STATE = 1
EVAL_VAL_SIZE = .1
EVAL_VAL_RANDOM_STATE = 1
