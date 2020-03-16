from keras import utils

from python.util import ordinal

DATA_SUBSET_RATIO = None
DATA_SUBSET_SEED = 1
DATA_PARAGRAPH_MIN_LEN = 256  # The minimum number of paragraphs in each text.
DATA_PARAGRAPH_MAX_LEN = 4096  # The maximum number of paragraphs in each text.
DATA_SENTENCE_MIN_LEN = 512  # The minimum number of sentences in each text.
DATA_SENTENCE_MAX_LEN = 16384  # The maximum number of sentences in each text.
DATA_MIN_TOKENS = 8  # The minimum number of tokens in each paragraph.
DATA_CATEGORIES_MODE = 'soft'
DATA_RETURN_OVERALL = True

TEXT_MAX_WORDS = 8192  # The maximum size of the vocabulary.
TEXT_N_PARAGRAPH_TOKENS = 128  # The maximum number of tokens to process in each paragraph.
TEXT_N_SENTENCES = 16  # The maximum number of sentences to process in each paragraph (HAN only).
TEXT_N_SENTENCE_TOKENS = 32  # The maximum number of tokens to process in each sentence.
TEXT_PADDING = 'pre'
TEXT_TRUNCATING = 'pre'

LABEL_MODE_ORDINAL = 'ordinal'
LABEL_MODE_CATEGORICAL = 'categorical'
LABEL_MODE_REGRESSION = 'regression'

EVAL_TEST_SIZE = 1/5
EVAL_TEST_RANDOM_STATE = 1
EVAL_VAL_SIZE = 1/4
EVAL_VAL_RANDOM_STATE = 1


def transform_labels(y, k, label_mode):
    if label_mode == LABEL_MODE_ORDINAL:
        return ordinal.to_multi_hot_ordinal(y, k=k)  # (n, k - 1)
    if label_mode == LABEL_MODE_CATEGORICAL:
        return utils.to_categorical(y, num_classes=k)  # (n, k)
    if label_mode == LABEL_MODE_REGRESSION:
        return y / k  # (n)
    raise ValueError('Unknown value for `label_mode`: {}'.format(label_mode))


def get_class_weight(k, label_mode, p=1):
    if label_mode == LABEL_MODE_ORDINAL:
        # For example, when `k` is 4, [[ 0 0 0 ], [ 1 0 0 ], [ 1 1 0 ], [ 1 1 1 ]].
        # Scale by `p`, then normalize.
        weight = []
        for i in range(1, k):
            _0 = (1 - i / k) ** p
            _1 = (i / k) ** p
            sum_ = _0 + _1
            weight.append({0: _0 / sum_, 1: _1 / sum_})
        return weight
    if label_mode == LABEL_MODE_CATEGORICAL:
        return {i: 1 / k for i in range(k)}
    if label_mode == LABEL_MODE_REGRESSION:
        return None
    raise ValueError('Unknown value for `label_mode`: {}'.format(label_mode))
