from tensorflow.keras import utils

from classification import ordinal


DATA_SUBSET_RATIO = None
DATA_SUBSET_SEED = None
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


def transform_labels(Y, category_k, label_mode):
    if label_mode == LABEL_MODE_ORDINAL:
        return [ordinal.to_multi_hot_ordinal(Y[j], k=k) for j, k in enumerate(category_k)]  # (c, n, k - 1)
    if label_mode == LABEL_MODE_CATEGORICAL:
        return [utils.to_categorical(Y[j], num_classes=k) for j, k in enumerate(category_k)]  # (c, n, k)
    if label_mode == LABEL_MODE_REGRESSION:
        return [Y[j] / k for j, k in enumerate(category_k)]  # (c, n)
    raise ValueError('Unknown value for `label_mode`: {}'.format(label_mode))


def get_category_class_weights(Y, label_mode, f='inverse'):
    if label_mode == LABEL_MODE_ORDINAL:
        category_class_weights = []  # [[dict]], since classification will be binary cross-entropy.
        for y in Y:
            class_weights = []
            for i in range(y.shape[1]):
                ones_count = sum(y[:, i])
                zeros_count = len(y) - ones_count
                if f == 'inverse':
                    class_weight = {0: 1 / (zeros_count + 1), 1: 1 / (ones_count + 1)}
                elif f == 'square inverse':
                    class_weight = {0: 1 / (zeros_count + 1)**2, 1: 1 / (ones_count + 1)**2}
                else:
                    raise ValueError('Unknown f: {}'.format(f))
                class_weights.append(class_weight)
            category_class_weights.append(class_weights)
        return category_class_weights
    if label_mode == LABEL_MODE_CATEGORICAL:
        category_class_weights = []  # [dict], since classification will be categorical cross-entropy.
        for y in Y:
            class_weight = dict()
            for i in range(y.shape[1]):
                count = sum(y[:, i])
                if f == 'inverse':
                    class_weight[i] = 1 / (count + 1)
                elif f == 'square inverse':
                    class_weight[i] = 1 / (count + 1)**2
                else:
                    raise ValueError('Unknown f: {}'.format(f))
            category_class_weights.append(class_weight)
        return category_class_weights
    if label_mode == LABEL_MODE_REGRESSION:
        return None  # No classes.
    raise ValueError('Unknown value for `label_mode`: {}'.format(label_mode))
