import argparse
import os

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from python.util import evaluation, shared_parameters
from python.util import ordinal
from python.util.net.attention_with_context import AttentionWithContext
from python import folders
from python.sites.bookcave import bookcave


def main():
    parser = argparse.ArgumentParser(
        description='Use a model trained on books to predict the categorical maturity levels of paragraphs.'
    )
    parser.add_argument('cnn',
                        help='CNN base file name.')
    parser.add_argument('rnn',
                        help='RNN base file name.')
    parser.add_argument('rnncnn',
                        help='RNN-CNN base file name.')
    args = parser.parse_args()

    # Load data.
    source = 'paragraph_tokens'
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    return_overall = shared_parameters.DATA_RETURN_OVERALL
    inputs, _, categories, category_levels, book_ids, books_df, _, _, _ = \
        bookcave.get_data({source},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          categories_mode=categories_mode,
                          return_overall=return_overall,
                          return_meta=True)
    text_source_tokens = list(zip(*inputs[source]))[0]

    # Load paragraph labels.
    predict_locations = []
    predict_tokens = []
    predict_source_labels = []
    for text_i, source_tokens in enumerate(text_source_tokens):
        book_id = book_ids[text_i]
        asin = books_df[books_df['id'] == book_id].iloc[0]['asin']
        category_labels = [bookcave.get_labels(asin, category)
                           for category in categories[:bookcave.CATEGORY_INDEX_OVERALL]]
        if any(labels is None for labels in category_labels):
            continue
        for source_i, tokens in enumerate(source_tokens):
            source_labels = [labels[source_i] for labels in category_labels]
            if any(label == -1 for label in source_labels):
                continue
            predict_locations.append((text_i, source_i))
            predict_tokens.append(tokens)
            predict_source_labels.append(source_labels)
    Q_true = np.zeros((len(categories), len(predict_source_labels)), dtype=np.int32)
    for i, source_labels in enumerate(predict_source_labels):
        for j, label in enumerate(source_labels):
            Q_true[j, i] = label
    if return_overall:
        Q_true[bookcave.CATEGORY_INDEX_OVERALL] = bookcave.get_y_overall(Q_true, categories_mode=categories_mode)

    # Tokenize text.
    max_words = shared_parameters.TEXT_MAX_WORDS
    split = '\t'
    tokenizer = Tokenizer(num_words=max_words, split=split)
    all_locations = []
    all_sources = []
    for text_i, source_tokens in enumerate(text_source_tokens):
        for source_i, tokens in enumerate(source_tokens):
            all_locations.append((text_i, source_i))
            all_sources.append(split.join(tokens))
    tokenizer.fit_on_texts(all_sources)

    n_tokens = shared_parameters.TEXT_N_PARAGRAPH_TOKENS
    padding = shared_parameters.TEXT_PADDING
    truncating = shared_parameters.TEXT_TRUNCATING
    P_predict = np.array([get_input_sequence([source_tokens], tokenizer, n_tokens, padding, truncating)
                          for source_tokens in predict_tokens])

    # Evaluate.
    for j, category in enumerate(categories):
        model_paths = [
            os.path.join(folders.MODELS_PATH, 'paragraph_cnn_maxavg_ordinal', '{}_{:d}.h5'.format(args.cnn, j)),
            os.path.join(folders.MODELS_PATH, 'paragraph_rnn_maxavg_ordinal', '{}_{:d}.h5'.format(args.rnn, j)),
            os.path.join(folders.MODELS_PATH, 'paragraph_rnncnn_maxavg_ordinal', '{}_{:d}.h5'.format(args.rnncnn, j))
        ]
        model_custom_objects = [
            None,
            {'AttentionWithContext': AttentionWithContext},
            {'AttentionWithContext': AttentionWithContext}
        ]

        q_true = Q_true[j]
        for m, model_path in enumerate(model_paths):
            print('\n{}'.format(model_path))
            model = load_model(model_path, custom_objects=model_custom_objects[m])
            evaluate_model(model,
                           P_predict,
                           q_true,
                           category)


def get_input_sequence(source_tokens, tokenizer, n_tokens, padding='pre', truncating='pre', split='\t'):
    return np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in source_tokens]),
                                  maxlen=n_tokens,
                                  padding=padding,
                                  truncating=truncating))


def evaluate_model(model, P_predict, q_true, category):
    # Predict.
    q_pred_ordinal = model.predict(P_predict)
    q_pred = ordinal.from_multi_hot_ordinal(q_pred_ordinal, threshold=.5)

    # Evaluate.
    print()
    print(category)
    confusion, metrics = evaluation.get_confusion_and_metrics(q_true, q_pred)
    print(confusion)
    for i, metric_name in enumerate(evaluation.METRIC_NAMES):
        print('{}: {:.4f}'.format(metric_name, metrics[i]))


if __name__ == '__main__':
    main()
