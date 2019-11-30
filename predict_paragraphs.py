import os
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from classification import data_utils, evaluation, ordinal, shared_parameters
from classification.net.attention_with_context import AttentionWithContext
import folders
from sites.bookcave import bookcave


def get_input_sequence(source_tokens, tokenizer, n_tokens, padding='pre', truncating='pre', split='\t'):
    return np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in source_tokens]),
                                  maxlen=n_tokens,
                                  padding=padding,
                                  truncating=truncating))


def evaluate_model(model, P_predict, Q_true, categories, overall_last=True, category_indices=None):
    # Predict.
    Q_pred_ordinal = model.predict(P_predict)
    Q_pred = [ordinal.from_multi_hot_ordinal(q, threshold=.5) for q in Q_pred_ordinal]

    # Evaluate.
    category_metrics = []
    for j, category in enumerate(categories):
        print()
        print(category)
        q_true = Q_true[j]
        q_pred = Q_pred[j]
        if category_indices is not None:
            q_true = q_true[category_indices[j]]
            q_pred = q_pred[category_indices[j]]
        confusion, metrics = evaluation.get_confusion_and_metrics(q_true, q_pred)
        print(confusion)
        for i, metric_name in evaluation.METRIC_NAMES:
            print('{}: {:.4f}'.format(metric_name, metrics[i]))
        category_metrics.append(metrics)

    # Average.
    print('\nAverage')
    if overall_last:
        n_average = len(category_metrics) - 1
    else:
        n_average = len(category_metrics)
    metrics_avg = [sum([metrics[i] for metrics in category_metrics[:n_average]])/n_average
                   for i in range(len(category_metrics[0]))]
    for i, metric_name in evaluation.METRIC_NAMES:
        print('{}: {:.4f}'.format(metric_name, metrics_avg[i]))


def main(argv):
    if len(argv) != 1:
        raise ValueError('Usage: <category_index>')
    category_index = int(argv[0])

    # Load data.
    source = 'paragraph_tokens'
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    return_overall = shared_parameters.DATA_RETURN_OVERALL
    inputs, Y, categories, category_levels, book_ids, books_df, _, _, categories_df = \
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

    # Reduce labels to the specified category, if needed.
    if category_index != -1:
        Y = np.array([Y[category_index]])
        categories = [categories[category_index]]
        category_levels = [category_levels[category_index]]

    # Load paragraph labels.
    predict_locations = []
    predict_tokens = []
    predict_source_labels = []
    for text_i, source_tokens in enumerate(text_source_tokens):
        book_id = book_ids[text_i]
        asin = books_df[books_df['id'] == book_id].iloc[0]['asin']
        category_labels = [bookcave.get_labels(asin, category) for category in categories[:bookcave.CATEGORY_INDEX_OVERALL]]
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
    if category_index == -1 and return_overall:
        Q_true[bookcave.CATEGORY_INDEX_OVERALL] = bookcave.get_y_overall(Q_true, categories_mode=categories_mode)

    # Get balanced indices.
    seed = 1
    category_balanced_indices = [data_utils.get_balanced_indices_sample(q_true, minlength=len(category_levels[j]), seed=seed)
                                 for j, q_true in enumerate(Q_true)]

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
    model_paths = [
        os.path.join(folders.MODELS_PATH, 'paragraph_cnn_max_ordinal', '33063788_overall_max-agg.h5'),
        os.path.join(folders.MODELS_PATH, 'paragraph_rnn_max_ordinal', '33063789_overall_max-agg.h5'),
        os.path.join(folders.MODELS_PATH, 'paragraph_rnncnn_max_ordinal', '33063790_overall_max-agg.h5')
    ]
    model_custom_objects = [
        None,
        {'AttentionWithContext': AttentionWithContext},
        {'AttentionWithContext': AttentionWithContext}
    ]
    for m, model_path in enumerate(model_paths):
        print('\n{}'.format(model_path))
        model = load_model(model_path, custom_objects=model_custom_objects[m])
        evaluate_model(model, P_predict, Q_true, categories, overall_last=return_overall)
        print('\nBalanced')
        evaluate_model(model, P_predict, Q_true, categories, overall_last=return_overall, category_indices=category_balanced_indices)


if __name__ == '__main__':
    main(sys.argv[1:])
