import os
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from classification import evaluation, ordinal, shared_parameters
from classification.net.attention_with_context import AttentionWithContext
import folders
from sites.bookcave import bookcave


def get_input_sequence(source_tokens, tokenizer, n_tokens, padding='pre', truncating='pre', split='\t'):
    return np.array(pad_sequences(tokenizer.texts_to_sequences([split.join(tokens) for tokens in source_tokens]),
                                  maxlen=n_tokens,
                                  padding=padding,
                                  truncating=truncating))


def get_balanced_indices(y, minlength=None, seed=None):
    if minlength is None:
        minlength = max(y) + 1

    # Map classes to indices of instances.
    class_to_indices = [[] for _ in range(minlength)]
    for index, value in enumerate(y):
        class_to_indices[value].append(index)

    # Find minimum number of instances over all classes.
    n = min([len(indices) for indices in class_to_indices])

    # Sub-sample `n` instances from each class.
    if seed is not None:
        np.random.seed(seed)
    balanced_indices = []
    for value, indices in enumerate(class_to_indices):
        balanced_indices.extend(np.random.choice(indices, n, replace=False))
    return balanced_indices


def load_model_and_evaluate(model_path, P_predict, Q_true, categories, category_indices=None, custom_objects=None):
    print()
    print(model_path)

    # Load model.
    model = load_model(model_path, custom_objects=custom_objects)
    print(model.summary())

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
        print(metrics[0])
        category_metrics.append(metrics)

    # Average.
    print()
    print('Average')
    metrics_avg = [sum([metrics[i] for metrics in category_metrics])/len(category_metrics)
                   for i in range(len(category_metrics))]
    print(metrics_avg[0])

    # Overall.
    if category_indices is not None:
        return
    print()
    print('Overall')
    q_true_overall = bookcave.get_overall_y(Q_true)
    q_pred_overall = bookcave.get_overall_y(Q_pred)
    confusion_overall, metrics_overall = evaluation.get_confusion_and_metrics(q_true_overall, q_pred_overall)
    print(confusion_overall)
    print(metrics_overall[0])


def main(argv):
    # Load data.
    source = 'paragraph_tokens'
    subset_ratio = shared_parameters.DATA_SUBSET_RATIO
    subset_seed = shared_parameters.DATA_SUBSET_SEED
    min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
    max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
    min_tokens = shared_parameters.DATA_MIN_TOKENS
    categories_mode = shared_parameters.DATA_CATEGORIES_MODE
    inputs, Y, categories, category_levels, book_ids, books_df, _, _, categories_df = \
        bookcave.get_data({source},
                          subset_ratio=subset_ratio,
                          subset_seed=subset_seed,
                          min_len=min_len,
                          max_len=max_len,
                          min_tokens=min_tokens,
                          categories_mode=categories_mode,
                          return_meta=True)
    text_source_tokens = list(zip(*inputs[source]))[0]

    # Load paragraph labels.
    predict_locations = []
    predict_tokens = []
    predict_source_labels = []
    for text_i, source_tokens in enumerate(text_source_tokens):
        book_id = book_ids[text_i]
        asin = books_df[books_df['id'] == book_id].iloc[0]['asin']
        category_labels = [bookcave.get_labels(asin, category) for category in categories]
        if any(labels is None for labels in category_labels):
            continue
        for source_i, tokens in enumerate(source_tokens):
            source_labels = [labels[source_i] for labels in category_labels]
            if any(label == -1 for label in source_labels):
                continue
            predict_locations.append((text_i, source_i))
            predict_tokens.append(tokens)
            predict_source_labels.append(source_labels)
    Q_true = np.array(predict_source_labels).transpose()

    # Get balanced indices.
    seed = 1
    category_balanced_indices = [get_balanced_indices(q_true, minlength=len(category_levels[j]), seed=seed)
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
        os.path.join(folders.MODELS_PATH, 'paragraph_cnn', 'ordinal', '32648160.h5'),
        os.path.join(folders.MODELS_PATH, 'paragraph_rnn', 'ordinal', '32648155_glove300-emb.h5')]
    model_custom_objects = [
        None,
        {'AttentionWithContext': AttentionWithContext}]
    for m, model_path in enumerate(model_paths):
        load_model_and_evaluate(model_path,
                                P_predict,
                                Q_true,
                                categories,
                                custom_objects=model_custom_objects[m])
        print()
        print('Balanced')
        load_model_and_evaluate(model_path,
                                P_predict,
                                Q_true,
                                categories,
                                category_indices=category_balanced_indices,
                                custom_objects=model_custom_objects[m])


if __name__ == '__main__':
    main(sys.argv[1:])
