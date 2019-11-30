#!/usr/bin/env python
# coding: utf-8

# # predict_paragraphs_baselines

# In[ ]:


import os

from tensorflow.keras.models import load_model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from classification import baselines, data_utils, evaluation, shared_parameters
import folders
import predict_paragraphs
from sites.bookcave import bookcave


# ## Load paragraphs

# In[ ]:


source = 'paragraph_tokens'
subset_ratio = shared_parameters.DATA_SUBSET_RATIO
subset_seed = shared_parameters.DATA_SUBSET_SEED
min_len = shared_parameters.DATA_PARAGRAPH_MIN_LEN
max_len = shared_parameters.DATA_PARAGRAPH_MAX_LEN
min_tokens = shared_parameters.DATA_MIN_TOKENS
categories_mode = shared_parameters.DATA_CATEGORIES_MODE
return_overall = shared_parameters.DATA_RETURN_OVERALL
inputs, Y, categories, category_levels, book_ids, books_df, _, _, _ = \
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


# ## Paragraph labels

# In[ ]:


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


# In[ ]:


Q_true = np.zeros((len(categories), len(predict_source_labels)), dtype=np.int32)
for i, source_labels in enumerate(predict_source_labels):
    for j, label in enumerate(source_labels):
        Q_true[j, i] = label
Q_true[bookcave.CATEGORY_INDEX_OVERALL] = bookcave.get_y_overall(Q_true, categories_mode=categories_mode)


# In[ ]:


seed = 1
category_balanced_indices = [data_utils.get_balanced_indices_sample(q_true, minlength=len(category_levels[j]), seed=seed)
                             for j, q_true in enumerate(Q_true)]


# ### Zero Rule

# In[ ]:


def predict_zero_r(category_indices=None):
    category_metrics_zero = []
    print('ZeroR')
    for j, category in enumerate(categories):
        print()
        print(category)
        q_true = Q_true[j]
        if category_indices is not None:
            q_true = q_true[category_indices[j]]
        q_pred_zero = [np.argmax(np.bincount(q_true, minlength=len(category_levels[j])))]*len(q_true)
        confusion_zero, metrics_zero = evaluation.get_confusion_and_metrics(q_true, q_pred_zero)
        print(confusion_zero)
        for i, metric_name in enumerate(evaluation.METRIC_NAMES):
            print('{}: {:.4f}'.format(metric_name, metrics_zero[i]))
        category_metrics_zero.append(metrics_zero)
    print('\nAverage')
    metrics_avg_zero = [sum([metrics_zero[i] for metrics_zero in category_metrics_zero[:-1]])/(len(category_metrics_zero) - 1)
                        for i in range(len(category_metrics_zero[0]))]
    for i, metric_name in enumerate(evaluation.METRIC_NAMES):
        print('{}: {:.4f}'.format(metric_name, metrics_avg_zero[i]))


# In[ ]:


predict_zero_r()
print('\nBalanced')
predict_zero_r(category_indices=category_balanced_indices)


# ## Vectorize text

# In[ ]:


def identity(v):
    return v


# In[ ]:


max_words = shared_parameters.TEXT_MAX_WORDS
vectorizer = TfidfVectorizer(
    preprocessor=identity,
    tokenizer=identity,
    analyzer='word',
    token_pattern=None,
    max_features=max_words,
    norm='l2',
    sublinear_tf=True)
text_tokens = []
for source_tokens in text_source_tokens:
    all_tokens = []
    for tokens in source_tokens:
        all_tokens.extend(tokens)
    text_tokens.append(all_tokens)
X_w = vectorizer.fit_transform(text_tokens)
len(vectorizer.get_feature_names())


# In[ ]:


P_w_predict = vectorizer.transform(predict_tokens)


# In[ ]:


test_size = shared_parameters.EVAL_TEST_SIZE  # b
test_random_state = shared_parameters.EVAL_TEST_RANDOM_STATE
Y_T = Y.transpose()  # (n, c)
X_w_train, _, Y_train_T, _ = train_test_split(X_w, Y_T, test_size=test_size, random_state=test_random_state)
Y_train = Y_train_T.transpose()  # (c, n * (1 - b))


# ### Bag-of-words (count-based)

# Fit.

# In[ ]:


def fit_baseline_classifiers(create_model, X_w, Y, categories, category_levels):
    category_classifiers = []
    for j, category in enumerate(categories):
        y = Y[j]
        k = len(category_levels[j])
        classifiers = baselines.fit_ordinal(create_model, X_w, y, k)
        category_classifiers.append(classifiers)
    return category_classifiers


# In[ ]:


create_models = [
    baselines.create_k_nearest_neighbors,
    baselines.create_linear_regression,
    baselines.create_logistic_regression,
    baselines.create_multinomial_naive_bayes,
    baselines.create_random_forest,
    baselines.create_svm]
model_category_classifiers = []
for create_model in create_models:
    print('Fitting {}...'.format(create_model.__name__[7:]))
    category_classifiers = fit_baseline_classifiers(create_model, X_w_train, Y_train, categories, category_levels)
    print('Done.')
    model_category_classifiers.append(category_classifiers)


# Predict.

# In[ ]:


def evaluate_baseline_classifiers(category_classifiers, P_w_predict, Q_true, categories, category_levels, category_indices=None):
    # Predict.
    Q_w_pred = []
    for j, classifiers in enumerate(category_classifiers):
        k = len(category_levels[j])
        if category_indices is not None:
            q_w_pred = baselines.predict_ordinal(classifiers, P_w_predict[category_indices[j]], k)
        else:
            q_w_pred = baselines.predict_ordinal(classifiers, P_w_predict, k)
        Q_w_pred.append(q_w_pred)

    # Evaluate.
    category_metrics = []
    for j, category in enumerate(categories):
        print()
        print(category)
        q_true = Q_true[j]
        if category_indices is not None:
            q_true = q_true[category_indices[j]]
        q_w_pred = Q_w_pred[j]
        confusion, metrics = evaluation.get_confusion_and_metrics(q_true, q_w_pred)
        print(confusion)
        for i, metric_name in enumerate(evaluation.METRIC_NAMES):
            print('{}: {:.4f}'.format(metric_name, metrics[i]))
        category_metrics.append(metrics)

    # Average.
    print('\nAverage')
    metrics_avg = [sum([metrics[i] for metrics in category_metrics[:-1]])/(len(category_metrics) - 1)
                   for i in range(len(category_metrics[0]))]
    for i, metric_name in enumerate(evaluation.METRIC_NAMES):
        print('{}: {:.4f}'.format(metric_name, metrics_avg[i]))


# In[ ]:


for m, category_classifiers in enumerate(model_category_classifiers):
    print()
    print('{}'.format(create_models[m].__name__[7:]))
    evaluate_baseline_classifiers(category_classifiers,
                                  P_w_predict,
                                  Q_true,
                                  categories,
                                  category_levels)
    print('\nBalanced')
    evaluate_baseline_classifiers(category_classifiers,
                                  P_w_predict,
                                  Q_true,
                                  categories,
                                  category_levels,
                                  category_indices=category_balanced_indices)


# ### MLP

# In[ ]:


mlp_path = os.path.join(folders.MODELS_PATH, 'multi_layer_perceptron', 'ordinal', '33025417_overall.h5')
mlp = load_model(mlp_path)
print(mlp.summary())
predict_paragraphs.evaluate_model(mlp, P_w_predict, Q_true, categories, overall_last=return_overall)
print('\nBalanced')
predict_paragraphs.evaluate_model(mlp, P_w_predict, Q_true, categories, overall_last=return_overall, category_indices=category_balanced_indices)


# In[ ]:




