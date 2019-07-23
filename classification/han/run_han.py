from keras.layers import GRU
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from classification.han import HAN
import folders
from sites.bookcave import bookcave
from text import load_embeddings


def main():
    """
    A small tutorial to use HAN module
    """
    # Load data.
    min_len, max_len = 250, 7500
    inputs, Y, categories, category_levels = \
        bookcave.get_data({'tokens'},
                          min_len=min_len,
                          max_len=max_len,
                          only_ids={'torture_mom'})
    text_paragraph_tokens, _ = zip(*inputs['tokens'])
    print('{:d} texts'.format(len(text_paragraph_tokens)))

    # Flatten tokens.
    all_tokens = []
    for text_i, paragraph_tokens in enumerate(text_paragraph_tokens):
        for paragraph_i, tokens in enumerate(paragraph_tokens):
            all_tokens.append(tokens)

    # Tokenize.
    max_words = 8192  # The maximum size of the vocabulary.
    tokenizer = Tokenizer(num_words=max_words, oov_token='__UNKNOWN__')
    tokenizer.fit_on_texts(all_tokens)

    # Convert to sequences.
    n_tokens = 128  # The number of tokens to process in each paragraph.
    text_paragraph_sequences = [pad_sequences(tokenizer.texts_to_sequences(paragraph_tokens), maxlen=n_tokens)
                                for paragraph_tokens in text_paragraph_tokens]

    # Load embedding.
    embedding_matrix = load_embeddings.get_embedding(tokenizer, folders.EMBEDDING_GLOVE_100_PATH, max_words)

    # Create model.
    n_classes = [len(levels) for levels in category_levels]
    n_paragraphs = 1024
    n_tokens_per_paragraph = 128
    model = HAN.create_model(
        n_classes,
        n_tokens,
        embedding_matrix,
        n_paragraphs,
        n_tokens_per_paragraph,
        rnn=GRU)

    # Compile.
    loss = 'binary_crossentropy'
    optimizer = Adam()
    metrics = ['binary_accuracy']
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)

    # Train.
    batch_size = 32
    epochs = 1
    model.fit(text_paragraph_tokens, Y, batch_size=batch_size, epochs=epochs)
    # han = HAN.HAN(text_paragraph_tokens,
    #               Y,
    #               category_levels,
    #               embedding_matrix,
    #               n_tokens,
    #               n_paragraphs,
    #               ordinal=True,
    #               validation_split=0.2,
    #               verbose=1)
    # print(han.get_model().summary())
    # han.set_hyperparameters({
    #     'l2_regularizer': 1e-13,
    #     'dropout_regularizer': 0.5,
    #     'rnn': GRU,
    #     'rnn_units': 128,
    #     'dense_units': 64,
    #     'activation': 'sigmoid',
    #     'optimizer': 'adam',
    #     'loss': 'binary_crossentropy'
    # })
    # han.show_hyperparameters()
    # print(han.get_model().summary())
    # han.train_model(epochs=3,
    #                 batch_size=16,
    #                 best_model_path='./best_model.h5')


if __name__ == '__main__':
    main()
