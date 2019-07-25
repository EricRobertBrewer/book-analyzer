import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attention_with_context import AttentionWithContext
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Input, Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.models import Model
from nltk import tokenize


class HAN(object):
    """
    HAN model is implemented here.
    """

    def __init__(self,
                 text_paragraph_tokens,
                 Y,
                 category_levels,
                 embedding_matrix,
                 n_tokens,
                 n_paragraphs,
                 ordinal=False,
                 validation_split=0.2,
                 verbose=0):
        """
        Initialize the HAN module
        Keyword arguments:
        text -- list of the articles for training.
        labels -- labels corresponding the given `text`.
        pretrained_embedded_vector_path -- path of any pretrained vector
        max_features -- max features embeddeding matrix can have. To more checkout https://keras.io/layers/embeddings/
        max_senten_len -- maximum sentence length. It is recommended not to use the maximum one but the one that covers 0.95 quatile of the data.
        max_senten_num -- maximum number of sentences. It is recommended not to use the maximum one but the one that covers 0.95 quatile of the data.
        embedding_size -- size of the embedding vector
        num_categories -- total number of categories.
        validation_split -- train-test split.
        verbose -- how much you want to see.
        """
        try:
            self.text_paragraph_tokens = text_paragraph_tokens
            self.Y = Y
            self.category_levels = category_levels
            self.embedding_matrix = embedding_matrix
            self.n_tokens = n_tokens
            self.n_paragraphs = n_paragraphs
            self.verbose = verbose
            # self.max_senten_len = max_senten_len
            # self.max_senten_num = max_senten_num
            self.validation_split = validation_split
            self.ordinal = ordinal
            # Initialize default hyperparameters
            # You can change it using `set_hyperparameters` function
            self.hyperparameters = {
                'l2_regularizer': None,
                'dropout_regularizer': None,
                'rnn': LSTM,
                'rnn_units': 150,
                'dense_units': 200,
                'activation': 'softmax',
                'optimizer': 'adam',
                'metrics': ['acc'],
                'loss': 'categorical_crossentropy'
            }
            if num_categories is not None:
                assert (num_categories == len(self.classes))
            assert (self.text.shape[0] == self.categories.shape[0])
            self.word_index = None
            self.x_train, self.y_train, self.x_val, self.y_val = self.split_dataset()
            self.model = None
            self.set_model()
            self.history = None
        except AssertionError:
            print('Input and label data must be of same size')

    def set_hyperparameters(self, tweaked_instances):
        """
        Set hyperparameters of HAN model.
        Keywords arguments:
        tweaked_instances -- dictionary of all those keys you want to change
        """
        for key, value in tweaked_instances.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
            else:
                raise KeyError(key + ' does not exist in hyperparameters')
            self.set_model()

    def show_hyperparameters(self):
        """
        To check the values of all the current hyperparameters
        """
        print('Hyperparameter\tCorresponding Value')
        for key, value in self.hyperparameters.items():
            print(key, '\t\t', value)

    def clean_string(self, string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()

    def split_dataset(self):
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels.iloc[indices]
        nb_validation_samples = int(self.validation_split * self.data.shape[0])

        x_train = self.data[:-nb_validation_samples]
        y_train = self.labels[:-nb_validation_samples]
        x_val = self.data[-nb_validation_samples:]
        y_val = self.labels[-nb_validation_samples:]
        if self.verbose == 1:
            print('Number of positive and negative reviews in traing and validation set')
            print(y_train.columns.tolist())
            print(y_train.sum(axis=0).tolist())
            print(y_val.sum(axis=0).tolist())
        return x_train, y_train, x_val, y_val

    def get_model(self):
        """
        Returns the HAN model so that it can be used as a part of pipeline
        """
        return self.model

    def get_embedding_layer(self):
        """
        Returns Embedding layer
        """
        return

    def set_model(self):
        """
        Set the HAN model according to the given hyperparameters
        """
        if self.hyperparameters['l2_regulizer'] is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = regularizers.l2(self.hyperparameters['l2_regulizer'])
        if self.hyperparameters['dropout_regulizer'] is None:
            dropout_regularizer = 1
        else:
            dropout_regularizer = self.hyperparameters['dropout_regulizer']
        word_input = Input(shape=(self.max_senten_len,), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)
        word_lstm = Bidirectional(
            self.hyperparameters['rnn'](self.hyperparameters['rnn_units'], return_sequences=True,
                                        kernel_regularizer=kernel_regularizer))(word_sequences)
        word_dense = TimeDistributed(
            Dense(self.hyperparameters['dense_units'], kernel_regularizer=kernel_regularizer))(word_lstm)
        word_att = AttentionWithContext()(word_dense)
        wordEncoder = Model(word_input, word_att)

        sent_input = Input(shape=(self.max_senten_num, self.max_senten_len), dtype='float32')
        sent_encoder = TimeDistributed(wordEncoder)(sent_input)
        sent_lstm = Bidirectional(self.hyperparameters['rnn'](
            self.hyperparameters['rnn_units'], return_sequences=True, kernel_regularizer=kernel_regularizer))(
            sent_encoder)
        sent_dense = TimeDistributed(
            Dense(self.hyperparameters['dense_units'], kernel_regularizer=kernel_regularizer))(sent_lstm)
        sent_att = Dropout(dropout_regularizer)(
            AttentionWithContext()(sent_dense))
        preds = Dense(len(self.classes))(sent_att)
        self.model = Model(sent_input, preds)
        self.model.compile(
            loss=self.hyperparameters['loss'], optimizer=self.hyperparameters['optimizer'],
            metrics=self.hyperparameters['metrics'])

    def train_model(self, epochs, batch_size, best_model_path=None, final_model_path=None, plot_learning_curve=True):
        """
        Training the model
        epochs -- Total number of epochs
        batch_size -- size of a batch
        best_model_path -- path to save best model i.e. weights with lowest validation score.
        final_model_path -- path to save final model i.e. final weights
        plot_learning_curve -- Want to checkout Learning curve
        """
        if best_model_path is not None:
            checkpoint = ModelCheckpoint(best_model_path, verbose=0, monitor='val_loss', save_best_only=True,
                                         mode='auto')
        self.history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                                      epochs=epochs, batch_size=batch_size, verbose=self.verbose,
                                      callbacks=[checkpoint])
        if plot_learning_curve:
            self.plot_results()
        if final_model_path is not None:
            self.model.save(final_model_path)

    def plot_results(self):
        """
        Plotting learning curve of last trained model.
        """
        # summarize history for accuracy
        plt.subplot(211)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # summarize history for loss
        plt.subplot(212)
        plt.plot(self.history.history['val_loss'])
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        time.sleep(10)
        plt.close()
