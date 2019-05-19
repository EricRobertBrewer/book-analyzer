import numpy as np

import tensorflow as tf
import keras
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, GlobalMaxPool1D, GRU, Input
from keras.models import Model
print('TensorFlow version: {}'.format(tf.__version__))

import bookcave


def create_model(n_tokens, emb_matrix, n_classes, train_emb=True, hidden_size=128, dense_size=64):
    inp = Input(n_tokens)
    x = Embedding(*emb_matrix.shape, weights=emb_matrix, trainable=train_emb)(inp)
    x = Bidirectional(GRU(hidden_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(dense_size, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes)(x)
    # x = Dropout(0.5)(x)
    x = Activation('sigmoid')(x)
    model = Model(inp, x)
    return model


def main():
    pass


if __name__ == '__main__':
    main()
