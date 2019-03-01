import numpy as np
import keras
from keras.layers import Conv2D, Dense, Dropout, Input, Flatten, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
import bookcave


def get_model(images_size, levels):
    inp = Input((*images_size, 3))

    x = Conv2D(32, (3, 3), input_shape=(3, *images_size), activation='relu')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = [Dense(len(levels[i]), activation='sigmoid')(x) for i in range(len(levels))]
    model = Model(inp, outputs)
    optimizer = Adam()
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    images_size = (512, 512)
    inputs, Y, categories, levels = bookcave.get_data({'text', 'images'},
                                                      text_input='filename',
                                                      images_source='cover',
                                                      images_size=images_size,
                                                      only_categories={1, 3, 5, 6})
    images = [load_img(book_images[0]) for book_images in inputs['images']]
    X = np.array([img_to_array(image) for image in images])
    print(X.shape)

    model = get_model(images_size, levels)


if __name__ == '__main__':
    main()
