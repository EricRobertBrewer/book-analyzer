import keras


def patch_tokenizer():
    """
    Fix a unicode encoding bug for Keras==2.0.5 (the version on the Fulton Supercomputer).
    https://github.com/keras-team/keras/issues/1072#issuecomment-295470970
    """

    def text_to_word_sequence(text,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True,
                              split=' '):
        if lower:
            text = text.lower()
        if type(text) == unicode:
            translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
        else:
            translate_table = keras.preprocessing.text.maketrans(filters, split * len(filters))
        text = text.translate(translate_table)
        seq = text.split(split)
        return [i for i in seq if i]

    keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence
