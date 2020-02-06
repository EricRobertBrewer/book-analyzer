import numpy as np
import tensorflow.keras.backend as K


def print_predictions(model, generator, categories):
    Y_pred = model.predict_generator(generator)
    print()
    for j, y_pred in enumerate(Y_pred):
        print(categories[j])
        print(y_pred)


def print_layer_outputs(model, X, Y):
    names, outputs = zip(*[(layer.name, layer.output) for layer in model.layers[1:]])  # Skip input layer.
    f = K.function([model.input, K.learning_phase()], outputs)
    instance_values = [f([x[np.newaxis, :], 0.]) for x in X]
    for name_i, name in enumerate(names):
        print('\n{} {}'.format(name, instance_values[0][name_i].shape))
        for instance_i, model_values in enumerate(instance_values):
            print('\n{}'.format(Y[:, instance_i]))
            print(model_values[name_i])
