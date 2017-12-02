import os
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.models import Model

# fix random seed for reproducibility
np.random.seed(7)

def create(network, first_output_dim, second_output_dim, first_layer_count, second_layer_count, interm_fraction, neuron_count, input_dim=47236):
    if network == 'dnn':
        return create_dnn(first_output_dim, second_output_dim, input_dim, first_layer_count, second_layer_count, interm_fraction, neuron_count)
    elif network == 'mlp':
        return create_mlp(first_output_dim, second_output_dim, input_dim, first_layer_count, second_layer_count, interm_fraction, neuron_count)

def create_dnn(first_output_dim, second_output_dim, input_dim, first_layer_count, second_layer_count, interm_fraction, neuron_count):
    """
    Creates a basic deep neural network model
    """
    def dnn_model_builder(layer_count, inp_dim, out_dim, loss):
        model = Sequential()
        for inx in range(int(layer_count)):
            if inx == 0:
                model.add(Dense(neuron_count, activation='relu', input_dim=inp_dim, name='dense-{}'.format(inx+1)))
            elif inx == layer_count - 1:
                model.add(Dense(out_dim, activation='softmax', name='output-softmax'))
                model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
                return model
            else:
                model.add(Dense(neuron_count, activation='relu', name='dense-{}'.format(inx+1)))

    # Create first model with specified number of layers
    first_model = dnn_model_builder(first_layer_count, input_dim, first_output_dim, loss='categorical_crossentropy')

    # Create intermediate layers with interm_fraction of first_model layers
    intermediate = Sequential()
    for inx in range(int(Decimal(first_layer_count*interm_fraction).quantize(0, ROUND_HALF_UP))):
        if inx == 0:
            intermediate.add(Dense(neuron_count, activation='relu', input_dim=input_dim, name='dense-{}'.format(inx+1)))
        else:
            intermediate.add(Dense(neuron_count, activation='relu', name='dense-{}'.format(inx+1)))
    intermediate.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


    second_model = dnn_model_builder(second_layer_count, input_dim, second_output_dim, loss='binary_crossentropy')
    latent_model = dnn_model_builder(second_layer_count, neuron_count, second_output_dim, loss='binary_crossentropy')

    return first_model, intermediate, second_model, latent_model


def create_mlp(first_output_dim, second_output_dim, input_dim, first_layer_count, second_layer_count, interm_fraction, neuron_count):
    """
    Creates a network modelled after a multilayer perceptron
    """

    def mlp_model_builder(layer_count, inp_dim, out_dim, loss):
        model = Sequential()
        for inx in range(int(layer_count)):
            if inx == 0:
                model.add(Dense(neuron_count, activation='relu', input_dim=inp_dim, name='dense-{}'.format(inx+1)))
            elif inx == layer_count - 1:
                model.add(Dense(out_dim, activation='softmax', name='output-softmax'))
                model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
                return model
            else:
                model.add(Dense(neuron_count, activation='relu', name='dense-{}'.format(inx+1)))

            model.add(Dropout(rate=0.5, name='drop-{}'.format(inx+1)))
            model.add(BatchNormalization(name='batch-{}'.format(inx+1)))

        
        
    # Create first model with specified number of layers
    first_model = mlp_model_builder(first_layer_count, input_dim, first_output_dim, loss='categorical_crossentropy')

    # Create intermediate layers with interm_fraction of first_model layers
    intermediate = Sequential()
    for inx in range(int(Decimal(first_layer_count*interm_fraction).quantize(0, ROUND_HALF_UP))):
        if inx == 0:
            intermediate.add(Dense(neuron_count, activation='relu', input_dim=input_dim, name='dense-{}'.format(inx+1)))
        else:
            intermediate.add(Dense(neuron_count, activation='relu', name='dense-{}'.format(inx+1)))
        
        intermediate.add(Dropout(rate=0.5, name='drop-{}'.format(inx+1)))
        intermediate.add(BatchNormalization(name='batch-{}'.format(inx+1)))

    intermediate.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    second_model = mlp_model_builder(second_layer_count, input_dim, second_output_dim, loss='binary_crossentropy')
    latent_model = mlp_model_builder(second_layer_count, neuron_count, second_output_dim, loss='binary_crossentropy')

    return first_model, intermediate, second_model, latent_model

if __name__ == '__main__':
    dm, ds = create_dnn()
    cm, cs = create_embedded_cnn()