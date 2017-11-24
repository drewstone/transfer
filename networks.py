import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from keras.models import Model

# fix random seed for reproducibility
np.random.seed(7)


def create_dnn():
    """
    Creates a basic deep neural network model
    Input dimension matches a 47236 dimension feature vector
    3 hidden layers of 256 neurons
    """
    model = Sequential()
    model.add(Dense(256, activation='sigmoid', input_dim=47236, name='dense-input'))
    model.add(Dense(256, activation='sigmoid', name='dense-layer2'))
    model.add(Dense(256, activation='sigmoid', name='dense-layer3'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(103, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    intermediate = Sequential()
    intermediate.add(Dense(256, activation='sigmoid', input_dim=47236, name='dense-input'))
    intermediate.add(Dense(256, activation='sigmoid', name='dense-layer2'))
    intermediate.add(Dense(256, activation='sigmoid', name='dense-layer3'))
    intermediate.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    shallow = Sequential()
    shallow.add(Dense(256, activation='sigmoid', input_dim=256))
    shallow.add(Dense(103, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, intermediate, shallow


def create_mlp():
    """
    Creates a basic deep neural network model
    Input dimension matches a 47236 dimension feature vector
    3 hidden layers of 256 neurons
    """
    model = Sequential()
    model.add(BatchNormalization(input_shape=(47236, 1), name='batch-norm1'))
    model.add(Dense(256, activation='relu', name='dense-layer1'))
    model.add(Dropout(rate=0.5, name='drop-layer1'))
    model.add(BatchNormalization(name='batch-norm2'))
    model.add(Dense(256, activation='relu', name='dense-layer2'))
    model.add(Dropout(rate=0.5, name='drop-layer2'))
    model.add(BatchNormalization(name='batch-norm3'))
    model.add(Dense(256, activation='relu', name='dense-layer3'))
    model.add(Dropout(rate=0.5, name='drop-layer3'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(103, activation='sigmoid'))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    intermediate = Sequential()
    intermediate.add(BatchNormalization(input_shape=(47236, 1), name='batch-norm1'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer1'))
    intermediate.add(Dropout(rate=0.5, name='drop-layer1'))
    intermediate.add(BatchNormalization(name='batch-norm2'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer2'))
    intermediate.add(Dropout(rate=0.5, name='drop-layer2'))
    intermediate.add(BatchNormalization(name='batch-norm3'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer3'))
    intermediate.add(Dropout(rate=0.5, name='drop-layer3'))
    intermediate.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    shallow = Sequential()
    shallow.add(Dense(256, activation='sigmoid', input_dim=256))
    shallow.add(Dense(103, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, intermediate, shallow

def create_cnn():
    """
    Creates a simple convolutional net
    with a single conv layer
    """
    model = Sequential()
    model.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1, input_shape=(47236, 1), name='conv-input'))
    model.add(GlobalMaxPooling1D(name='gmp-layer'))
    model.add(Dense(256, activation='relu', name='dense-layer1'))
    model.add(Dropout(0.2, name='drop-layer1'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(103, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    intermediate = Sequential()
    intermediate.add(Conv1D(64, 3, padding='valid', activation='relu', strides=1, input_shape=(47236, 1), name='conv-input'))
    intermediate.add(GlobalMaxPooling1D(name='gmp-layer'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer1'))
    intermediate.add(Dropout(0.2, name='drop-layer1'))
    intermediate.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    shallow.add(Dense(256, activation='sigmoid'))
    shallow.add(Dense(103, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, intermediate, shallow

def create_wider_cnn():
    """
    Creates a simple convolutional net
    with a single conv layer
    """
    model = Sequential()
    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1, input_shape=(47236, 1), name='conv-input'))
    model.add(GlobalMaxPooling1D(name='gmp-layer'))
    model.add(Dense(256, activation='relu', name='dense-layer1'))
    model.add(Dropout(0.2, name='drop-layer1'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(103, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    intermediate = Sequential()
    intermediate.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1, input_shape=(47236, 1), name='conv-input'))
    intermediate.add(GlobalMaxPooling1D(name='gmp-layer'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer1'))
    intermediate.add(Dropout(0.2, name='drop-layer1'))
    intermediate.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    shallow.add(Dense(256, activation='sigmoid'))
    shallow.add(Dense(103, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, intermediate, shallow

def create_large_cnn():
    """
    Creates a complex convolutional neural network
    with multiple convolutional filters
    """
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(47236, 1), name='conv-input'))
    model.add(Conv1D(64, 3, activation='relu', name='conv-layer2'))
    model.add(MaxPooling1D(3, name='mp-layer'))
    model.add(Conv1D(128, 3, activation='relu', name='conv-layer3'))
    model.add(Conv1D(128, 3, activation='relu', name='conv-layer4'))
    model.add(GlobalAveragePooling1D(name='gp-layer'))
    model.add(Dropout(0.5, name='dropout'))
    model.add(Dense(256, activation='sigmoid', name='dense-layer'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(103, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    intermediate = Sequential()
    intermediate.add(Conv1D(64, 3, activation='relu', input_shape=(47236, 1), name='conv-input'))
    intermediate.add(Conv1D(64, 3, activation='relu', name='conv-layer2'))
    intermediate.add(MaxPooling1D(3, name='mp-layer'))
    intermediate.add(Conv1D(128, 3, activation='relu', name='conv-layer3'))
    intermediate.add(Conv1D(128, 3, activation='relu', name='conv-layer4'))
    intermediate.add(GlobalAveragePooling1D(name='gp-layer'))
    intermediate.add(Dropout(0.5, name='dropout'))
    intermediate.add(Dense(256, activation='sigmoid', name='dense-layer'))
    intermediate.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    shallow = Sequential()
    shallow.add(Dense(256, activation='sigmoid', input_dim=128))
    shallow.add(Dense(103, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, intermediate, shallow

if __name__ == '__main__':
    dm, ds = create_dnn()
    cm, cs = create_embedded_cnn()