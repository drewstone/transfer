import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from keras.models import Model

# fix random seed for reproducibility
np.random.seed(7)

def create_dnn(first_output_dim, second_output_dim, input_dim=47236):
    """
    Creates a basic deep neural network model
    Input dimension matches a 47236 dimension feature vector
    3 hidden layers of 256 neurons
    """
    first_model = Sequential()
    first_model.add(Dense(256, activation='relu', input_dim=input_dim, name='dense-input'))
    first_model.add(Dense(256, activation='relu', name='dense-layer2'))
    first_model.add(Dense(256, activation='relu', name='dense-layer3'))
    first_model.add(Dense(256, activation='relu'))
    first_model.add(Dense(first_output_dim, activation='sigmoid'))
    first_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    intermediate = Sequential()
    intermediate.add(Dense(256, activation='relu', input_dim=47236, name='dense-input'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer2'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer3'))
    intermediate.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    intermediate_transferred_model = Sequential()
    intermediate_transferred_model.add(Dense(256, activation='relu', input_dim=input_dim, name='dense-input'))
    intermediate_transferred_model.add(Dense(256, activation='relu', name='dense-layer2'))
    intermediate_transferred_model.add(Dense(256, activation='relu', name='dense-layer3'))
    intermediate_transferred_model.add(Dense(256, activation='relu'))
    intermediate_transferred_model.add(Dense(second_output_dim, activation='sigmoid'))
    intermediate_transferred_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    second_model = Sequential()
    second_model.add(Dense(256, activation='relu', input_dim=input_dim))
    second_model.add(Dense(256, activation='relu'))
    second_model.add(Dense(256, activation='relu'))
    second_model.add(Dense(256, activation='relu'))
    second_model.add(Dense(second_output_dim, activation='sigmoid'))
    second_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    shallow = Sequential()
    shallow.add(Dense(256, activation='relu', input_dim=256))
    shallow.add(Dense(second_output_dim, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return first_model, intermediate, intermediate_transferred_model, second_model, shallow


def create_mlp(first_output_dim, second_output_dim, input_dim=47236):
    """
    Creates a basic deep neural network model
    Input dimension matches a 47236 dimension feature vector
    3 hidden layers of 256 neurons
    """
    first_model = Sequential()
    first_model.add(Dense(256, activation='relu', input_dim=input_dim, name='dense-layer1'))
    first_model.add(Dropout(rate=0.5, name='drop-layer1'))
    first_model.add(BatchNormalization(name='batch-norm2'))
    first_model.add(Dense(256, activation='relu', name='dense-layer2'))
    first_model.add(Dropout(rate=0.5, name='drop-layer2'))
    first_model.add(BatchNormalization(name='batch-norm3'))
    first_model.add(Dense(256, activation='relu', name='dense-layer3'))
    first_model.add(Dropout(rate=0.5, name='drop-layer3'))
    first_model.add(Dense(256, activation='relu'))
    first_model.add(Dense(first_output_dim, activation='sigmoid'))
    first_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    intermediate = Sequential()
    intermediate.add(Dense(256, activation='relu', input_dim=47236, name='dense-layer1'))
    intermediate.add(Dropout(rate=0.5, name='drop-layer1'))
    intermediate.add(BatchNormalization(name='batch-norm2'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer2'))
    intermediate.add(Dropout(rate=0.5, name='drop-layer2'))
    intermediate.add(BatchNormalization(name='batch-norm3'))
    intermediate.add(Dense(256, activation='relu', name='dense-layer3'))
    intermediate.add(Dropout(rate=0.5, name='drop-layer3'))
    intermediate.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    intermediate_transferred_model = Sequential()
    intermediate_transferred_model.add(Dense(256, activation='relu', input_dim=input_dim, name='dense-layer1'))
    intermediate_transferred_model.add(Dropout(rate=0.5, name='drop-layer1'))
    intermediate_transferred_model.add(BatchNormalization(name='batch-norm2'))
    intermediate_transferred_model.add(Dense(256, activation='relu', name='dense-layer2'))
    intermediate_transferred_model.add(Dropout(rate=0.5, name='drop-layer2'))
    intermediate_transferred_model.add(BatchNormalization(name='batch-norm3'))
    intermediate_transferred_model.add(Dense(256, activation='relu', name='dense-layer3'))
    intermediate_transferred_model.add(Dropout(rate=0.5, name='drop-layer3'))
    intermediate_transferred_model.add(Dense(256, activation='relu'))
    intermediate_transferred_model.add(Dense(second_output_dim, activation='sigmoid'))
    intermediate_transferred_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    second_model = Sequential()
    second_model.add(Dense(256, activation='relu', input_dim=input_dim))
    second_model.add(Dropout(rate=0.5))
    second_model.add(BatchNormalization())
    second_model.add(Dense(256, activation='relu'))
    second_model.add(Dropout(rate=0.5))
    second_model.add(BatchNormalization())
    second_model.add(Dense(256, activation='relu'))
    second_model.add(Dropout(rate=0.5))
    second_model.add(Dense(256, activation='relu'))
    second_model.add(Dense(second_output_dim, activation='sigmoid'))
    second_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    shallow = Sequential()
    shallow.add(Dense(256, activation='relu', input_dim=256))
    shallow.add(Dense(second_output_dim, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return first_model, intermediate, intermediate_transferred_model, second_model, shallow

if __name__ == '__main__':
    dm, ds = create_dnn()
    cm, cs = create_embedded_cnn()