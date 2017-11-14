import os
import processing
import numpy as np
from sklearn.datasets import fetch_rcv1
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
np.random.seed(7)


def create_dnn():
    """
    Creates a basic deep neural network model
    Input dimension matches a 47236 dimension feature vector
    3 hidden layers of 256 neurons
    """
    model = Sequential()
    model.add(Dense(256, activation='sigmoid', input_dim=47236))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(103, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    shallow = Sequential()
    shallow.add(Dense(256, activation='sigmoid', input_dim=256))
    shallow.add(Dense(103, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, shallow

def create_embedded_dnn():
    """
    Creates a neural network with an intial embedding layer
    and a convolution layer
    """
    model = Sequential()
    model.add(Embedding(max_features, 500, input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(103))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    shallow = Sequential()
    shallow.add(Dense(256, activation='sigmoid', input_dim=256))
    shallow.add(Dense(103, activation='sigmoid'))
    shallow.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, shallow

if __name__ == '__main__':
    