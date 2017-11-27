import os
import argparse
import numpy as np
from sklearn.datasets import fetch_rcv1

import keras
from keras.models import Sequential, Model
from keras.layers import Dense

import preprocess
import networks
import callbacks
import plotting

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def train_and_validate(model, data, validation_split=0.33, epochs=10):
    """
    Trains a model over specified amount of data with specified train/validation split
    """
    X, Y = data
    cbs = callbacks.get_callbacks(name="initial_training")
    history = model.fit(X, Y, validation_split=validation_split, batch_size=64, epochs=epochs, callbacks=cbs)

    return model, history, cbs

def transfer_and_repeat(model, intermediate, shallow, data, validation_split=0.33, epochs=10):
    """
    Trains a new shallower network using second split of data
    given a particular data split, stored in the data directory
    """
    X, Y = data

    # Save model weights to load into intermediate model
    save_model(model, 'temp_model')
    intermediate.load_weights('models/temp_model.h5', by_name=True)

    # Compute intermediate transformation from previous intermediate model over new data
    preds = intermediate.predict(X, batch_size=64)

    # Fit shallower model using predictions and labels of new data
    cbs = callbacks.get_callbacks(name="transfer_training")
    history = shallow.fit(preds, Y, validation_split=validation_split, batch_size=64, epochs=epochs, callbacks=cbs)
    
    return intermediate, shallow, history, cbs

def get_data(split_type, amt):
    data = preprocess.get_data(split_type)
    data = tuple(filter(lambda x: x, data))

    if split_type in ["random", "simple"]:
        first, second = data
        X1, Y1 = first
        X2, Y2 = second
        return X1[:amt].todense(), Y1[:amt].todense(), X2[:amt].todense(), Y2[:amt].todense(), X1[amt:amt+50000].todense(), Y1[amt:amt+50000].todense()
    elif split_type in ["c_topics", "g_topics", "e_topics", "m_topics"]:
        first, second, holdout = data
        X1, Y1 = first
        X2, Y2 = second
        X3, Y3 = holdout
        return X1[:amt].todense(), Y1[:amt].todense(), X2[:amt].todense(), Y2[:amt].todense(), X3[:50000].todense(), Y3[:50000].todense()

def save_model(model, name):
    if not os.path.exists('./models'):
        os.makedirs('./models')

    model.save('models/{}.h5'.format(name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cnn', '-c', action='store_true')
    parser.add_argument('--dnn', '-d', action='store_true')
    args = parser.parse_args()

    amount = 1000
    val_split = 0.67

    # Fetch data and make simple split of data
    X1, Y1, X2, Y2 = get_data(split_type='simple', amt=amount)

    if args.cnn:
        # Need to expand dimension for CNN to make sense
        X1 = np.expand_dims(X1, axis=2)
        X2 = np.expand_dims(X2, axis=2)
        main, intermediate, shallow = networks.create_cnn()
    elif args.dnn:
        main, intermediate, shallow = networks.create_dnn()
    else:
        main, intermediate, shallow = networks.create_dnn()

    # Split data for training/testing for before and after transfer
    first_half = (X1, Y1) 
    second_half = (X2, Y2)

    # Train and transfer
    main, history, cbs = train_and_validate(main, data=first_half, validation_split=val_split)
    intermediate, shallow, shallow_history, shallow_cbs = transfer_and_repeat(main, intermediate, shallow, data=second_half, validation_split=val_split)
    plotting.plot_acc(history, name="main")
    plotting.plot_acc(shallow_history, name="shallow")
