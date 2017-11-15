import os
import argparse
import preprocess
import networks
import numpy as np
from sklearn.datasets import fetch_rcv1

import keras
from keras.models import Sequential, Model
from keras.layers import Dense

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def train_and_validate(model, data, validation_split=0.33, epochs=10):
    """
    Trains a model over specified amount of data with specified train/validation split
    """
    X, Y = data
    time_callback = TimeHistory()
    history = model.fit(X, Y, validation_split=validation_split, batch_size=32, epochs=epochs, callbacks=[time_callback])

    return model, history, time_callback

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
    preds = intermediate.predict(X, batch_size=32)

    # Fit shallower model using predictions and labels of new data
    time_callback = TimeHistory()
    history = shallow.fit(preds, Y, validation_split=validation_split, batch_size=32, epochs=epochs, callbacks=[time_callback])
    return shallow, history, time_callback

def get_data(split_type, amt):
    data = preprocess.get_data(split_type)
    return [data[i][:amt] for i in range(len(data))]

def save_model(model, name):
    if not os.path.exists('./models'):
        os.makedirs('./models')

    model.save('models/{}.h5'.format(name))

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cnn', '-c', action='store_true')
    parser.add_argument('--dnn', '-d', action='store_true')
    args = parser.parse_args()

    amount = 100000
    val_split = 0.67

    # Fetch data and make simple split of data
    X1, Y1, X2, Y2 = [elt.todense() for elt in get_data(split_type='simple', amt=amount)]
    
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
    first_half, second_half = (X1, Y1), (X2, Y2)

    # Train and transfer
    main, history, time_cb = train_and_validate(main, data=first_half, validation_split=val_split)
    shallow, shallow_history, time_cb = transfer_and_repeat(main, intermediate, shallow, data=second_half, validation_split=val_split)
