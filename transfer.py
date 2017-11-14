import os
import preprocess
import networks
import numpy as np
from sklearn.datasets import fetch_rcv1
from keras.models import Sequential, Model
from keras.layers import Dense

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def train_and_validate(model, data, validation_split=0.33):
    """
    Trains a model over specified amount of data with specified train/validation split
    """
    X, Y = data
    history = model.fit(X, Y, validation_split=validation_split, batch_size=32)

    # Gets layer 3 output model
    intermediate_model = Model(inputs=model.input, outputs=model.layers[2].output)
    return model, intermediate_model, history

def transfer_and_repeat(intermediate_prev_model, shallow_model, data, total_amt=100000, validation_split=0.33):
    """
    Trains a new shallower network using second split of data
    given a particular data split, stored in the data directory
    """
    # Compute intermediate transformation from previous intermediate model over new data
    X, Y = data
    preds = intermediate_model.predict(X, batch_size=32)

    # Fit shallower model using predictions and labels of new data
    history = shallow_model.fit(preds, Y, validation_split=validation_split, batch_size=32)
    return shallow_model, history

def get_data(split_type, total_amt):
    data = preprocess.get_data(split_type)
    return [data[i][:total_amt] for i in range(len(data))]

def save_model(model, name):
    if not os.path.exists('./models'):
        os.makedirs('./models')

    model.save('models/{}.h5'.format(name))

if __name__ == '__main__':
    # Fetch data and make simple split of data

    amount = 100000
    val_split = 0.67

    X1, Y1, X2, Y2 = get_data('simple')
    first_half = (X1[:amount].todense(), Y1[:amount].todense())
    second_half = (X2[:amount].todense(), Y2[:amount].todense())

    # Create model templates
    m, s = networks.create_dnn()

    # Train and transfer
    model, intermediate_model, history = train_and_validate(m, data=first_half, validation_split=val_split)
    shallow_model, shallow_history = transfer_and_repeat(intermediate_model, s, data=second_half, validation_split=val_split)

    # save models for later
    save_model(model, 'simple_model')
    save_model(intermediate_model, 'simple_int_model')
    save_model(shallow_model, 'simple_shallow_model')
