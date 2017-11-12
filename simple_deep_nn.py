import os
import processing
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def create_main_model():
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
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
	return model

def create_shallow_model():
	"""
	Creates the shallower version of the DNN above
	"""
	model = Sequential()
	model.add(Dense(256, activation='sigmoid', input_dim=256))
	model.add(Dense(103, activation='sigmoid'))
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

def train_and_validate(model, data_split='random', total_amt=100000, validation_split=0.33):
	"""
	Trains a model over specified amount of data with specified train/validation split
	"""
	data = processing.load_sparse_csr('data/{}/first_data.npz'.format(data_split))[:total_amt]
	lbls = processing.load_sparse_csr('data/{}/first_labels.npz'.format(data_split))[:total_amt]
	history = model.fit(data.todense(), lbls.todense(), validation_split=validation_split, batch_size=32)

	# Gets layer 3 output model
	intermediate_model = Model(inputs=model.input, outputs=model.layers[2].output)

	# Save models for future use
	save_model(model, data_split, 'model')
	save_model(intermediate_model, data_split, 'intermediate_model')

	return model, intermediate_model, history

def transfer_and_repeat(intermediate_prev_model, shallow_model, data_split='random', total_amt=100000, validation_split=0.33):
	"""
	Trains a new shallower network using second split of data
	given a particular data split, stored in the data directory
	"""
	data = processing.load_sparse_csr('data/{}/second_data.npz'.format(data_split))[:total_amt]
	lbls = processing.load_sparse_csr('data/{}/second_labels.npz'.format(data_split))[:total_amt]

	# Compute intermediate transformation from previous intermediate model over new data
	preds = intermediate_model.predict(data.todense(), batch_size=32)

	# Fit shallower model using predictions and labels of new data
	history = shallow_model.fit(preds, lbls.todense(), validation_split=validation_split, batch_size=32)

	# Save model for future use
	save_model(shallow_model, data_split, 'shallow_model')

	return shallow_model, history

def save_model(model, data_split, name):
	dirpath = './models/{}'.format(data_split)
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

	model.save('{}/{}.h5'.format(dirpath, name))

if __name__ == '__main__':
	m = create_main_model()
	s = create_shallow_model()

	model, intermediate_model, history = train_and_validate(m, data_split='simple', total_amt=100000, validation_split=0.33)
	shallow_model, shallow_history = transfer_and_repeat(intermediate_model, s, data_split='simple', total_amt=100000, validation_split=0.33)
