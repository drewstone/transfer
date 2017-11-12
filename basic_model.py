import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
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

model = Sequential()
model.add(Dense(256, activation='sigmoid', input_dim=47236))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(103, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

from keras.models import load_model

train_amt = 23149

r1_data = processing.load_sparse_csr('data/random/first_data.npz')
r1_lbls = processing.load_sparse_csr('data/random/first_labels.npz')

r1_training = (r1_data[:train_amt], r1_lbls[:train_amt])
test = (r1_data[train_amt:train_amt+50000], r1_lbls[train_amt:train_amt+50000])

model = load_model('random_split_model.h5')

from sklearn.metrics import matthews_corrcoef

out = model.predict_proba(test[0].todense())
out = np.array(out)

threshold = np.arange(0.1,0.9,0.1)

acc = []
accuracies = []
best_threshold = np.zeros(out.shape[1])
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append( matthews_corrcoef(test[1].todense()[:,i],y_pred))
    acc   = np.array(acc)
    index = np.where(acc==acc.min()) 
    accuracies.append(acc.max()) 
    best_threshold[i] = threshold[index[0][0]]
    acc = []

print("best thresholds", best_threshold)
print(len(best_threshold))
print(best_threshold*(0.85))
y_pred = np.array([[1 if out[i,j]>=best_threshold[j]*(0.65) else 0 for j in range(test[1].todense().shape[1])] for i in range(len(test[1].todense()))])

print("-"*40)
print("Matthews Correlation Coefficient")
print("Class wise accuracies")
print(accuracies)

print("other statistics\n")
total_correctly_predicted = len([i for i in range(len(test[1].todense())) if (test[1].todense()[i]==y_pred[i]).sum() == 5])
print("Fully correct output")
print(total_correctly_predicted)
print(total_correctly_predicted/400.)