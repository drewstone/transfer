import numpy as np
from sklearn.metrics import matthews_corrcoef

def matthews_correlation(model, X, Y):
	"""
	Computes matthews correlation coefficient amongst output labels
	to find best threshold for future predictions

	Returns predicted values on testset, class-level accuracies, and thesholds
	"""
	out = model.predict_proba(X)
	out = np.array(out)

	threshold = np.arange(0.1,0.9,0.1)

	acc = []
	accuracies = []
	best_threshold = np.zeros(out.shape[1])
	for i in range(out.shape[1]):
	    y_prob = np.array(out[:,i])
	    for j in threshold:
	        y_pred = [1 if prob>=j else 0 for prob in y_prob]
	        acc.append( matthews_corrcoef(Y[:,i],y_pred))
	    acc   = np.array(acc)
	    index = np.where(acc==acc.min()) 
	    accuracies.append(acc.max()) 
	    best_threshold[i] = threshold[index[0][0]]
	    acc = []

	y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(Y.shape[1])] for i in range(len(Y))])
	return y_pred, accuracies, best_threshold

def recall_accuracy(model, X, Y):
	"""
	Proportion of true labels vs. what we got
	"""
	return