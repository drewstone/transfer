import os
import random
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_rcv1


np.random.seed(7)
random.seed(7)

def create_dirs(name):
    if not os.path.exists('./data'):
        os.makedirs('./data')

    if not os.path.exists('./data/{}'.format(name)):
        os.makedirs('./data/{}'.format(name))

def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def random_split(rcv1_obj):
    create_dirs('random')
    rows = int(rcv1_obj.data.shape[0])
    first_split = random.sample(range(rows), int(rows/2))
    second_split = np.delete(np.arange(rows), first_split, 0)
    
    first_data = rcv1_obj.data[np.array(first_split)]
    first_labels = rcv1_obj.target[np.array(first_split)]
    second_data = rcv1_obj.data[np.array(second_split)]
    second_labels = rcv1_obj.target[np.array(second_split)]
    
    save_sparse_csr('data/random/first_data', first_data)
    save_sparse_csr('data/random/first_labels', first_labels)
    save_sparse_csr('data/random/second_data', second_data)
    save_sparse_csr('data/random/second_labels', second_labels)

def simple_split(rcv1_obj):
    create_dirs('simple')
    rows = int(rcv1_obj.data.shape[0])
    first_split = np.arange(int(rows/2))
    second_split = np.delete(np.arange(rows), first_split, 0)
    
    first_data = rcv1_obj.data[np.array(first_split)]
    first_labels = rcv1_obj.target[np.array(first_split)]
    second_data = rcv1_obj.data[np.array(second_split)]
    second_labels = rcv1_obj.target[np.array(second_split)]
    
    save_sparse_csr('data/simple/first_data', first_data)
    save_sparse_csr('data/simple/first_labels', first_labels)
    save_sparse_csr('data/simple/second_data', second_data)
    save_sparse_csr('data/simple/second_labels', second_labels)

if __name__ == '__main__':
    np.random.seed(7)
    rcv1 = fetch_rcv1()
    random_split(rcv1)
    simple_split(rcv1)