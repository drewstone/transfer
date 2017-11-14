import os
import random
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_rcv1


np.random.seed(7)
random.seed(7)

def get_data(split_type='random'):
    if os.path.exists('./data/{}'.format(split_type)):
        X1 = load_sparse_csr('data/{}/first_data.npz'.format(split_type))
        Y1 = load_sparse_csr('data/{}/first_labels.npz'.format(split_type))
        X2 = load_sparse_csr('data/{}/second_data.npz'.format(split_type))
        Y2 = load_sparse_csr('data/{}/second_labels.npz'.format(split_type))
        return X1, Y1, X2, Y2
    else:
        rcv1 = fetch_rcv1()
        return split(rcv1, split_type)

def split(rcv1, split_type):
    rows = int(rcv1.data.shape[0])
    if split_type == 'random':
        first_split = random.sample(range(rows), int(rows/2))
        second_split = np.delete(np.arange(rows), first_split, 0)
    elif split_type == 'simple':
        first_split = np.arange(int(rows/2))
        second_split = np.delete(np.arange(rows), first_split, 0)

    return split_and_save(rcv1, split_type, first_split, second_split)


def split_and_save(rcv1, split_type, first_split, second_split):
    x1, y1 = rcv1.data[np.array(first_split)], rcv1.target[np.array(first_split)]
    x2, y2 = rcv1.data[np.array(second_split)], rcv1.target[np.array(second_split)]

    create_dirs(split_type)
    save_sparse_csr('data/{}/first_data'.format(split_type), x1)
    save_sparse_csr('data/{}/first_labels'.format(split_type), y1)
    save_sparse_csr('data/{}/second_data'.format(split_type), x2)
    save_sparse_csr('data/{}/second_labels'.format(split_type), y2)

    return x1, y1, x2, y2

def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def create_dirs(name):
    if not os.path.exists('./data'):
        os.makedirs('./data')

    if not os.path.exists('./data/{}'.format(name)):
        os.makedirs('./data/{}'.format(name))

if __name__ == '__main__':
    rcv1 = fetch_rcv1()
    random_split(rcv1)
    simple_split(rcv1)