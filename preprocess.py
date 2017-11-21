import os
import re
import json
import random
import numpy as np
from scipy.sparse import csr_matrix, find
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
    def reusable_split(split):
        return rcv1.data[np.array(split)], rcv1.target[np.array(split)]

    data = json.load(open('categories.json'))
    rows = int(rcv1.data.shape[0])
    if split_type == 'random':
        first_split = random.sample(range(rows), int(rows/2))
        second_split = np.delete(np.arange(rows), first_split, 0)
        
    elif split_type == 'simple':
        first_split = np.arange(int(rows/2))
        second_split = np.delete(np.arange(rows), first_split, 0)

    elif split_type == 'c_topics':
        rgx = re.compile('C[0-9]*')
        splitvec = np.zeros(len(data.keys()))
        for inx, elt in enumerate(data.keys()):
            if rgx.match(elt):
                splitvec[inx] = 1

        splitvec_ind = find(csr_matrix(splitvec))[1]
        max_ind, min_ind = np.amax(splitvec_ind), np.amin(splitvec_ind)


        first_split, second_split, holdout_split = [], [], []
        for inx, elt in enumerate(rcv1.target):
            elt_ind = find(elt)[1]
            elt_max_ind, elt_min_ind = np.amax(splitvec_ind), np.amin(splitvec_ind)

            if elt_max_ind <= max_ind:
                first_split.append(inx)
            elif elt_min_ind > max_ind:
                second_split.append(inx)
            else:
                holdout_split.append(inx)
        

    first, second = reusable_split(first_split), reusable_split(second_split)

    if holdout_split:
        holdout = reusable_split(holdout_split)
        return save(split_type, first, second, holdout)

    return save(split_type, first, second, None)


def save(split_type, first, second, holdout):
    x1, y1 = first
    x2, y2 = second

    create_dirs(split_type)
    save_sparse_csr('data/{}/first_data'.format(split_type), x1)
    save_sparse_csr('data/{}/first_labels'.format(split_type), y1)
    save_sparse_csr('data/{}/second_data'.format(split_type), x2)
    save_sparse_csr('data/{}/second_labels'.format(split_type), y2)

    if holdout:
        x3, y3 = holdout
        save_sparse_csr('data/{}/holdout_data'.format(split_type), x3)
        save_sparse_csr('data/{}/holdout_labels'.format(split_type), y3)

    return x1, y1, x2, y2

def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_parse_csr(filename):
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
    split(rcv1, 'c_topics')
