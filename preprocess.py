import os
import re
import json
import random
import argparse
import numpy as np
from scipy.sparse import csr_matrix, find
from sklearn.datasets import fetch_rcv1


np.random.seed(7)
random.seed(7)

def get_data(split_type='random'):
    if os.path.exists('./data/features.npz') and os.path.exists('./data/labels.npz'):
        ftrs = load_sparse_csr('data/features.npz')
        lbls = load_sparse_csr('data/labels.npz')

        if (os.path.exists('./data/{}_first.npy'.format(split_type))
                and os.path.exists('./data/{}_second.npy'.format(split_type))):
            # Get split indices array
            first_ind = np.load('data/{}_first.npy'.format(split_type))
            second_ind = np.load('data/{}_second.npy'.format(split_type))
            holdout = None

            if split_type in ['c_topics', 'e_topics', 'g_topics', 'm_topics']:
                holdout_ind = np.load('data/{}_holdout.npy'.format(split_type))
                holdout = reusable_split(ftrs, lbls, holdout_ind)
            
            first = reusable_split(ftrs, lbls, first_ind)
            second = reusable_split(ftrs, lbls, second_ind)
            
            return first, second, holdout
        else:
            return split(ftrs, lbls, split_type)
            
    else:
        rcv1 = fetch_rcv1()
        save_sparse_csr('data/features', rcv1.data)
        save_sparse_csr('data/labels', rcv1.target)
        return split(rcv1.data, rcv1.target, split_type)

def reusable_split(features, labels, indices):
    return features[np.array(indices)], labels[np.array(indices)]

def split(features, labels, split_type, holdout_split=None):
    def filter_split(min_ind, max_ind):
        f, s, h = [], [], []
        for inx, elt in enumerate(labels):
            elt_ind = find(elt)[1]
            elt_max_ind, elt_min_ind = np.amax(elt_ind), np.amin(elt_ind)

            if elt_max_ind <= max_ind and elt_min_ind >= min_ind:
                f.append(inx)
            elif elt_min_ind > max_ind or elt_max_ind < min_ind:
                s.append(inx)
            else:
                h.append(inx)

        return f, s, h

    def regex_splitter(regex):
        data = json.load(open('categories.json'))
        splitvec = np.zeros(len(data.keys()))
        for inx, elt in enumerate(data.keys()):
            if regex.match(elt):
                splitvec[inx] = 1
        
        splitvec_ind = find(csr_matrix(splitvec))[1]
        max_ind, min_ind = np.amax(splitvec_ind), np.amin(splitvec_ind)
        return min_ind, max_ind

    rows = int(features.shape[0])

    if split_type == 'random':
        first_split = random.sample(range(rows), int(rows/2))
        second_split = np.delete(np.arange(rows), first_split, 0)
        save(split_type, first_split, second_split)
        
    elif split_type == 'simple':
        first_split = np.arange(int(rows/2))
        second_split = np.delete(np.arange(rows), first_split, 0)
        save(split_type, first_split, second_split)

    elif split_type == 'c_topics':
        rgx = re.compile('C[0-9]*')
        min_ind, max_ind = regex_splitter(rgx)
        first_split, second_split, holdout_split = filter_split(min_ind, max_ind)
        save(split_type, first_split, second_split, holdout_split)

    elif split_type == 'g_topics':
        rgx = re.compile('G[0-9]*')
        min_ind, max_ind = regex_splitter(rgx)
        first_split, second_split, holdout_split = filter_split(min_ind, max_ind)
        save(split_type, first_split, second_split, holdout_split)

    elif split_type == 'e_topics':
        rgx = re.compile('E[0-9]*')
        min_ind, max_ind = regex_splitter(rgx)
        first_split, second_split, holdout_split = filter_split(min_ind, max_ind)
        save(split_type, first_split, second_split, holdout_split)

    elif split_type == 'm_topics':
        rgx = re.compile('M[0-9]*')
        min_ind, max_ind = regex_splitter(rgx)
        first_split, second_split, holdout_split = filter_split(min_ind, max_ind)
        save(split_type, first_split, second_split, holdout_split)

    first = reusable_split(features, labels, first_split)
    second = reusable_split(features, labels, second_split)
    holdout = None

    if holdout_split:
        holdout = reusable_split(features, labels, holdout_split)

    return first, second, holdout


def save(split_type, first, second, holdout=None):
    np.save('data/{}_first'.format(split_type), first)
    np.save('data/{}_second'.format(split_type), second)

    if holdout:
        np.save('data/{}_holdout'.format(split_type), holdout)

def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--random', '-r', action='store_true')
    parser.add_argument('--simple', '-s', action='store_true')
    parser.add_argument('--ctopics', '-c', action='store_true')
    parser.add_argument('--gtopics', '-g', action='store_true')
    parser.add_argument('--etopics', '-e', action='store_true')
    parser.add_argument('--mtopics', '-m', action='store_true')

    args = parser.parse_args()

    if args.random:
        ftrs, lbls, holdout = get_data('random')
    elif args.simple:
        ftrs, lbls, holdout = get_data('simple')
    elif args.ctopics:
        ftrs, lbls, holdout = get_data('c_topics')
    elif args.gtopics:
        ftrs, lbls, holdout = get_data('g_topics')
    elif args.etopics:
        ftrs, lbls, holdout = get_data('e_topics')
    elif args.mtopics:
        ftrs, lbls, holdout = get_data('m_topics')
    else:
        ftrs, lbls, holdout = [], [], []

    print(ftrs, lbls, holdout)
