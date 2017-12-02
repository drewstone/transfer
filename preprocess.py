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
        features = load_sparse_csr('data/features.npz')
        labels = load_sparse_csr('data/labels.npz')
        if ((os.path.exists('./data/first-indices-{}.npy'.format(split_type)) 
                and os.path.exists('./data/second-indices-{}.npy'.format(split_type)))
                or os.path.exists('./data/first-labels-{}.npy'.format(split_type))):
            
            if not split_type == 'coarse':
                first_features = load_sparse_csr('./data/first-features-{}.npz'.format(split_type))
                first_labels = load_sparse_csr('./data/first-labels-{}.npz'.format(split_type))

                second_features = load_sparse_csr('./data/second-features-{}.npz'.format(split_type))
                second_labels = load_sparse_csr('./data/second-labels-{}.npz'.format(split_type))

                # Get split indices array
                first_ind = np.load('./data/first-indices-{}.npy'.format(split_type))
                second_ind = np.load('./data/second-indices-{}.npy'.format(split_type))
                return first_features, first_labels, second_features, second_labels, (first_ind, second_ind)
            else:
                first_labels = np.load('./data/first-labels-coarse.npy')
                second_labels = np.load('./data/second-labels-coarse.npy')
                return features, first_labels, second_labels
        else:
            if split_type == 'coarse':
                return coarse_categorical_split(features, labels)
            else:
                return split(features, labels, split_type)
            
    else:
        rcv1 = fetch_rcv1()
        save_sparse_csr('data/features', rcv1.data)
        save_sparse_csr('data/labels', rcv1.target)

        if split_type == 'coarse':
            return coarse_categorical_split(features, labels)
        else:
            return split(features, labels, split_type)

def reusable_split(features, labels, indices):
    return features[np.array(indices)], labels[np.array(indices)]

def coarse_categorical_split(features, labels):
    cmin, cmax = regex_splitter(re.compile('C[0-9]*'))
    emin, emax = regex_splitter(re.compile('E[0-9]*'))
    gmin, gmax = regex_splitter(re.compile('G[0-9]*'))
    mmin, mmax = regex_splitter(re.compile('M[0-9]*'))

    carr = np.arange(cmin, cmax+1)
    earr = np.arange(emin, emax+1)
    garr = np.arange(gmin, gmax+1)
    marr = np.arange(mmin, mmax+1)

    hots = []
    binary_hots = []
    for inx, elt in enumerate(labels):
        label = np.array(elt.todense())[0]

        cvote = label[carr].sum()
        evote = label[earr].sum()
        gvote = label[garr].sum()
        mvote = label[marr].sum()

        maxi = 0
        index = 0
        for iinx, elt in enumerate([cvote, evote, gvote, mvote]):
            if elt > maxi:
                maxi = elt
                index = iinx

        hot_arr = np.zeros(3)
        if index < 3:
            hot_arr[index] = 1
            binary_hot = 0
        else:
            binary_hot = 1

        hots.append(hot_arr)
        binary_hots.append(binary_hot)

    np.save('./data/first-labels-coarse', hots)
    np.save('./data/second-labels-coarse', binary_hots)
    return features, hots, binary_hots


def regex_splitter(regex):
    data = json.load(open('categories.json'))
    splitvec = np.zeros(len(data.keys()))
    for inx, elt in enumerate(data.keys()):
        if regex.match(elt):
            splitvec[inx] = 1
    
    splitvec_ind = find(csr_matrix(splitvec))[1]
    max_ind, min_ind = np.amax(splitvec_ind), np.amin(splitvec_ind)
    return min_ind, max_ind

def split(features, labels, split_type, holdout_split=None):
    def filter_split(first_ind, second_ind):
        f, s = [], []

        for inx, elt in enumerate(labels):
            elt_ind = find(elt)[1]
            first_flag = False
            second_flag = False

            for i in elt_ind:
                if i in first_ind and not first_flag:
                    f.append(inx)
                    first_flag = True
                if i in second_ind and not second_flag:
                    s.append(inx)
                    second_flag = True

                if first_flag and second_flag:
                    break
        return f, s

    cat_count = int(labels.shape[1])

    if split_type == 'random':
        first_ind = random.sample(range(cat_count), int(np.floor(float(cat_count)/2)) - 1)
        second_ind = np.delete(np.arange(cat_count), first_ind)
        
    elif split_type == 'simple':
        first_ind = np.arange(int(np.floor(float(cat_count)/2)))
        second_ind = np.arange(int(np.floor(float(cat_count)/2)), cat_count)

    elif split_type == 'c_topics':
        rgx = re.compile('C[0-9]*')
        min_ind, max_ind = regex_splitter(rgx)
        first_ind = np.arange(min_ind, max_ind+1)
        second_ind = np.delete(np.arange(cat_count), first_ind)

    elif split_type == 'g_topics':
        rgx = re.compile('G[0-9]*')
        min_ind, max_ind = regex_splitter(rgx)
        first_ind = np.arange(min_ind, max_ind+1)
        second_ind = np.delete(np.arange(cat_count), first_ind)

    elif split_type == 'e_topics':
        rgx = re.compile('E[0-9]*')
        min_ind, max_ind = regex_splitter(rgx)
        first_ind = np.arange(min_ind, max_ind+1)
        second_ind = np.delete(np.arange(cat_count), first_ind)

    elif split_type == 'm_topics':
        rgx = re.compile('M[0-9]*')
        min_ind, max_ind = regex_splitter(rgx)
        first_ind = np.arange(min_ind, max_ind+1)
        second_ind = np.delete(np.arange(cat_count), first_ind)

    # save indices of categories
    np.save('./data/first-indices-{}'.format(split_type), first_ind)
    np.save('./data/second-indices-{}'.format(split_type), second_ind)

    first_split, second_split = filter_split(first_ind, second_ind)
    
    save_sparse_csr('./data/first-features-{}'.format(split_type), features[first_split])
    save_sparse_csr('./data/first-labels-{}'.format(split_type), labels[first_split])
    save_sparse_csr('./data/second-features-{}'.format(split_type), features[second_split])
    save_sparse_csr('./data/second-labels-{}'.format(split_type), labels[second_split])

    return (features[first_split], labels[first_split]), (features[second_split], labels[second_split]), (first_ind, second_ind)

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
    parser.add_argument('--coarse', action='store_true')

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
    elif args.coarse:
        ftrs, lbls = get_data('coarse')
    else:
        ftrs, lbls, holdout = [], [], []
