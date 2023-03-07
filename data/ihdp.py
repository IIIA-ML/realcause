"""
File for loading the IHDP semi-synthetic dataset.

Hill (2011) took the real covariates from the IHDP data and
generated semi-synthetic data by generating the outcomes via random functions
("response surfaces"). Response surface A corresponds to a linear function; we
do not provide that data in this file. Response surface B corresponds to a
nonlinear function; this is what we provide in this file. We get it from Shalit
et al. (2017) who get it from the NPCI (Dorie, 2016) R package.

References:

    Dorie, V. (2016). NPCI: Non-parametrics for Causal Inference.
        https://github.com/vdorie/npci

    Hill, J. (2011). Bayesian Nonparametric Modeling for Causal Inference.
        Journal of Computational and Graphical Statistics, 20:1, 217-240.

    Shalit, U., Johansson, F.D. & Sontag, D. (2017). Estimating individual
        treatment effect: generalization bounds and algorithms. Proceedings of
        the 34th International Conference on Machine Learning.
"""

import numpy as np
import pandas as pd
import os
from utils import download_dataset, unzip

IHDP_100_TRAIN_URL = 'http://www.fredjo.com/files/ihdp_npci_1-100.train.npz'
IHDP_100_TEST_URL = 'http://www.fredjo.com/files/ihdp_npci_1-100.test.npz'
IHDP_1000_TRAIN_URL = 'http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip'
IHDP_1000_TEST_URL = 'http://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip'

SPLIT_OPTIONS = {'train', 'test', 'all'}
N_REALIZATIONS_OPTIONS = {1, 100, 1000}


def load_and_format_covariates(file_path):

    data = pd.read_csv(file_path, delimiter=',')

    binfeats = ["bw","b.head","preterm","birth.o","nnhealth","momage"]
    contfeats = ["sex","twin","b.marr","mom.lths","mom.hs",	"mom.scoll","cig","first","booze","drugs","work.dur","prenatal","ark","ein","har","mia","pen","tex","was",'momwhite','momblack','momhisp']

    perm = binfeats + contfeats
    x = data[perm]
    return x.values

def load_all_other_vars(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    t, y, y0, y1, y2 = data['z'], data['y'], data['y_0'], data['y_1'],  data['y_2']
    mu_0, mu_1, mu_2 =  data['mu_0'], data['mu_1'], data['mu_2']
    #return t.values.reshape(-1, 1), y, y0, y1, y2, mu_0, mu_1, mu_2
    return t.values.reshape(-1, 1), y.values, y0.values, y1.values, y2.values, mu_0.values, mu_1.values, mu_2.values


def load_ihdp_tri(return_ate=True, return_ites=True):
    d={}
    d['w'] = load_and_format_covariates('/home/bvelasco/realcause/datasets/ihdp_tri.csv')
    t, y, y0, y1, y2, mu_0, mu_1, mu_2 = load_all_other_vars('/home/bvelasco/realcause/datasets/ihdp_tri.csv')

    d['t'] = t
    d['y'] = y

    d['y_0'] = y0
    d['y_1'] = y1
    d['y_2'] = y2

    d['mu_0'] = mu_0
    d['mu_1'] = mu_1
    d['mu_2'] = mu_2

    if return_ites:
        ites = [y1-y0, y2-y1]
        d['ites'] = ites

    if return_ate:
        ate = [y1.mean()-y0.mean(), y2.mean()-y1.mean()]
        d['ate'] = ate

    return d


'''def load_ihdp_tri(split='all', i=0, observe_counterfactuals=False, return_ites=False,
              return_ate=False, dataroot=None):
    """
        Load a single instance of the IHDP dataset

        :param split: 'train', 'test', or 'both' (the default IHDP split is 90/10)
        :param i: dataset instance (0 <= i < 1000)
        :param observe_counterfactuals: if True, return double-sized dataset with
            both y0 (first half) and y1 (second half) observed
        :param return_ites: if True, return ITEs
        :param return_ate: if True, return ATE
        :return: dictionary of results
        """
    if 0 <= i < 100:
        n_realizations = 100
    elif 100 <= i < 1000:
        n_realizations = 1000
        i = i - 100
    else:
        raise ValueError('Invalid i: {} ... Valid i: 0 <= i < 1000'.format(i))

    Tri = True
    if Tri:
        n_realizations = 1

    if split == 'all':
        train, test = load_ihdp_tri_datasets(split=split, n_realizations=n_realizations,
                                         dataroot=dataroot)
    else:
        data = load_ihdp_tri_datasets(split=split, n_realizations=n_realizations,
                                  dataroot=dataroot)

    print('Here')
    print(train)
    ws = []
    ts = []
    y0s = []
    y1s = []
    y2s = []
    itess10 = []
    itess20 = []
    #datasets = [train.f, test.f] if split == 'all' else [data.f]
    datasets = [train, test] if split == 'all' else [data]
    for dataset in datasets:
        w = dataset.x[:, :, i]
        t = dataset.t[:, i]
        y0 = dataset.y0[:, i]
        y1 = dataset.y1[:, i]
        y2 = dataset.y2[:, i]
        ites10 = dataset.mu1[:, i] - dataset.mu0[:, i]
        ites20 = dataset.mu2[:, i] - dataset.mu0[:, i]

        ws.append(w)
        ts.append(t)
        y0s.append(y0)
        y1s.append(y1)
        y2s.append(y2)
        itess10.append(ites10)
        itess20.append(ites20)

    w = np.vstack(ws)
    t = np.concatenate(ts)
    y0 = np.concatenate(y0)
    y1 = np.concatenate(y1)
    y2 = np.concatenate(y2)
    ites10 = np.concatenate(itess10)
    ites20 = np.concatenate(itess20)
    ate10 = np.mean(ites10)
    ate20 = np.mean(ites20)

    d = {}
    if observe_counterfactuals:
        d['w'] = np.vstack([w, w.copy()])
        d['t'] = np.concatenate([t, np.logical_not(t.copy()).astype(int)])
        d['ys'] = np.concatenate([y0, y1, y2])
        ites = np.concatenate([ites10, ites20])  # comment if you don't want duplicates
    else:
        d['w'] = w
        d['t'] = t
        d['y0'] = y0
        d['y1'] = y1
        d['y2'] = y2


    if return_ites:
        d['ite10'] = ites10
        d['ites20'] = ites20

    if return_ate:
        d['ate10'] = ate10
        d['ate20'] = ate20


    print('Im here')
    print(d)
    return d

def load_ihdp_tri_datasets(split='train', n_realizations=1, dataroot=None):

    if split.lower() not in SPLIT_OPTIONS:
        raise ValueError('Invalid "split" option {} ... valid options: {}'
                         .format(split, SPLIT_OPTIONS))
    if isinstance(n_realizations, str):
        n_realizations = int(n_realizations)
    if n_realizations not in N_REALIZATIONS_OPTIONS:
        raise ValueError('Invalid "n_realizations" option {} ... valid options: {}'
                         .format(n_realizations, N_REALIZATIONS_OPTIONS))
    #if n_realizations == 100:
    if split == 'train' or split == 'all':
        data = np.genfromtxt(os.path.join(dataroot,'ihdp_tri.csv'), delimiter=',')
        train = data
    if split == 'test' or split == 'all':
        data = np.genfromtxt(os.path.join(dataroot, 'ihdp_tri.csv'), delimiter=',')
        test = data
    #elif n_realizations == 1000:
    if split == 'train' or split == 'all':
        data = np.genfromtxt(os.path.join(dataroot, 'ihdp_tri.csv'), delimiter=',')
        train = data
    if split == 'test' or split == 'all':
        data = np.genfromtxt(os.path.join(dataroot, 'ihdp_tri.csv'), delimiter=',')
        test = data

    if split == 'train':
        return train
    elif split == 'test':
        return test
    elif split == 'all':
        return train, test
'''


def load_ihdp(split='all', i=0, observe_counterfactuals=False, return_ites=False,
              return_ate=False, dataroot=None):
    """
    Load a single instance of the IHDP dataset

    :param split: 'train', 'test', or 'both' (the default IHDP split is 90/10)
    :param i: dataset instance (0 <= i < 1000)
    :param observe_counterfactuals: if True, return double-sized dataset with
        both y0 (first half) and y1 (second half) observed
    :param return_ites: if True, return ITEs
    :param return_ate: if True, return ATE
    :return: dictionary of results
    """
    if 0 <= i < 100:
        n_realizations = 100
    elif 100 <= i < 1000:
        n_realizations = 1000
        i = i - 100
    else:
        raise ValueError('Invalid i: {} ... Valid i: 0 <= i < 1000'.format(i))

    if split == 'all':
        train, test = load_ihdp_datasets(split=split, n_realizations=n_realizations,
                                         dataroot=dataroot)
    else:
        data = load_ihdp_datasets(split=split, n_realizations=n_realizations,
                                  dataroot=dataroot)

    ws = []
    ts = []
    ys = []
    ys_cf = []
    itess = []
    datasets = [train.f, test.f] if split == 'all' else [data.f]
    for dataset in datasets:
        w = dataset.x[:, :, i]
        t = dataset.t[:, i]
        y = dataset.yf[:, i]
        y_cf = dataset.ycf[:, i]
        ites = dataset.mu1[:, i] - dataset.mu0[:, i]

        ws.append(w)
        ts.append(t)
        ys.append(y)
        ys_cf.append(y_cf)
        itess.append(ites)

    w = np.vstack(ws)
    t = np.concatenate(ts)
    y = np.concatenate(ys)
    y_cf = np.concatenate(ys_cf)
    ites = np.concatenate(itess)
    ate = np.mean(ites)

    d = {}
    if observe_counterfactuals:
        d['w'] = np.vstack([w, w.copy()])
        d['t'] = np.concatenate([t, np.logical_not(t.copy()).astype(int)])
        d['y'] = np.concatenate([y, y_cf])
        ites = np.concatenate([ites, ites.copy()])  # comment if you don't want duplicates
    else:
        d['w'] = w
        d['t'] = t
        d['y'] = y

    if return_ites:
        d['ites'] = ites
    if return_ate:
        d['ate'] = ate

    return d


def load_ihdp_datasets(split='train', n_realizations=100, dataroot=None):
    """
    Load the IHDP data with the nonlinear response surface ("B") that was used
    by Shalit et al. (2017). Description of variables:
        x: covariates (25: 6 continuous and 19 binary)
        t: treatment (binary)
        yf: "factual" (observed) outcome
        ycf: "counterfactual" outcome (random)
        mu0: noiseless potential outcome under control
        mu1: noiseless potential outcome under treatment
        ate: I guess just what was reported in the Hill (2011) paper...
            Not actually accurate. The actual SATEs for the data are the
            following (using (mu1 - mu0).mean()):
                train100:   4.54328871735309
                test100:    4.269906127209613
                all100:     4.406597422281352

                train1000:  4.402550421661204
                test1000:   4.374712690625632
                all1000:    4.388631556143418
        yadd: ???
        ymul: ???

    :param split: 'train', 'test', or 'both'
    :param n_realizations: 100 or 1000 (the two options that the data source provides)
    :return: NpzFile with all the data ndarrays in the 'f' attribute
    """
    if split.lower() not in SPLIT_OPTIONS:
        raise ValueError('Invalid "split" option {} ... valid options: {}'
                         .format(split, SPLIT_OPTIONS))
    if isinstance(n_realizations, str):
        n_realizations = int(n_realizations)
    if n_realizations not in N_REALIZATIONS_OPTIONS:
        raise ValueError('Invalid "n_realizations" option {} ... valid options: {}'
                         .format(n_realizations, N_REALIZATIONS_OPTIONS))
    if n_realizations == 100:
        if split == 'train' or split == 'all':
            path = download_dataset(IHDP_100_TRAIN_URL, 'IHDP train 100', dataroot=dataroot)
            train = np.load(path)
        if split == 'test' or split == 'all':
            path = download_dataset(IHDP_100_TEST_URL, 'IHDP test 100', dataroot=dataroot)
            test = np.load(path)
    elif n_realizations == 1000:
        if split == 'train' or split == 'all':
            path = download_dataset(IHDP_1000_TRAIN_URL, 'IHDP train 1000', dataroot=dataroot)
            unzip_path = unzip(path)
            train = np.load(unzip_path)
        if split == 'test' or split == 'all':
            path = download_dataset(IHDP_1000_TEST_URL, 'IHDP test 1000', dataroot=dataroot)
            unzip_path = unzip(path)
            test = np.load(unzip_path)

    if split == 'train':
        return train
    elif split == 'test':
        return test
    elif split == 'all':
        return train, test
