import pandas as pd

SPLIT_OPTIONS = {'train', 'test', 'all'}
N_REALIZATIONS_OPTIONS = {1, 100, 1000}


def load_and_format_covariates(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    return data.values


def load_all_other_vars(file_path, num_treatments):
    data = pd.read_csv(file_path, delimiter=',')
    t, y = data['z'], data['y']

    ys_dict = {}
    for i in range(num_treatments):
        ys_dict['y{}'.format(i)] = data['y_{}'.format(i)]

    return t.values.reshape(-1, 1), y.values, ys_dict


def load_data(file_path, num_treatments, return_ate=True, return_ites=True):
    d = {}
    d['w'] = load_and_format_covariates(file_path)
    t, y, ys_dict = load_all_other_vars(file_path, num_treatments)

    d['t'] = t
    d['y'] = y

    for i in range(num_treatments):
        d['y_{}'.format(i)] = ys_dict['y{}'.format(i)]

    if return_ites:
        ites = []
        for i in range(num_treatments):
            ites.append((d['y_{}'.format(i)]-d['y_0']))
        d['ites'] = ites

    if return_ate:
        ate = []
        for i in range(num_treatments):
            ate.append((d['y_{}'.format(i)].mean()-d['y_0'].mean()))
        d['ate'] = ate

    return d
