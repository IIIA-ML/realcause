import numpy as np
from models import distributions
from models.base import BaseGenModel
from models import preprocess
from models.preprocess import PlaceHolderTransform
import torch
from torch import nn
from torch.utils import data
from itertools import chain


class MLPParams:
    def __init__(self, n_hidden_layers=1, dim_h=64, activation=nn.ReLU()):
        self.n_hidden_layers = n_hidden_layers
        self.dim_h = dim_h
        self.activation = activation


_DEFAULT_MLP = dict(mlp_params_t_w=MLPParams(), mlp_params_y_tw=MLPParams())


class TrainingParams:
    def __init__(
        self,
        batch_size=32,
        lr=0.001,
        num_epochs=100,
        verbose=True,
        print_every_iters=100,
        eval_every=100,
        plot_every=1000,
        optim=torch.optim.Adam,
        **optim_args
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_every_iters = print_every_iters
        self.optim = optim
        self.eval_every = eval_every
        self.plot_every = plot_every
        self.optim_args = optim_args


class CausalDataset(data.Dataset):
    def __init__(
        self,
        w,
        t,
        y,
        wtype="float32",
        ttype="float32",
        ytype="float32",
        w_transform: preprocess.Preprocess = preprocess.PlaceHolderTransform(),
        t_transform: preprocess.Preprocess = preprocess.PlaceHolderTransform(),
        y_transform: preprocess.Preprocess = preprocess.PlaceHolderTransform(),
    ):
        self.w = w.astype(wtype)
        self.t = t.astype(ttype)
        self.y = y.astype(ytype)

        # todo: no need anymore, remove?
        self.w_transform = w_transform
        self.t_transform = t_transform
        self.y_transform = y_transform

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, index):
        return (
            self.w_transform.transform(self.w[index]),
            self.t_transform.transform(self.t[index]),
            self.y_transform.transform(self.y[index]),
        )


# TODO: for more complex w, we might need to share parameters (dependent on the problem)
class MLP(BaseGenModel):
    def __init__(
        self,
        w,
        t,
        y,
        seed=1,
        network_params=None,
        training_params=TrainingParams(),
        binary_treatment=False,
        outcome_distribution: distributions.BaseDistribution = distributions.FactorialGaussian(),
        outcome_min=None,
        outcome_max=None,
        train_prop=1,
        val_prop=0,
        test_prop=0,
        shuffle=True,
        early_stop=True,
        ignore_w=False,
        grad_norm=float("inf"),
        w_transform=PlaceHolderTransform,
        t_transform=PlaceHolderTransform,
        y_transform=PlaceHolderTransform,
        savepath=".cache_best_model.pt",
        test_size=None,
    ):
        super(MLP, self).__init__(
            *self._matricize((w, t, y)),
            seed=seed,
            train_prop=train_prop,
            val_prop=val_prop,
            test_prop=test_prop,
            shuffle=shuffle,
            w_transform=w_transform,
            t_transform=t_transform,
            y_transform=y_transform,
            test_size=test_size
        )

        self.binary_treatment = binary_treatment
        if binary_treatment:  # todo: input?
            self.treatment_distribution = distributions.Bernoulli()
        else:
            self.treatment_distribution = distributions.FactorialGaussian()
        self.outcome_distribution = outcome_distribution
        self.outcome_min = outcome_min
        self.outcome_max = outcome_max
        self.early_stop = early_stop
        self.ignore_w = ignore_w
        self.grad_norm = grad_norm
        self.savepath = savepath

        self.dim_w = self.w_transformed.shape[1]
        self.dim_t = self.t_transformed.shape[1]
        self.dim_y = self.y_transformed.shape[1]

        if network_params is None:
            network_params = _DEFAULT_MLP
        self.network_params = network_params
        self.build_networks()

        # TODO: multiple optimizers ?
        self.training_params = training_params
        self.optim = training_params.optim(
            chain(*[net.parameters() for net in self.networks]),
            training_params.lr,
            **training_params.optim_args
        )

        # TODO: binary treatment -> long data type
        self.data_loader = data.DataLoader(
            CausalDataset(self.w_transformed, self.t_transformed, self.y_transformed),
            batch_size=training_params.batch_size,
            shuffle=True,
        )

        if len(self.val_idxs) > 0:
            self.data_loader_val = data.DataLoader(
                CausalDataset(
                    self.w_val_transformed,
                    self.t_val_transformed,
                    self.y_val_transformed,
                ),
                batch_size=training_params.batch_size,
                shuffle=True,
            )

        self.best_val_loss = float("inf")

    def _matricize(self, data):
        return [np.reshape(d, [d.shape[0], -1]) for d in data]

    def _build_mlp(self, dim_x, dim_y, MLP_params=MLPParams(), output_multiplier=2):
        dim_h = MLP_params.dim_h
        hidden_layers = [nn.Linear(dim_x, dim_h), MLP_params.activation]
        for _ in range(MLP_params.n_hidden_layers - 1):
            hidden_layers += [nn.Linear(dim_h, dim_h), MLP_params.activation]
        hidden_layers += [nn.Linear(dim_h, dim_y * output_multiplier)]
        return nn.Sequential(*hidden_layers)

    def build_networks(self):
        self.MLP_params_t_w = self.network_params["mlp_params_t_w"]
        self.MLP_params_y_tw = self.network_params["mlp_params_y_tw"]
        output_multiplier_t = 1 if self.binary_treatment else 2
        self.mlp_t_w = self._build_mlp(
            self.dim_w, self.dim_t, self.MLP_params_t_w, output_multiplier_t
        )
        self.mlp_y_tw = self._build_mlp(
            self.dim_w + self.dim_t,
            self.dim_y,
            self.MLP_params_y_tw,
            self.outcome_distribution.num_params,
        )
        self.networks = [self.mlp_t_w, self.mlp_y_tw]

    def _get_loss(self, w, t, y):
        t_ = self.mlp_t_w(w)
        if self.ignore_w:
            w = torch.zeros_like(w)
        y_ = self.mlp_y_tw(torch.cat([w, t], dim=1))
        loss_t = self.treatment_distribution.loss(t, t_)
        loss_y = self.outcome_distribution.loss(y, y_)
        loss = loss_t + loss_y
        return loss, loss_t, loss_y

    def train(self, early_stop=None, print_=print, comet_exp=None):
        if early_stop is None:
            early_stop = self.early_stop

        c = 0
        self.best_val_loss = float("inf")
        for _ in range(self.training_params.num_epochs):
            for w, t, y in self.data_loader:

                self.optim.zero_grad()
                loss, loss_t, loss_y = self._get_loss(w, t, y)
                # TODO: learning rate can be separately adjusted by weighting the losses here
                loss.backward()

                torch.nn.utils.clip_grad_norm(
                    chain(*[net.parameters() for net in self.networks]), self.grad_norm
                )

                self.optim.step()

                c += 1

                if (
                    self.training_params.verbose
                    and c % self.training_params.print_every_iters == 0
                ):
                    print_("Iteration {}: {} {}".format(c, loss_t, loss_y))

                    if comet_exp is not None:
                        comet_exp.log_metric("loss_t", loss_t.item())
                        comet_exp.log_metric("loss_y", loss_y.item())

                if c % self.training_params.eval_every == 0 and len(self.val_idxs) > 0:
                    loss_val = self.evaluate(self.data_loader_val).item()
                    print_("Iteration {} valid loss {}".format(c, loss_val))
                    if loss_val < self.best_val_loss:
                        self.best_val_loss = loss_val
                        print("saving best-val-loss model")
                        torch.save(
                            [net.state_dict() for net in self.networks], self.savepath
                        )
                        # todo: this is not ideal since we cannot run multiple experiments at the same time
                        #       without overwriting the saved model

        if early_stop and len(self.val_idxs) > 0:
            print("loading best-val-loss model (early stopping checkpoint)")
            for net, params in zip(self.networks, torch.load(self.savepath)):
                net.load_state_dict(params)

    @torch.no_grad()
    def evaluate(self, data_loader):
        loss = 0
        n = 0
        for net in self.networks:
            net.eval()

        for w, t, y in data_loader:
            loss += self._get_loss(w, t, y)[0] * w.size(0)
            n += w.size(0)

        for net in self.networks:
            net.train()
        return loss / n

    def _sample_t(self, w=None, positivity=0):
        # todo: this (positivity) is only for binary treatment though. do we have to deal with different types of treatments?
        t_ = self.mlp_t_w(torch.from_numpy(w).float())
        return self.treatment_distribution.sample(t_ + positivity)

    def _sample_y(self, t, w=None):
        if self.ignore_w:
            w = np.zeros_like(w)
        wt = np.concatenate([w, t], 1)
        y_ = self.mlp_y_tw(torch.from_numpy(wt).float())
        y_samples = self.outcome_distribution.sample(y_)

        if self.outcome_min is not None or self.outcome_max is not None:
            return np.clip(y_samples, self.outcome_min, self.outcome_max)
        else:
            return y_samples

    def mean_y(self, t, w):
        if self.ignore_w:
            w = np.zeros_like(w)
        wt = np.concatenate([w, t], 1)
        return self.outcome_distribution.mean(
            self.mlp_y_tw(torch.from_numpy(wt).float())
        )


if __name__ == "__main__":
    from data.lalonde import load_lalonde
    import matplotlib.pyplot as plt
    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    # data = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)
    #
    # dgm = DataGenModel(data)
    # data_samples = dgm.sample()

    # w, t, y = generate_wty_linear_multi_w_data(500, data_format=NUMPY, wdim=5)
    # mlp = MLP(w, t, y)
    # data_samples = mlp.sample()
    # mlp.plot_ty_dists()
    # uni_metrics = mlp.get_univariate_quant_metrics()
    # multi_ty_metrics = mlp.get_multivariate_quant_metrics(include_w=False, n_permutations=10)
    # multi_wty_metrics = mlp.get_multivariate_quant_metrics(include_w=True, n_permutations=10)

    dataset = 2
    if dataset == 1:
        w, t, y = load_lalonde()
        # dist = distributions.MixedDistribution([0.0], distributions.LogLogistic())
        dist = distributions.FactorialGaussian()
        training_params = TrainingParams(
            lr=0.0005, batch_size=128, num_epochs=100, verbose=False
        )
        mlp_params_y_tw = MLPParams(n_hidden_layers=2, dim_h=256)
        early_stop = True
        ignore_w = False
    elif dataset == 2:
        w, t, y = load_lalonde(rct=True)
        # dist = distributions.MixedDistribution([0.0], distributions.LogLogistic())
        dist = distributions.FactorialGaussian()
        training_params = TrainingParams(lr=0.001, batch_size=64, num_epochs=200)
        mlp_params_y_tw = MLPParams(n_hidden_layers=2, dim_h=1024)
        early_stop = True
        ignore_w = False
    elif dataset == 3:
        w, t, y = load_lalonde(obs_version="cps1")
        dist = distributions.MixedDistribution(
            [0.0, 25564.669921875 / y.max()], distributions.LogNormal()
        )
        training_params = TrainingParams(lr=0.0005, batch_size=128, num_epochs=1000)
        mlp_params_y_tw = MLPParams(
            n_hidden_layers=3, dim_h=512, activation=torch.nn.LeakyReLU()
        )
        early_stop = True
        ignore_w = False
    else:
        raise (Exception("dataset {} not implemented".format(dataset)))

    param = torch.zeros(1, dist.num_params, requires_grad=True)
    y_torch = torch.from_numpy(y / y.max()).float()[:, None]
    for i in range(500):
        param.grad = None
        nll = -dist.likelihood(y_torch, param.expand(len(y), -1)).mean()
        nll.backward()
        param.data.sub_(0.01 * param.grad.data)
        print(i)

    plt.hist(y / y.max(), 50, density=True, alpha=0.5, range=(0, 1))
    n_ = 1000
    ll = dist.likelihood(torch.linspace(0, 1, n_)[:, None], param.expand(n_, -1))
    plt.plot(np.linspace(0, 1, n_), np.exp(ll.data.numpy()), "x", ms=2)

    y_samples = dist.sample(param.expand(n_, -1))
    plt.hist(y_samples, 50, density=True, alpha=0.5, range=(0, 1))
    plt.legend(["data", "density", "samples"], loc=1)

    mlp = MLP(
        w,
        t,
        y,
        training_params=training_params,
        network_params=dict(
            mlp_params_t_w=MLPParams(), mlp_params_y_tw=mlp_params_y_tw
        ),
        binary_treatment=True,
        outcome_distribution=dist,
        outcome_min=0.0,
        outcome_max=1.0,
        train_prop=0.5,
        val_prop=0.1,
        test_prop=0.4,
        seed=1,
        early_stop=early_stop,
        ignore_w=ignore_w,
        w_transform=preprocess.Standardize,
        y_transform=preprocess.Normalize,
    )
    mlp.train()
    data_samples = mlp.sample()
    # mlp.plot_ty_dists()
    uni_metrics = mlp.get_univariate_quant_metrics(dataset="test")
    pp.pprint(uni_metrics)
    print("noisy ate:", mlp.noisy_ate())
    # multi_ty_metrics = mlp.get_multivariate_quant_metrics(include_w=False, n_permutations=10)
    # multi_wty_metrics = mlp.get_multivariate_quant_metrics(include_w=True, n_permutations=10)
