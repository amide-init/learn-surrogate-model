"""
Deep ensemble surrogate for AFN-CMA-ES.

Trained on rank-normalised objective values.
Returns predictive mean and uncertainty for acquisition and hybrid CMA ranking.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RANDOM_SEED = 42
DEVICE = torch.device("cpu")
torch.set_num_threads(1)

_ACTIVATIONS = [F.gelu, F.silu, F.elu, torch.tanh, F.softsign]


class _Net(nn.Module):

    def __init__(self, d: int, activation):
        super().__init__()
        self.act = activation
        h1 = min(128, max(32, 10 * d))
        h2 = min(64,  max(16,  5 * d))
        self.fc1         = nn.Linear(d,  h1)
        self.fc2         = nn.Linear(h1, h2)
        self.mu_head     = nn.Linear(h2, 1)
        self.logvar_head = nn.Linear(h2, 1)

    def forward(self, x):
        h       = self.act(self.fc1(x))
        h       = self.act(self.fc2(h))
        mu      = self.mu_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1).clamp(-4, 0)
        return mu, log_var


def _loss(mu, log_var, y):
    nll = 0.5 * (log_var + (y - mu) ** 2 / log_var.exp()).mean()
    mse = ((y - mu) ** 2).mean()
    return nll + 0.5 * mse


class DeepEnsemble:

    def __init__(
        self,
        d: int,
        n_models: int = 5,
        epochs: int = 500,
        retrain_epochs: int = 120,
        lr: float = 1e-3,
        anchor_lam: float = 1e-3,
    ):
        self.d              = d
        self.n_models       = n_models
        self.epochs         = epochs
        self.retrain_epochs = retrain_epochs
        self.lr             = lr
        self.anchor_lam     = anchor_lam
        self.models         = []
        self.opts           = []
        self._init_params   = []
        self._stats         = None
        self._first_fit     = True

    def _initialize_models(self):
        if self.models:
            return
        for i in range(self.n_models):
            torch.manual_seed(RANDOM_SEED + i)
            net  = _Net(self.d, _ACTIVATIONS[i % len(_ACTIVATIONS)]).to(DEVICE)
            init = [p.detach().clone() for p in net.parameters()]
            self.models.append(net)
            self.opts.append(
                torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-5)
            )
            self._init_params.append(init)

    def fit(self, X, y):
        self._initialize_models()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        epochs          = self.epochs if self._first_fit else self.retrain_epochs
        self._first_fit = False
        xm = X.mean(axis=0)
        xs = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        ym = float(y.mean())
        ys = float(max(y.std(), 1e-8))
        self._stats = (xm, xs, ym, ys)
        Xn = torch.as_tensor((X - xm) / xs, dtype=torch.float32, device=DEVICE)
        yn = torch.as_tensor((y - ym) / ys, dtype=torch.float32, device=DEVICE)
        for model, opt, init_params in zip(self.models, self.opts, self._init_params):
            model.train()
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                mu, log_var = model(Xn)
                loss_val    = _loss(mu, log_var, yn)
                anchor      = sum(
                    ((p - p0.to(DEVICE)) ** 2).sum()
                    for p, p0 in zip(model.parameters(), init_params)
                )
                loss = loss_val + self.anchor_lam * anchor
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()

    def predict(self, X):
        if self._stats is None:
            raise RuntimeError("predict() called before fit()")
        xm, xs, ym, ys = self._stats
        X  = np.asarray(X, dtype=np.float32)
        Xn = torch.as_tensor((X - xm) / xs, dtype=torch.float32, device=DEVICE)
        mus, vars_ = [], []
        with torch.no_grad():
            for model in self.models:
                mu, log_var = model(Xn)
                mus.append(mu.cpu().numpy())
                vars_.append(log_var.exp().cpu().numpy())
        mus          = np.stack(mus)
        vars_        = np.stack(vars_)
        ensemble_mu  = mus.mean(axis=0)
        ensemble_var = (vars_ + mus ** 2).mean(axis=0) - ensemble_mu ** 2
        ensemble_var = np.maximum(ensemble_var, 1e-12)
        return ensemble_mu * ys + ym, np.sqrt(ensemble_var) * ys
