import numpy as np
import torch
import torch.nn as nn

RANDOM_SEED = 42

DEVICE = torch.device("cpu")


# =========================================================
# NETWORK
# =========================================================

class _Net(nn.Module):

    def __init__(self, d):

        super().__init__()

        self.body = nn.Sequential(

            nn.Linear(d, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(32, 1)

        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x):

        h = self.body(x)

        mu = self.mu_head(h).squeeze(-1)

        log_var = (
            self.logvar_head(h)
            .squeeze(-1)
            .clamp(-10, 10)
        )

        return mu, log_var


# =========================================================
# LOSS
# =========================================================

def _nll_loss(mu, log_var, y):

    return (
        0.5 * (
            log_var
            +
            (y - mu) ** 2 / log_var.exp()
        )
    ).mean()


# =========================================================
# DEEP ENSEMBLE
# =========================================================

class DeepEnsemble:

    def __init__(
        self,
        d,
        n_models=5,
        epochs=50,
        lr=1e-3,
    ):

        self.d = d

        self.n_models = n_models

        self.epochs = epochs

        self.lr = lr

        self.models = []

        self.opts = []

        self._stats = None

    # =====================================================
    # INITIALIZE MODELS ONCE
    # =====================================================

    def _initialize_models(self):

        if len(self.models) > 0:
            return

        for i in range(self.n_models):

            torch.manual_seed(
                RANDOM_SEED + i
            )

            net = _Net(self.d).to(DEVICE)

            opt = torch.optim.Adam(
                net.parameters(),
                lr=self.lr
            )

            self.models.append(net)

            self.opts.append(opt)

    # =====================================================
    # FIT
    # =====================================================

    def fit(self, X, y):

        self._initialize_models()

        xm = X.mean(0)

        xs = np.where(
            X.std(0) < 1e-8,
            1.0,
            X.std(0)
        )

        ym = float(y.mean())

        ys = float(max(y.std(), 1e-8))

        self._stats = (xm, xs, ym, ys)

        Xn = torch.FloatTensor(
            (X - xm) / xs
        ).to(DEVICE)

        yn = torch.FloatTensor(
            (y - ym) / ys
        ).to(DEVICE)

        for model, opt in zip(
            self.models,
            self.opts
        ):

            idx = np.random.choice(
                len(Xn),
                len(Xn),
                replace=True
            )

            X_boot = Xn[idx]

            y_boot = yn[idx]

            model.train()

            for _ in range(self.epochs):

                opt.zero_grad()

                mu, log_var = model(X_boot)

                loss = _nll_loss(
                    mu,
                    log_var,
                    y_boot
                )

                loss.backward()

                opt.step()

            model.eval()

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, X):

        xm, xs, ym, ys = self._stats

        Xn = torch.FloatTensor(
            (X - xm) / xs
        ).to(DEVICE)

        mus = []

        vars_ = []

        with torch.no_grad():

            for model in self.models:

                mu, log_var = model(Xn)

                mus.append(
                    mu.cpu().numpy()
                )

                vars_.append(
                    log_var.exp().cpu().numpy()
                )

        mus = np.stack(mus)

        vars_ = np.stack(vars_)

        ensemble_mu = mus.mean(0)

        ensemble_var = (
            (vars_ + mus ** 2).mean(0)
            - ensemble_mu ** 2
        )

        ensemble_var = np.maximum(
            ensemble_var,
            1e-12
        )

        ensemble_std = np.sqrt(
            ensemble_var
        )

        return (
            ensemble_mu * ys + ym,
            ensemble_std * ys
        )