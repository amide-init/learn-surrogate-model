# surrogate.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# NETWORK
# =========================================================

class _Net(nn.Module):

    def __init__(self, d):

        super().__init__()

        self.body = nn.Sequential(

            nn.Linear(d, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
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
# GAUSSIAN NLL LOSS
# =========================================================

def _nll_loss(mu, log_var, y):

    """
    Gaussian Negative Log Likelihood

    0.5 * (
        log(sigma^2)
        +
        (y - mu)^2 / sigma^2
    )
    """

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
        epochs=100,
        batch_size=32,
        lr=1e-3,
    ):

        self.d = d

        self.n_models = n_models

        self.epochs = epochs

        self.batch_size = batch_size

        self.lr = lr

        self.models = []

        self._stats = None

    # =====================================================
    # FIT
    # =====================================================

    def fit(self, X, y):

        # -------------------------------------------------
        # NORMALIZATION
        # -------------------------------------------------

        xm = X.mean(axis=0)

        xs = np.where(
            X.std(axis=0) < 1e-8,
            1.0,
            X.std(axis=0)
        )

        ym = float(y.mean())

        ys = float(max(y.std(), 1e-8))

        self._stats = (xm, xs, ym, ys)

        Xn = (X - xm) / xs

        yn = (y - ym) / ys

        X_tensor = torch.FloatTensor(Xn)

        y_tensor = torch.FloatTensor(yn)

        self.models = []

        # =================================================
        # TRAIN EACH MODEL
        # =================================================

        for i in range(self.n_models):

            torch.manual_seed(RANDOM_SEED + i)

            np.random.seed(RANDOM_SEED + i)

            # ---------------------------------------------
            # BOOTSTRAP SAMPLING
            # ---------------------------------------------

            idx = np.random.choice(
                len(X_tensor),
                len(X_tensor),
                replace=True
            )

            X_boot = X_tensor[idx]

            y_boot = y_tensor[idx]

            dataset = TensorDataset(X_boot, y_boot)

            loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(dataset)),
                shuffle=True
            )

            # ---------------------------------------------
            # MODEL
            # ---------------------------------------------

            net = _Net(self.d).to(DEVICE)

            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=self.lr,
                weight_decay=1e-4
            )

            best_loss = float("inf")

            patience = 20

            patience_counter = 0

            # ---------------------------------------------
            # TRAINING LOOP
            # ---------------------------------------------

            for epoch in range(self.epochs):

                net.train()

                epoch_loss = 0.0

                for xb, yb in loader:

                    xb = xb.to(DEVICE)

                    yb = yb.to(DEVICE)

                    optimizer.zero_grad()

                    mu, log_var = net(xb)

                    loss = _nll_loss(mu, log_var, yb)

                    loss.backward()

                    optimizer.step()

                    epoch_loss += loss.item()

                epoch_loss /= len(loader)

                # -----------------------------------------
                # EARLY STOPPING
                # -----------------------------------------

                if epoch_loss < best_loss:

                    best_loss = epoch_loss

                    patience_counter = 0

                else:

                    patience_counter += 1

                if patience_counter >= patience:

                    break

            net.eval()

            self.models.append(net)

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, X):

        xm, xs, ym, ys = self._stats

        Xn = (X - xm) / xs

        X_tensor = torch.FloatTensor(Xn).to(DEVICE)

        mus = []

        vars_ = []

        with torch.no_grad():

            for model in self.models:

                mu, log_var = model(X_tensor)

                mu = mu.cpu().numpy()

                var = log_var.exp().cpu().numpy()

                mus.append(mu)

                vars_.append(var)

        mus = np.stack(mus)

        vars_ = np.stack(vars_)

        # =================================================
        # MIXTURE OF GAUSSIANS
        #
        # total variance =
        # mean(predicted variance)
        # +
        # variance of ensemble means
        # =================================================

        ensemble_mu = mus.mean(axis=0)

        ensemble_var = (
            (vars_ + mus ** 2).mean(axis=0)
            -
            ensemble_mu ** 2
        )

        ensemble_var = np.maximum(
            ensemble_var,
            1e-12
        )

        ensemble_std = np.sqrt(ensemble_var)

        # =================================================
        # DENORMALIZE
        # =================================================

        final_mu = ensemble_mu * ys + ym

        final_std = ensemble_std * ys

        return final_mu, final_std