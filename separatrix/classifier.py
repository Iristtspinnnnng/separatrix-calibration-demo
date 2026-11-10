"""classifier.py
========================================================================
Stage 2: train a NEURAL-NET classifier on (X, basin_label) and expose
the MC-DROPOUT ENTROPY as the uncertainty metric.

Directly mirrors Algorithm 1 of arXiv:2608.14743:
    "neural network f(D1; theta) with dropout layers"
    "uncertainty metric ... based on Monte Carlo dropout [15],
     which is represented by an entropy H."

Two interchangeable backends, same semantics:

  * backend="torch"   (default when torch is installed)
        a small nn.Module on the GPU (CUDA) if one is visible.
        dropout is applied to the LAST HIDDEN layer, exactly like the
        sklearn version below, so the entropy numbers are comparable.
        60k points x 150 epochs ~= seconds on a laptop RTX 4060.

  * backend="sklearn" (fallback, pure CPU)
        an MLPClassifier whose last hidden activations are masked by a
        fresh Bernoulli mask at every stochastic forward pass
        (Gal & Ghahramani 2016 MC dropout, done by hand because
        sklearn's MLPClassifier has no dropout kwarg).

Both expose:
        .fit(X, y, warm_start=..., max_iter=...)   -> continuation training
        .predict(X)                                -> hard class (deterministic)
        .predict_proba(X, n_mc=30)                 -> MC-averaged probabilities
        .mc_entropy(X, M=12)                       -> MC predictive entropy H
        .mc_predict(X, M=12)                       -> (mean_proba, H) in ONE pass
                                                      (cost fix: one set of
                                                       forward passes feeds both
                                                       the entropy and the
                                                       confidence diagnostic)

The entropy H is the predictive entropy of the stochastic forward passes:
        H(x) = - sum_c  p_c(x) * log p_c(x),
        p_c(x) = (1/M) sum_{m=1}^{M} softmax(f_m(x))_c
========================================================================
"""
import numpy as np

from sklearn.preprocessing import StandardScaler

__all__ = ["MCClassifier", "predictive_entropy", "backend_report"]

# --- optional torch ------------------------------------------------------
_HAVE_TORCH = False
torch = None
try:                                        # pragma: no cover
    import torch as _torch
    torch = _torch
    _HAVE_TORCH = True
except Exception:                           # pragma: no cover
    pass

try:
    from sklearn.neural_network import MLPClassifier
    _HAVE_SKLEARN = True
except Exception:                           # pragma: no cover
    _HAVE_SKLEARN = False


def predictive_entropy(probs):
    """
    probs : ndarray, shape (n_samples, n_classes)
            each row is a probability vector (must sum to 1).
    Returns H in nats, shape (n_samples,).
    """
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


def backend_report():
    """Human-readable description of what will actually be used."""
    if not _HAVE_TORCH:
        return ("sklearn (CPU)", "torch is not installed -> CPU sklearn MLP")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return ("torch (GPU)",
                f"torch {torch.__version__} / CUDA {torch.version.cuda} / {name}")
    return ("torch (CPU)",
            f"torch {torch.__version__}, no CUDA device visible")


# =========================================================================
# sklearn backend
# =========================================================================
class _MCForwardMLP:
    """
    Wrap a FITTED sklearn MLPClassifier and expose `predict_proba`
    with a RANDOM dropout mask on the last hidden layer, so we can
    run multiple stochastic forward passes (MC dropout).
    """
    def __init__(self, clf, drop_rate=0.5, rng=None):
        self.clf = clf
        self.drop_rate = drop_rate
        self.rng = rng or np.random.default_rng()

        coefs = clf.coefs_
        intercepts = clf.intercepts_
        self.W = [np.asarray(w, dtype=float) for w in coefs]
        self.b = [np.asarray(b, dtype=float).reshape(-1) for b in intercepts]
        self.n_layers = len(self.W)           # input->h1->...->output
        self.L_last = self.n_layers - 2       # index of last HIDDEN layer

    def _act(self, z, kind):
        if kind == "relu":
            return np.maximum(0.0, z)
        if kind == "identity":
            return z
        raise ValueError(kind)

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        """One stochastic forward pass (dropout applied to last hidden)."""
        a = np.asarray(X, dtype=float)
        for ell in range(self.n_layers - 1):
            z = a @ self.W[ell] + self.b[ell]
            a = self._act(z, self.clf.activation)
            if ell == self.L_last and self.drop_rate > 0.0:
                mask = (self.rng.uniform(size=a.shape) >=
                        self.drop_rate).astype(float)
                mask *= (1.0 / (1.0 - self.drop_rate))   # inverted dropout
                a = a * mask
        z_out = a @ self.W[-1] + self.b[-1]
        return self._softmax(z_out.reshape(1, -1) if a.ndim == 1 else z_out)

    def predict_proba(self, X):
        return self.forward(np.asarray(X, dtype=float))


# =========================================================================
# torch backend
# =========================================================================
if _HAVE_TORCH:                             # pragma: no branch
    class _TorchMLP(torch.nn.Module):
        """Linear->ReLU stack, dropout on the LAST hidden layer only
        (mirrors `_MCForwardMLP` so both backends compute the same H)."""

        def __init__(self, hidden, n_in, n_out, dropout=0.5):
            super().__init__()
            mods, prev = [], n_in
            for h in hidden:
                mods += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
                prev = h
            self.body = torch.nn.Sequential(*mods)
            self.drop = torch.nn.Dropout(dropout)     # inverted dropout
            self.head = torch.nn.Linear(prev, n_out)

        def forward(self, x):
            return self.head(self.drop(self.body(x)))  # returns LOGITS


# =========================================================================
# public wrapper
# =========================================================================
class MCClassifier:
    """
    A wrapper that provides:
        .fit(X, y, warm_start=False, max_iter=None)
        .predict(X)                  -> hard class (deterministic)
        .predict_proba(X, n_mc=30)   -> mean predictive probabilities
        .mc_entropy(X, M=12)         -> MC-dropout predictive entropy H(x)
        .mc_predict(X, M=12)         -> (mean_proba, H) with ONE MC loop
    """

    def __init__(self, hidden=(256, 256, 128), dropout=0.5,
                 max_iter=400, random_state=0,
                 backend="auto", device=None,
                 batch_size=4096, lr=1e-3, n_iter_no_change=15,
                 validation_fraction=0.1, tol=1e-4, verbose=False):
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.max_iter = max_iter
        self.random_state = random_state
        self.backend = backend               # "auto" | "torch" | "sklearn"
        self.device = device                 # None -> cuda if available
        self.batch_size = batch_size
        self.lr = lr
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.clf = None                      # sklearn model
        self.net_ = None                     # torch model
        self.classes_ = None
        self.backend_ = None
        self.device_ = None
        self.n_epochs_ = 0
        self.loss_curve_ = []

    # -- backend plumbing -------------------------------------------------
    def _resolve_backend(self):
        b = self.backend
        if b == "auto":
            return "torch" if _HAVE_TORCH else "sklearn"
        if b == "torch" and not _HAVE_TORCH:
            raise RuntimeError("backend='torch' requested but torch is not importable")
        if b == "sklearn" and not _HAVE_SKLEARN:
            raise RuntimeError("backend='sklearn' requested but scikit-learn is missing")
        return b

    def describe(self):
        if self.backend_ == "torch":
            return f"torch on {self.device_}"
        return "sklearn MLP (CPU)"

    # -- training ---------------------------------------------------------
    def fit(self, X, y, warm_start=False, max_iter=None):
        """
        warm_start=True reuses the weights of the previous fit and only
        runs `max_iter` extra epochs (sklearn: MLPClassifier(warm_start=True);
        torch: keep the nn.Module and its scaler).

        The scaler is fitted ONCE (first fit) and reused, so continuation
        training is not disturbed by a rescaling of the inputs.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        self.backend_ = self._resolve_backend()
        epochs = int(max_iter if max_iter is not None else self.max_iter)

        if self.backend_ == "torch":
            warm = (warm_start and self.net_ is not None
                    and np.array_equal(classes, self.classes_))
            self._fit_torch(X, y, classes, epochs, warm)
        else:
            warm = (warm_start and self.clf is not None
                    and np.array_equal(classes, self.classes_))
            self._fit_sklearn(X, y, classes, epochs, warm)

        self.classes_ = classes
        return self

    def _fit_sklearn(self, X, y, classes, epochs, warm):
        if not warm:
            self.scaler = StandardScaler().fit(X)
            self.clf = MLPClassifier(
                hidden_layer_sizes=self.hidden,
                activation="relu",
                solver="adam",
                max_iter=epochs,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
            )
            self.clf.fit(self.scaler.transform(X), y)
        else:
            # sklearn keeps the early-stopping bookkeeping (_best_loss_,
            # _no_improvement_count) across warm_start fits; on a changed
            # dataset that makes it stop after a single epoch.  Reset it so
            # a continuation fit actually trains `epochs` epochs.
            for attr, val in (("_best_loss_", np.inf),
                              ("_no_improvement_count", 0)):
                if hasattr(self.clf, attr):
                    setattr(self.clf, attr, val)
            self.clf.set_params(warm_start=True, max_iter=epochs,
                                random_state=self.random_state)
            self.clf.fit(self.scaler.transform(X), y)
        self.n_epochs_ = int(getattr(self.clf, "n_iter_", 0))
        self.loss_curve_ = list(getattr(self.clf, "loss_curve_", []))

    def _fit_torch(self, X, y, classes, epochs, warm):
        dev = torch.device(self.device) if self.device else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = dev

        if not warm:
            torch.manual_seed(self.random_state)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(self.random_state)
            self.scaler = StandardScaler().fit(X)
            self.net_ = _TorchMLP(self.hidden, n_in=X.shape[1],
                                  n_out=len(classes),
                                  dropout=self.dropout).to(dev)

        Xs = self.scaler.transform(X).astype(np.float32)
        y_idx = np.searchsorted(classes, y).astype(np.int64)

        n = len(Xs)
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(n)
        n_val = int(self.validation_fraction * n) if n >= 40 else 0
        val_idx, tr_idx = order[:n_val], order[n_val:]

        Xt = torch.from_numpy(np.ascontiguousarray(Xs[tr_idx])).to(dev)
        yt = torch.from_numpy(np.ascontiguousarray(y_idx[tr_idx])).to(dev)
        if n_val:
            Xv = torch.from_numpy(np.ascontiguousarray(Xs[val_idx])).to(dev)
            yv = torch.from_numpy(np.ascontiguousarray(y_idx[val_idx])).to(dev)

        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        lossf = torch.nn.CrossEntropyLoss()
        bs = max(1, min(self.batch_size, len(Xt)))
        best = float("inf")
        bad = 0
        used = 0
        self.loss_curve_ = []

        self.net_.train()
        for ep in range(epochs):
            perm = torch.randperm(len(Xt), device=dev)
            tot = 0.0
            for s in range(0, len(Xt), bs):
                b = perm[s:s + bs]
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.net_(Xt[b]), yt[b])
                loss.backward()
                opt.step()
                tot += loss.detach().item() * len(b)
            train_loss = tot / len(Xt)
            self.loss_curve_.append(train_loss)
            used = ep + 1

            if n_val:
                self.net_.eval()
                with torch.no_grad():
                    val_loss = float(lossf(self.net_(Xv), yv))
                self.net_.train()
                if val_loss < best - self.tol:
                    best, bad = val_loss, 0
                else:
                    bad += 1
                    if bad >= self.n_iter_no_change:
                        break
            if self.verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"      [torch:{dev.type}] epoch {ep+1:4d} "
                      f"train_loss={train_loss:.4f}"
                      + (f" val_loss={val_loss:.4f}" if n_val else ""),
                      flush=True)
        self.n_epochs_ = used
        self.net_.eval()

    # -- inference --------------------------------------------------------
    def _torch_passes(self, Xs, n_mc):
        """(n_mc, n, C) array of softmax outputs with dropout ACTIVE."""
        dev = self.device_
        Xs = np.ascontiguousarray(Xs, dtype=np.float32)
        Xt = torch.from_numpy(Xs).to(dev)
        self.net_.train()                     # dropout ON (MC sampling)
        out = np.empty((n_mc, len(Xs), len(self.classes_)), dtype=np.float32)
        with torch.no_grad():
            for m in range(n_mc):
                out[m] = torch.softmax(self.net_(Xt), dim=1).cpu().numpy()
        self.net_.eval()
        return out

    def _mc_proba(self, Xs, n_mc):
        """(n_mc, n_samples, n_classes) stochastic forward passes."""
        if self.backend_ == "torch":
            return self._torch_passes(Xs, n_mc)
        rng = np.random.default_rng(self.random_state)
        wrap = _MCForwardMLP(self.clf, drop_rate=self.dropout, rng=rng)
        return np.asarray([wrap.forward(Xs) for _ in range(n_mc)])

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.backend_ == "torch":
            dev = self.device_
            Xs = self.scaler.transform(X).astype(np.float32)
            self.net_.eval()
            with torch.no_grad():
                logits = self.net_(torch.from_numpy(
                    np.ascontiguousarray(Xs)).to(dev))
            idx = logits.argmax(dim=1).cpu().numpy()
            return self.classes_[idx]
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X, n_mc=30):
        """Mean predictive probabilities over `n_mc` stochastic passes."""
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        return self._mc_proba(Xs, n_mc).mean(axis=0)

    def mc_entropy(self, X, M=12):
        """MC-dropout predictive entropy (the paper's uncertainty metric)."""
        _, H = self.mc_predict(X, M=M)
        return H

    def mc_predict(self, X, M=12):
        """
        ONE set of M stochastic forward passes -> (mean_proba, H).

        This is the cost fix: the entropy field and the confidence
        diagnostic used to be computed by two separate MC loops (12 and 30
        passes, the first 12 masks of which were literally identical).
        """
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        probas = self._mc_proba(Xs, M).mean(axis=0)     # (N, C)
        return probas, predictive_entropy(probas)
