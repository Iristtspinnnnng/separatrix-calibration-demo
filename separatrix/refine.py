"""The entropy-driven iterative-refinement loop (stage 3 of the pipeline).
Stage 3 of the paper's pipeline, plus the diagnostic that motivated this
repository.

It implements the loop of arXiv:2608.14743 (Algorithm 2): train a classifier,
read its MC-dropout entropy, keep the high-entropy points, resample around
them, re-label with the real dynamics, repeat.  Then it measures what the
loop does to the classifier and to the entropy that drives it.

The hypothesis under test
-------------------------
    "Oversampling the separatrix makes the training distribution unbalanced,
    and with hard basin labels the classifier's confidence degrades and the
    entropy signal gets corrupted."

Four numbers would have to move for that story to hold:

    * class_balance       -- how imbalanced the set becomes;
    * mean_max_prob       -- "confidence", rising while accuracy on the true
                             separatrix does not rise;
    * boundary_accuracy   -- accuracy on the (wide, dynamical) proxy band;
    * entropy_separability-- AUROC of entropy separating band from interior,
                             which drops if the signal is corrupted.

Two of those are measured on the *current training set*, whose composition
changes every round -- a measurement trap isolated in
``experiments/metric_bias.py``: the same models, scored on a fixed
independent population, move in the opposite direction.

PERFORMANCE NOTES (why this loop used to take hours)
----------------------------------------------------
The first version of this loop was not buggy -- it did ~9x more work than it
had to, re-training from scratch every round while the dataset grew by 30% per
round (15% selected x 2 jittered copies appended, forever):

    n = 90k -> 117k -> 152k -> 198k -> 257k      (x1.30 per round)
    total training volume = 9.05x one fit
    one fit at n=90k     = ~100 epochs, 6..30 min on CPU

Three structural fixes are applied here:

  (1) WARM START.  Round r>0 continues the previous network for
      `extra_epochs` epochs instead of re-training from scratch.

  (2) BOUNDED TRAINING BUDGET.  The training set is assembled every round
      from a uniform sample of the original dataset (the "base pool") and a
      sample of the accumulated boundary pool, with a FIXED total size
      `max_samples` and a boundary share that still grows round by round
      (`boundary_fraction_*`).  The mechanism the demo is about -- the
      boundary crowding out uniform coverage, i.e. the imbalance -- is
      preserved exactly, but the cost stays constant instead of growing.

  (3) ONE MC LOOP INSTEAD OF THREE.  `clf.mc_predict(X, M)` returns
      (mean_proba, H) from a single set of M stochastic forward passes; the
      confidence diagnostic reuses that mean_proba, and the hard predictions
      come from a single `.predict` call.

Also: the oracle boundary mask is now computed once per point (cached in the
pools) instead of being recomputed for the whole training set every round.

------------------------------------------------------------------------
PERFORMANCE NOTES (why this loop used to take hours)
------------------------------------------------------------------------
The first version of this loop was not buggy -- it was just doing ~9x
more work than it had to, and it re-trained from scratch every round
while the dataset grew by 30% per round (15% selected x 2 jittered
copies appended, forever):

    n = 90k -> 117k -> 152k -> 198k -> 257k      (x1.30 per round)
    total training volume = 9.05x one fit
    one fit at n=90k     = ~100 epochs, 6..30 min on CPU

Two structural fixes are applied here:

  (2) WARM START.  Round r>0 continues the previous network for
      `extra_epochs` epochs instead of re-training from scratch.

  (3) BOUNDED TRAINING BUDGET.  The training set is assembled every
      round from
          * a uniform sample of the original dataset (the "base pool"),
          * a sample of the accumulated boundary pool,
      with a FIXED total size `max_samples` and a boundary share that
      still grows round by round (`boundary_fraction_*`).  The physics
      that the demo wants to show -- the boundary crowding out uniform
      coverage, i.e. the imbalance -- is preserved exactly, but the
      cost stays constant instead of doubling.

  (4) ONE MC LOOP INSTEAD OF THREE.  `clf.mc_predict(X, M)` returns
      (mean_proba, H) from a single set of M stochastic forward passes;
      the confidence diagnostic reuses that mean_proba, and the hard
      predictions come from a single `.predict` call.

Also: the oracle boundary mask is now computed once per point (cached
in the pools) instead of being recomputed for the whole training set
every round.
"""
import time
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from .classifier import MCClassifier                          # noqa: E402
from .dynamics import newton_root                            # noqa: E402

__all__ = ["true_boundary_mask", "run_iterative_refinement", "print_report"]


def true_boundary_mask(X, p=3, band=0.06):
    """Cheap dynamical proxy for "this point is near the separatrix".

    A point is in the proxy band if its Newton iterate leaves the root
    neighbourhood (``|z| > 1.4``) within 6 steps.  Cheap and cacheable, and
    used ONLY for the in-sample diagnostic, never for training.  Its width is
    set implicitly by the 6-step threshold: against the geometric 1-cell band
    it has precision 0.19 / recall 0.90, i.e. it is a halo of about 6-8
    cells.  Prefer :func:`separatrix.evaluate.make_eval_grid` for anything
    that will be quoted.
    """
    out = np.zeros(len(X), dtype=bool)
    for i, (x, y) in enumerate(X):
        z = x + 1j * y
        in_band = False
        for _ in range(6):
            z = z - (z**p - 1.0) / (p * z ** (p - 1))
            if abs(z) > 1.4:        # escaped the root neighbourhood
                in_band = True
                break
        out[i] = in_band
    return out


def run_iterative_refinement(X0, y0, xg, yg,
                             rounds=5,
                             entropy_percentile=85,
                             oversample_factor=3,
                             max_samples=60_000,
                             boundary_fraction_start=0.15,
                             boundary_fraction_step=0.10,
                             boundary_fraction_max=0.60,
                             boundary_pool_cap=240_000,
                             warm_start=True,
                             warmup_epochs=150,
                             extra_epochs=120,
                             mc_samples=12,
                             backend="auto",
                             device=None,
                             hidden=(256, 256, 128),
                             seed=0,
                             verbose=True,
                             on_round=None,
                             return_model=False,
                             eval_set=None,
                             fair_bands=(1, 8),
                             keep_train_set=False):
    """
    One iterative-refinement loop per round:
      a) assemble a fixed-size training set (uniform base + boundary pool)
      b) fit / warm-start the classifier  ->  entropy field H(x)
      c) pick top-(100-p)th-percentile of H as "separatrix proxy"
      d) generate (oversample) more points there
      e) label them by REAL DYNAMICS (true sampling, unavoidable)
      f) MEASURE calibration health

    Parameters
    ----------
    max_samples : int
        Fixed training-set size per round (cost fix).  The old code grew
        without bound: 90k -> 257k over the 5 rounds.
    boundary_fraction_* : float
        Share of the training set taken from the accumulated boundary
        pool.  It rises round by round (0.15 -> 0.25 -> ... -> 0.60), which
        is exactly the imbalance mechanism the demo is about.
    warm_start : bool
        Continue training from the previous round's weights.
    warmup_epochs, extra_epochs : int
        Epoch budget for the first (cold) fit and for the warm-started ones.
    mc_samples : int
        Number of MC-dropout forward passes per round (feeds BOTH the
        entropy field and the confidence diagnostic).
    on_round : callable, optional
        `on_round(round_index, clf, log_entry)` called right after the
        diagnostics of each round, i.e. before the oversampling step.
        Used by sweep_capacity.py to evaluate every intermediate model on
        an independent evaluation grid.
    return_model : bool
        Also return the (warm-started) final classifier.
    eval_set : dict, optional
        Output of `evaluate.make_eval_grid(...)`.  When given, every round's
        model is ALSO scored on this fixed, independent population, and the
        results are stored as `fair_*` keys in the log.  Without it, the
        `boundary_acc` / `entropy_separability` keys keep their original
        (population-confounded) meaning -- see metric_bias.py.
    fair_bands : tuple of int
        Band widths (grid cells) used for the `fair_*` metrics.
    keep_train_set : bool
        Put a reference to the round's training set under the key
        `_X_train` in the log entry (used by island_study.py, and by any
        hook that needs to know what the sampler could see).

    Returns
    -------
    list of per-round diagnostic dicts, or `(log, clf)` when
    `return_model=True`.
    """
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    # ---------------- base (uniform) pool -------------------------------
    X_base = np.asarray(X0, dtype=np.float64)
    y_base = np.asarray(y0)
    if len(X_base) > max_samples:
        idx = rng.choice(len(X_base), max_samples, replace=False)
        X_base, y_base = X_base[idx], y_base[idx]
    base_mask = true_boundary_mask(X_base)          # oracle, cached once

    # ---------------- boundary pool (grows, but capped) ------------------
    X_bnd = np.empty((0, 2), dtype=np.float64)
    y_bnd = np.empty((0,), dtype=np.int8)
    bnd_mask = np.empty((0,), dtype=bool)

    xg_use = np.asarray(xg, dtype=float)
    yg_use = np.asarray(yg, dtype=float)
    lo, hi_lim = min(xg_use.min(), yg_use.min()), max(xg_use.max(), yg_use.max())

    n_cold = len(X_base)
    clf = None
    log = []

    for r in range(rounds):
        # ---- (a) assemble a BOUNDED training set -----------------------
        frac_b = min(boundary_fraction_max,
                     boundary_fraction_start + r * boundary_fraction_step)
        n_total = int(min(max_samples, len(X_base) + len(X_bnd)))
        n_b = int(min(len(X_bnd), round(frac_b * n_total)))
        n_u = max(1, n_total - n_b)

        idx_u = (rng.choice(len(X_base), n_u, replace=False)
                 if n_u < len(X_base) else np.arange(len(X_base)))
        idx_b = (rng.choice(len(X_bnd), n_b, replace=False)
                 if n_b > 0 else np.array([], dtype=int))

        X = np.vstack([X_base[idx_u], X_bnd[idx_b]])
        y = np.concatenate([y_base[idx_u], y_bnd[idx_b]])
        bmask = np.concatenate([base_mask[idx_u], bnd_mask[idx_b]])

        # ---- (b) fit (cold once, warm afterwards) ----------------------
        if clf is None:
            clf = MCClassifier(hidden=hidden, random_state=seed,
                               backend=backend, device=device)
        warm = bool(warm_start and r > 0)
        epochs = extra_epochs if warm else warmup_epochs
        t0 = time.perf_counter()
        clf.fit(X, y, warm_start=warm, max_iter=epochs)
        t_fit = time.perf_counter() - t0

        # ---- entropy + confidence from ONE MC loop (cost fix 4) --------
        t0 = time.perf_counter()
        mean_proba, H = clf.mc_predict(X, M=mc_samples)
        t_mc = time.perf_counter() - t0

        # ---- locate boundary proxy (high-entropy region) ---------------
        thr = np.percentile(H, entropy_percentile)
        hi = H >= thr

        # ---- diagnostics (BEFORE oversampling this round) --------------
        # NOTE (measurement trap, see metric_bias.py): these are computed on
        # the CURRENT training set, whose composition changes every round
        # (0% -> 55% boundary points).  Round-to-round differences therefore
        # mix model change with population change.  The same models scored on
        # a FIXED population (`eval_set=`) give the opposite sign.
        pred = clf.predict(X)
        correct = pred == y
        if bmask.any():
            boundary_acc = correct[bmask].mean()
            interior_acc = correct[~bmask].mean() if (~bmask).any() else np.nan
        else:
            boundary_acc = interior_acc = np.nan

        conf = mean_proba.max(axis=1)
        mean_conf = float(conf.mean())
        ece = float(np.abs(conf - correct.astype(float)).mean())   # |conf-acc|

        if bmask.any() and (~bmask).any():
            try:
                sep_auc = float(roc_auc_score(bmask.astype(int), H))
            except Exception:
                sep_auc = np.nan
        else:
            sep_auc = np.nan

        # ---- fair diagnostics on a FIXED evaluation population ---------
        fair = {}
        if eval_set is not None:
            from .evaluate import evaluate_on_grid
            fair = evaluate_on_grid(clf, eval_set, mc_samples=mc_samples,
                                    bands=fair_bands)
            fair = {f"fair_{k}": v for k, v in fair.items()}

        counts = {int(k): int((y == k).sum()) for k in np.unique(y)}
        balance_ratio = (min(counts.values()) / max(counts.values())
                         if len(counts) > 1 else 1.0)

        log.append(dict(round=r + 1,
                        n_samples=int(len(y)),
                        n_uniform=int(len(idx_u)),
                        n_boundary=int(len(idx_b)),
                        boundary_fraction=float(len(idx_b) / len(y)),
                        boundary_pool=int(len(X_bnd)),
                        entropy_threshold=float(thr),
                        hi_entropy_fraction=float(hi.mean()),
                        boundary_acc=float(boundary_acc),
                        interior_acc=float(interior_acc),
                        mean_confidence=mean_conf,
                        calibration_gap=float(mean_conf - correct.mean()),
                        expected_calibration_error=ece,
                        entropy_separability=sep_auc,
                        class_balance_ratio=float(balance_ratio),
                        class_counts=counts,
                        epochs=int(clf.n_epochs_),
                        warm_started=bool(warm),
                        fit_seconds=float(t_fit),
                        mc_seconds=float(t_mc),
                        backend=clf.describe(),
                        **fair))
        if verbose:
            d = log[-1]
            fair_txt = ""
            if "fair_acc_band1" in d:
                fair_txt = (f"  | FAIR (fixed pop): b1={d['fair_acc_band1']:.3f} "
                            f"b8={d.get('fair_acc_band8', float('nan')):.3f} "
                            f"ECE@b1={d['fair_ece_band1']:.3f} "
                            f"sep={d.get('fair_sep_auc_hard', float('nan')):.3f}")
            print(f"   round {d['round']}: n_train={d['n_samples']:>6} "
                  f"(uniform {d['n_uniform']:>6} / boundary {d['n_boundary']:>6}"
                  f" = {100*d['boundary_fraction']:4.1f}%)  "
                  f"epochs={d['epochs']:>3}{'w' if d['warm_started'] else 'c'}  "
                  f"fit={d['fit_seconds']:5.2f}s  mc={d['mc_seconds']:4.2f}s  "
                  f"bnd_acc(in-sample)={d['boundary_acc']:.3f} "
                  f"conf={d['mean_confidence']:.3f}{fair_txt}", flush=True)

        if on_round is not None:
            if keep_train_set:
                log[-1]["_X_train"] = X
            on_round(r, clf, log[-1])

        # ---- (c/d) oversample high-entropy region (THE PAPER'S STEP) ---
        X_hi = X[hi]
        n_new = len(X_hi) * (oversample_factor - 1)
        if n_new == 0:
            continue
        jitter = rng.normal(scale=0.03, size=(n_new, 2))
        new_xy = X_hi.repeat(oversample_factor - 1, axis=0) + jitter
        new_xy = np.clip(new_xy, lo, hi_lim)

        # ---- (e) label by REAL DYNAMICS (true sampling, unavoidable) ---
        new_y = np.asarray([newton_root(x + 1j * yc, p=3)
                            for x, yc in new_xy], dtype=np.int8)
        keep = new_y > 0
        new_xy, new_y = new_xy[keep], new_y[keep]
        if len(new_xy) == 0:
            continue

        X_bnd = np.vstack([X_bnd, new_xy])
        y_bnd = np.concatenate([y_bnd, new_y])
        bnd_mask = np.concatenate([bnd_mask, true_boundary_mask(new_xy)])

        # keep the pool bounded in memory as well
        if len(X_bnd) > boundary_pool_cap:
            sel = rng.choice(len(X_bnd), boundary_pool_cap, replace=False)
            X_bnd, y_bnd, bnd_mask = X_bnd[sel], y_bnd[sel], bnd_mask[sel]

    if verbose:
        print(f"   [refinement] total {time.perf_counter()-t_start:5.1f}s "
              f"({n_cold} uniform points, {clf.describe()})", flush=True)
    return (log, clf) if return_model else log


def print_report(log):
    print("\n" + "=" * 100)
    print(" TABLE A -- IN-SAMPLE (population changes every round: confounded)")
    print("=" * 100)
    print("{:>6} {:>8} {:>7} {:>9} {:>12} {:>11} {:>10} {:>8} {:>8}"
          .format("round", "n_train", "bnd%", "balance", "boundary_acc",
                  "interior_acc", "confidence", "gap", "sep_AUC"))
    print("-" * 100)
    for d in log:
        print("{:>6} {:>8} {:>7.1f} {:>9.2f} {:>12.3f} {:>11.3f} "
              "{:>10.3f} {:>8.3f} {:>8.3f}"
              .format(d["round"], d["n_samples"],
                      100 * d["boundary_fraction"],
                      d["class_balance_ratio"],
                      d["boundary_acc"], d["interior_acc"],
                      d["mean_confidence"],
                      d.get("calibration_gap", float("nan")),
                      d["entropy_separability"]))

    if "fair_acc_band1" in log[0]:
        print("\n" + "=" * 100)
        print(" TABLE B -- FIXED POPULATION (independent grid, geometric "
              "fractal bands)")
        print("=" * 100)
        print("{:>6} {:>14} {:>14} {:>14} {:>15} {:>12}"
              .format("round", "acc band1", "acc band8", "acc interior",
                      "ECE in band1", "sep_AUC"))
        print("-" * 100)
        for d in log:
            print("{:>6} {:>14.3f} {:>14.3f} {:>14.3f} {:>15.3f} {:>12.3f}"
                  .format(d["round"], d["fair_acc_band1"],
                          d.get("fair_acc_band8", float("nan")),
                          d["fair_acc_interior"],
                          d["fair_ece_band1"],
                          d.get("fair_sep_auc_hard", float("nan"))))

    print("\nREADING THE TABLES")
    print(" * bnd%        share of the training set taken from the oversampled")
    print("               boundary pool (0% -> 55%): the distribution really")
    print("               does become boundary-dominated.")
    print(" * TABLE A is computed on the CURRENT training set, whose")
    print("   composition changes every round.  Its 'boundary_acc' therefore")
    print("   drops by ~5 points even though the models get BETTER: the")
    print("   metric's population, not the model, is what changes.")
    print(" * TABLE B scores the same models on a fixed, independent")
    print("   population.  This is the number to trust: separatrix accuracy")
    print("   RISES, and the calibration error inside the band FALLS.")
    print(" * 'band1' means 'within 1 grid cell of a true label change'.")
    print("   The separatrix has Lebesgue measure zero, so every 'boundary")
    print("   accuracy' is a band-width-dependent number -- see")
    print("   separatrix/evaluate.py and the band-width table from")
    print("   experiments/sweep_capacity.py.")


if __name__ == "__main__":                       # pragma: no cover
    from .dataset import make_dataset
    from .evaluate import make_eval_grid
    from .paths import result
    import json

    X, y, grid = make_dataset(n=300)
    ev = make_eval_grid(n=401)
    log = run_iterative_refinement(X, y, grid[0], grid[1], rounds=5,
                                   eval_set=ev)
    print_report(log)
    with open(result("demo_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
