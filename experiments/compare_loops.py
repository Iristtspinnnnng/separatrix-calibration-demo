"""compare_loops.py -- the ORIGINAL unbounded refinement loop (rebuilt from
the pre-optimisation source) vs the new bounded+warm-started one, both on
the torch/GPU backend, so the diagnostic table can be compared fairly.

Run:  python compare_loops.py 300
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys, time
import numpy as np

from separatrix.dataset import make_dataset
from separatrix.dynamics import newton_root
from separatrix.classifier import MCClassifier
from separatrix.refine import true_boundary_mask, run_iterative_refinement


def original_loop(X0, y0, xg, yg, rounds=5, entropy_percentile=85,
                  oversample_factor=3, seed=0, mc=12):
    """Verbatim logic of the old oversample.run_iterative_refinement."""
    rng = np.random.default_rng(seed)
    X, y = X0.copy(), y0.copy()
    log = []
    xg_use = np.asarray(xg)
    for r in range(rounds):
        t0 = time.perf_counter()
        clf = MCClassifier(random_state=seed + r).fit(X, y)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        H = clf.mc_entropy(X, M=mc)
        proba = clf.predict_proba(X)          # n_mc=30 internally
        t_mc = time.perf_counter() - t0
        thr = np.percentile(H, entropy_percentile)
        hi = H >= thr

        bmask = true_boundary_mask(X)
        pred = clf.predict(X)
        correct = pred == y
        boundary_acc = correct[bmask].mean() if bmask.any() else np.nan
        interior_acc = correct[~bmask].mean() if (~bmask).any() else np.nan
        conf = proba.max(axis=1).mean()
        from sklearn.metrics import roc_auc_score
        sep = roc_auc_score(bmask.astype(int), H)
        counts = {int(k): int((y == k).sum()) for k in np.unique(y)}
        log.append(dict(round=r + 1, n_samples=len(y),
                        boundary_fraction=float(hi.mean()),
                        class_balance_ratio=min(counts.values()) / max(counts.values()),
                        boundary_acc=float(boundary_acc),
                        interior_acc=float(interior_acc),
                        mean_confidence=float(conf),
                        entropy_separability=float(sep),
                        epochs=clf.n_epochs_, fit_seconds=t_fit,
                        mc_seconds=t_mc))

        X_hi = X[hi]
        jitter = rng.normal(scale=0.03, size=(len(X_hi) * (oversample_factor - 1), 2))
        new_xy = X_hi.repeat(oversample_factor - 1, axis=0) + jitter
        new_xy = np.clip(new_xy, xg_use.min(), xg_use.max())
        new_y = np.asarray([newton_root(x + 1j * yc, p=3) for x, yc in new_xy],
                           dtype=np.int8)
        keep = new_y > 0
        new_xy, new_y = new_xy[keep], new_y[keep]
        X = np.vstack([X, new_xy])
        y = np.concatenate([y, new_y])
        print(f"  [orig round {r+1}] n={len(y):>7} fit={t_fit:5.1f}s mc={t_mc:4.1f}s "
              f"bnd_acc={boundary_acc:.3f} conf={conf:.3f} sep={sep:.3f}", flush=True)
    return log


def show(log, title):
    print(f"\n--- {title} ---")
    print("{:>6} {:>8} {:>8} {:>12} {:>11} {:>10} {:>8} {:>6} {:>7}".format(
        "round", "n_train", "bnd%", "boundary_acc", "interior_acc",
        "confidence", "sep_AUC", "epochs", "fit_s"))
    for d in log:
        bf = 100 * d.get("boundary_fraction", np.nan)
        print("{:>6} {:>8} {:>8.1f} {:>12.3f} {:>11.3f} {:>10.3f} {:>8.3f} {:>6} {:>7.1f}"
              .format(d["round"], d["n_samples"], bf, d["boundary_acc"],
                      d["interior_acc"], d["mean_confidence"],
                      d["entropy_separability"], d["epochs"], d["fit_seconds"]))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    X, y, grid = make_dataset(n=n)
    print(f"n={len(X)}")

    t0 = time.perf_counter()
    log_orig = original_loop(X, y, grid[0], grid[1], mc=12)
    t_orig = time.perf_counter() - t0
    show(log_orig, f"ORIGINAL (unbounded, from scratch) total {t_orig:.1f}s")

    for ee in (25, 120):
        t0 = time.perf_counter()
        log_new = run_iterative_refinement(X, y, grid[0], grid[1], rounds=5,
                                           warmup_epochs=150, extra_epochs=ee,
                                           verbose=False)
        t_new = time.perf_counter() - t0
        show(log_new, f"NEW bounded+warm (extra_epochs={ee}) total {t_new:.1f}s")
