"""Is "boundary accuracy falls" a property of the model, or of the metric?

The refinement loop scores `boundary_acc` on the **current training set**.  That
set is re-drawn every round (0 % -> 55 % boundary points), so a round-to-round
difference mixes model change with population change.  Here each round's model
is also scored on a **fixed** held-out uniform population with one oracle held
fixed, which separates the two.

Output (5 rounds, 300x300 grid, seed 0):

    escape-proxy accuracy on the CHANGING training set : 0.819 -> 0.772  (-0.047)
    escape-proxy accuracy on a FIXED uniform population: 0.816 -> 0.890  (+0.074)

This is the single most important methodological result of the repository: the
same models, the same oracle, opposite signs.

Run
---
    python experiments/metric_bias.py        # ~25 s on a 4060
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np

from separatrix.dataset import make_dataset
from separatrix.evaluate import evaluate_on_grid, make_eval_grid
from separatrix.paths import result
from separatrix.refine import run_iterative_refinement, true_boundary_mask


def main():
    X, y, grid = make_dataset(n=300)
    ev = make_eval_grid(n=401, bands=(1, 8))

    # fixed population: uniform points, never used to fit the metric
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(X), 60_000, replace=False)
    X_pop, y_pop = X[idx], y[idx]
    esc_pop = true_boundary_mask(X_pop)          # the old dynamical proxy

    rows = []

    def hook(r, clf, entry):
        pred = clf.predict(X_pop)
        ok = pred == y_pop
        g = evaluate_on_grid(clf, ev, mc_samples=12, bands=(1, 8))
        rows.append(dict(round=r + 1,
                         escape_acc_train=float(entry["boundary_acc"]),
                         escape_acc_pop=float(ok[esc_pop].mean()),
                         interior_acc_pop=float(ok[~esc_pop].mean()),
                         band1_pop=float(g["acc_band1"]),
                         band8_pop=float(g["acc_band8"]),
                         conf_band1=float(g["conf_band1"]),
                         ece_band1=float(g["ece_band1"])))

    run_iterative_refinement(X, y, grid[0], grid[1], rounds=5, seed=0,
                             hidden=(256, 256), on_round=hook, verbose=False)

    print(f"\n{'round':>6} {'esc(train set)':>15} {'esc(fixed pop)':>15} "
          f"{'int(fixed pop)':>15} {'band1(pop)':>11} {'band8(pop)':>11} "
          f"{'ece@b1':>8}")
    for d in rows:
        print(f"{d['round']:>6} {d['escape_acc_train']:>15.3f} "
              f"{d['escape_acc_pop']:>15.3f} {d['interior_acc_pop']:>15.3f} "
              f"{d['band1_pop']:>11.3f} {d['band8_pop']:>11.3f} "
              f"{d['ece_band1']:>8.3f}")

    e_train = [d["escape_acc_train"] for d in rows]
    e_pop = [d["escape_acc_pop"] for d in rows]
    d_train = e_train[-1] - e_train[0]
    d_pop = e_pop[-1] - e_pop[0]
    print(f"\nescape-proxy accuracy on the CHANGING training set : "
          f"{e_train[0]:.3f} -> {e_train[-1]:.3f}  ({d_train:+.3f})")
    print(f"escape-proxy accuracy on a FIXED uniform population: "
          f"{e_pop[0]:.3f} -> {e_pop[-1]:.3f}  ({d_pop:+.3f})")

    with open(result("metric_bias.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows,
                       changing_training_set=[round(v, 3) for v in e_train],
                       fixed_population=[round(v, 3) for v in e_pop],
                       delta_train=round(d_train, 3),
                       delta_population=round(d_pop, 3)), f, indent=1)
    print("wrote", result("metric_bias.json"))


if __name__ == "__main__":
    main()
