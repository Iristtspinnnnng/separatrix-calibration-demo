"""sweep_matched.py
========================================================================
Does the capacity trend survive when the training budget is matched?

sweep_capacity.py gave every network the same EPOCH count, so wide nets got
far more FLOPs per round and tiny nets got almost none.  Two stricter
controls, both with the same 5-round refinement loop, same seeds, same
fixed 60k training set and the same independent 401x401 evaluation grid:

  fixed-epoch      : the baseline (150 cold / 120 warm epochs each)   [sweep_capacity]
  train-to-plateau : every fit runs until its own validation loss
                     plateaus (early stopping, cap 400) -> removes
                     "the big net was undertrained"
  matched-FLOPs    : every configuration spends the SAME total work
                     ~ n_train * epochs * params  per round
                     (reference = the 256x256 budget; epochs capped)

Reported per configuration and regime, at round 1 and round 5:
  band-1 / band-8 accuracy on the fixed population, ECE inside band 1,
  epochs used, and total work = sum_r(epochs_r * n_train * params).

Run:  python sweep_matched.py            (~10-15 min on a 4060)
      python sweep_matched.py --quick    (3 configs, fixed + plateau only)
========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys
import json
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from separatrix.dataset import make_dataset
from separatrix.refine import run_iterative_refinement
from separatrix.evaluate import make_eval_grid, evaluate_on_grid, params_of
from separatrix.paths import figure, result

CONFIGS = [(16, 16), (64, 64), (128, 128), (256, 256), (512, 512),
           (256, 256, 256)]
QUICK = [(16, 16), (128, 128), (256, 256)]
BANDS = (1, 8)

REF_PARAMS, REF_EPOCHS = params_of((256, 256)), 150     # matched-FLOPs anchor
FLOP_CAP, FLOP_FLOOR = 1500, 6
N_TRAIN = 60_000


def name_of(hidden):
    return "x".join(str(h) for h in hidden)


def flop_epochs(hidden):
    p = params_of(hidden)
    return int(np.clip(round(REF_PARAMS * REF_EPOCHS / p), FLOP_FLOOR, FLOP_CAP))


def run_one(hidden, X, y, grid, ev, warmup, extra, rounds=5, seed=0):
    store = []

    def hook(r, clf, entry):
        m = evaluate_on_grid(clf, ev, mc_samples=12, bands=BANDS)
        m.update(round=r + 1, epochs=entry["epochs"])
        store.append(m)

    t0 = time.perf_counter()
    run_iterative_refinement(X, y, grid[0], grid[1], rounds=rounds,
                             hidden=hidden, seed=seed,
                             warmup_epochs=warmup, extra_epochs=extra,
                             max_samples=N_TRAIN, on_round=hook, verbose=False)
    wall = time.perf_counter() - t0
    epochs = [m["epochs"] for m in store]
    work = float(sum(e * N_TRAIN * params_of(hidden) for e in epochs))
    r1, r5 = store[0], store[-1]
    return dict(tag=f"{name_of(hidden)}", hidden=list(hidden),
                params=params_of(hidden),
                regime=None, epochs_r1=int(epochs[0]),
                epochs_total=int(sum(epochs)),
                work=work, wall_s=round(wall, 1),
                acc_band1_r1=r1["acc_band1"], acc_band1_r5=r5["acc_band1"],
                d_acc_band1=r5["acc_band1"] - r1["acc_band1"],
                acc_band8_r1=r1["acc_band8"], acc_band8_r5=r5["acc_band8"],
                d_acc_band8=r5["acc_band8"] - r1["acc_band8"],
                acc_int_r5=r5["acc_interior"],
                ece_band1_r1=r1["ece_band1"], ece_band1_r5=r5["ece_band1"],
                sep_hard_r1=r1.get("sep_auc_hard"),
                sep_hard_r5=r5.get("sep_auc_hard"))


def main(quick=False):
    X, y, grid = make_dataset(n=300)
    ev = make_eval_grid(n=401, bands=BANDS, interior_band=16)
    configs = QUICK if quick else CONFIGS

    regimes = [("fixed-epoch", 150, 120), ("train-to-plateau", 400, 400)]
    if not quick:
        regimes.append(("matched-FLOPs", None, None))

    rows = []
    for regime, warmup, extra in regimes:
        print(f"\n=== regime: {regime} ===", flush=True)
        for hidden in configs:
            if regime == "matched-FLOPs":
                ep = flop_epochs(hidden)
                warmup = extra = ep
            r = run_one(hidden, X, y, grid, ev, warmup, extra)
            r["regime"] = regime
            r["budget_epochs"] = int(warmup)
            rows.append(r)
            print(f"  {r['tag']:>16} params={r['params']:>7} "
                  f"ep/round={r['budget_epochs']:>5} used={r['epochs_total']:>5} "
                  f"band1 {r['acc_band1_r1']:.3f}->{r['acc_band1_r5']:.3f} "
                  f"({r['d_acc_band1']:+.3f})  band8 "
                  f"{r['acc_band8_r1']:.3f}->{r['acc_band8_r5']:.3f} "
                  f" ECE@b1 {r['ece_band1_r1']:.3f}->{r['ece_band1_r5']:.3f} "
                  f" work={r['work']/1e9:6.2f}e9  {r['wall_s']:5.1f}s",
                  flush=True)

    keys = list(rows[0].keys())
    with open(str(result("matched_sweep.csv")), "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    with open(str(result("matched_sweep.json")), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("\nwrote matched_sweep.csv / .json")

    make_figure(rows, regimes=[r[0] for r in regimes])
    analyse(rows)


def make_figure(rows, regimes, out=figure("fig4_matched_budget.png")):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    colors = {"fixed-epoch": "#7f8c8d", "train-to-plateau": "#2980b9",
              "matched-FLOPs": "#c0392b"}
    markers = {"fixed-epoch": "o", "train-to-plateau": "s",
               "matched-FLOPs": "^"}

    # (a) round-5 band-1 accuracy vs capacity, per regime
    ax = axes[0, 0]
    for reg in regimes:
        sub = sorted([r for r in rows if r["regime"] == reg],
                     key=lambda r: r["params"])
        ax.plot([r["params"] for r in sub], [r["acc_band1_r5"] for r in sub],
                marker=markers[reg], color=colors[reg], label=reg)
    ax.set_xscale("log")
    ax.set_xlabel("MLP parameters"); ax.set_ylabel("band-1 accuracy (round 5)")
    ax.set_title("(a) Capacity trend under three budgets", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) refinement gain vs capacity
    ax = axes[0, 1]
    for reg in regimes:
        sub = sorted([r for r in rows if r["regime"] == reg],
                     key=lambda r: r["params"])
        ax.plot([r["params"] for r in sub], [r["d_acc_band1"] for r in sub],
                marker=markers[reg], color=colors[reg], label=reg)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("MLP parameters")
    ax.set_ylabel("round 5 - round 1 (band-1 accuracy)")
    ax.set_title("(b) Is the gain from oversampling size-dependent?", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) what was actually matched: work vs params
    ax = axes[1, 0]
    for reg in regimes:
        sub = sorted([r for r in rows if r["regime"] == reg],
                     key=lambda r: r["params"])
        ax.plot([r["params"] for r in sub], [r["work"] / 1e9 for r in sub],
                marker=markers[reg], color=colors[reg], label=reg)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MLP parameters")
    ax.set_ylabel("training work  sum(epochs * n * params)  [1e9]")
    ax.set_title("(c) Work spent per regime", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (d) compute-efficiency frontier
    ax = axes[1, 1]
    for reg in regimes:
        sub = sorted([r for r in rows if r["regime"] == reg],
                     key=lambda r: r["work"])
        ax.plot([r["work"] / 1e9 for r in sub], [r["acc_band1_r5"] for r in sub],
                marker=markers[reg], color=colors[reg], label=reg)
        for r in sub:
            ax.annotate(r["tag"], (r["work"] / 1e9, r["acc_band1_r5"]),
                        fontsize=6.5, xytext=(2, 3),
                        textcoords="offset points", color=colors[reg])
    ax.set_xscale("log")
    ax.set_xlabel("training work [1e9]")
    ax.set_ylabel("band-1 accuracy (round 5)")
    ax.set_title("(d) Accuracy per unit of compute", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Matched-budget control: capacity vs optimisation budget",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def analyse(rows):
    print("\n--- summary ---")
    for reg in sorted({r["regime"] for r in rows}):
        sub = sorted([r for r in rows if r["regime"] == reg],
                     key=lambda r: r["params"])
        small, big = sub[0], sub[-1]
        print(f"{reg:>17}: band1 r1 {small['acc_band1_r1']:.3f} -> "
              f"{big['acc_band1_r1']:.3f}  "
              f"(capacity gain {big['acc_band1_r1']-small['acc_band1_r1']:+.3f})"
              f"   refinement gain {small['d_acc_band1']:+.3f} .. "
              f"{max(r['d_acc_band1'] for r in sub):+.3f}")
    print("\nIf the capacity ordering and the sign of the refinement gain "
          "survive the plateau and matched-FLOPs regimes, the trend is "
          "expressivity, not undertraining.")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
