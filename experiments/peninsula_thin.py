"""Peninsula arm 1: recovery as a function of corridor thickness.

Measured on the paper's own benchmark, with no constructed potential (see
:mod:`separatrix.peninsula` for why a smooth 2-D flow cannot supply a finger).

A cell's corridor thickness is ``t = 2 * distance to the nearest cell of
another basin``.  Thin corridors are the fingers hanging off the densely
sampled boundary; thick cells are basin interiors.  Evaluation cells are
stratified by ``t``, and two numbers are reported per stratum:

    recovery = fraction of cells labelled with their own basin
    freeze   = fraction labelled with the *neighbouring* basin

Sweeping the training-grid spacing ``h`` then shows whether the transition is
governed by ``t/h`` (the peninsula analogue of the island criterion ``w/h``)
and how capacity moves it.

Run
---
    python experiments/peninsula_thin.py --check
    python experiments/peninsula_thin.py --sweep --seeds 0 1 2
    python experiments/peninsula_thin.py --figure results/thin.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from separatrix.classifier import MCClassifier
from separatrix.paths import figure, result
from separatrix.peninsula import (CAPS, cached_fine_labels, prepare_thickness,
                                  raster_train_grid, thickness)


def check():
    """Ground-truth sanity: label mix and thickness of the evaluation cells."""
    strata, _ = prepare_thickness(seed=0, per_bin=300)
    print(f"{'bin':>4} {'t range':>16} {'t_mean':>8} {'cells':>6} "
          f"{'#labels':>8}")
    for s in strata:
        vals, counts = np.unique(s["label"], return_counts=True)
        share = dict(zip(vals.tolist(),
                         (100 * counts / counts.sum()).round(0).tolist()))
        print(f"{s['bin']:>4} [{s['lo']:.4f},{s['hi']:.4f}) "
              f"{s['t'].mean():>8.4f} {s['n']:>6} {share}")


def run_point(n, cap, seed, strata, labels=None):
    """Train one classifier on an ``n x n`` grid and score each stratum."""
    X, y, h = raster_train_grid(n, labels, seed)
    clf = MCClassifier(hidden=CAPS[cap], random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X, y, max_iter=250)
    wall = time.perf_counter() - t0
    rows = []
    for s in strata:
        pred = clf.predict(s["xy"])
        has_nb = s["near"] > 0
        rows.append(dict(bin=s["bin"], t_lo=s["lo"], t_hi=s["hi"],
                         t_mean=float(np.mean(s["t"])),
                         recovery=float((pred == s["label"]).mean()),
                         freeze=(float((pred[has_nb] == s["near"][has_nb]).mean())
                                 if has_nb.any() else float("nan")),
                         freeze_cov=float(has_nb.mean()),
                         n=len(s["xy"])))
    return rows, h, int(len(X)), round(wall, 2)


def sweep(args):
    labels = cached_fine_labels()
    t = thickness(labels)[0]
    strata, _ = prepare_thickness(labels=labels, t=t, seed=0,
                                  per_bin=args.per_bin)
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    new = not os.path.exists(out)
    f = open(out, "a", newline="", encoding="utf-8")
    fields = ["n_grid", "h", "cap", "seed", "bin", "t_lo", "t_hi", "t_mean",
              "w_over_h", "recovery", "freeze", "freeze_cov", "n_cells",
              "n_train", "wall_s"]
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for n in args.grid:
        for cap in args.caps:
            for seed in args.seeds:
                rows, h, n_train, wall = run_point(n, cap, seed, strata, labels)
                for r in rows:
                    r.update(n_grid=n, h=h, cap=cap, seed=seed,
                             w_over_h=r["t_mean"] / h, n_cells=r["n"],
                             n_train=n_train, wall_s=wall)
                    w.writerow({k: r[k] for k in fields})
                f.flush()
                line = "  ".join(f"b{r['bin']}:{r['recovery']:.2f}" for r in rows)
                print(f"grid={n:>3} {cap:>5} seed={seed}  {line}  "
                      f"({wall:.1f}s)", flush=True)
    f.close()
    print(f"wrote {out}")


def make_figure(path=None, out=None):
    """Three panels: the knee, density insensitivity, neighbour takeover."""
    path = path or str(result("thin.csv"))
    out = out or figure("fig10_peninsula_thin.png")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("h", "t_mean", "w_over_h", "recovery", "freeze", "t_lo"):
            r[k] = float(r[k])
        r["bin"] = int(r["bin"])
        r["n_grid"] = int(r["n_grid"])

    def agg(family_rows, key):
        groups = {}
        for r in family_rows:
            groups.setdefault((r["cap"], r["n_grid"], r["bin"]),
                              []).append(r[key])
        return {k: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0)
                for k, v in groups.items()}

    colors = {"w32": "#95a5a6", "w128": "#2980b9", "w512": "#c0392b"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) recovery vs absolute corridor thickness, per capacity, finest grid
    ax = axes[0]
    ng = max(r["n_grid"] for r in rows)
    sub = [r for r in rows if r["n_grid"] == ng]
    a_rec = agg(sub, "recovery")
    for cap in sorted({r["cap"] for r in sub}):
        pts = sorted([r for r in sub if r["cap"] == cap],
                     key=lambda r: r["t_mean"])
        x = [r["t_mean"] for r in pts]
        m = [a_rec[(cap, ng, r["bin"])][0] for r in pts]
        s = [a_rec[(cap, ng, r["bin"])][1] for r in pts]
        ax.errorbar(x, m, yerr=s, marker="o", color=colors.get(cap, "k"),
                    capsize=3, label=cap, lw=1.5)
    ax.set_xscale("log")
    ax.axhline(1 / 3, color="k", lw=0.7, ls=":", label="chance (3 basins)")
    ax.set_xlabel("corridor thickness t (plane units)")
    ax.set_ylabel("recovery inside the corridor")
    ax.set_title(f"(a) Thin fingers are frozen; capacity moves the\n"
                 f"knee (grid {ng}x{ng}, mean ± std over seeds)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) density insensitivity at fixed capacity
    ax = axes[1]
    for cap in ("w32", "w512"):
        for ng2, ls in zip(sorted({r["n_grid"] for r in rows}),
                           ("-", "--", ":")):
            pts = sorted([r for r in rows
                          if r["cap"] == cap and r["n_grid"] == ng2],
                         key=lambda r: r["t_mean"])
            if not pts:
                continue
            a2 = agg([r for r in rows
                      if r["cap"] == cap and r["n_grid"] == ng2], "recovery")
            ax.plot([r["t_mean"] for r in pts],
                    [a2[(cap, ng2, r["bin"])][0] for r in pts],
                    marker="o", ms=4, ls=ls, color=colors[cap], alpha=0.8,
                    label=f"{cap}, grid {ng2}")
    ax.set_xscale("log")
    ax.set_xlabel("corridor thickness t")
    ax.set_ylabel("recovery")
    ax.set_title("(b) Sampling density barely matters:\n"
                 "the root is already densely sampled", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    # (c) freeze (labelled as the neighbouring basin)
    ax = axes[2]
    a_frz = agg(sub, "freeze")
    for cap in sorted({r["cap"] for r in sub}):
        pts = sorted([r for r in sub if r["cap"] == cap],
                     key=lambda r: r["t_mean"])
        ax.plot([r["t_mean"] for r in pts],
                [a_frz[(cap, ng, r["bin"])][0] for r in pts],
                marker="s", color=colors.get(cap, "k"), label=cap)
    ax.set_xscale("log")
    ax.set_xlabel("corridor thickness t")
    ax.set_ylabel("frozen into the neighbouring basin")
    ax.set_title("(c) The failure mode: neighbours take over", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Peninsula freezing on the paper's own fractal, by corridor "
                 "thickness", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="print the ground-truth strata and exit")
    ap.add_argument("--sweep", action="store_true",
                    help="run the (grid, capacity, seed) sweep into --out")
    ap.add_argument("--figure", nargs="?", const="", default=None,
                    help="draw the figure from --out (or from the given CSV)")
    ap.add_argument("--grid", nargs="+", type=int, default=[200, 400, 800])
    ap.add_argument("--caps", nargs="+", default=["w32", "w128", "w512"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--per-bin", type=int, default=1200)
    ap.add_argument("--out", default=str(result("thin.csv")))
    a = ap.parse_args()
    if a.check:
        check()
    if a.sweep:
        sweep(a)
    if a.figure is not None:
        make_figure(a.figure or a.out)


if __name__ == "__main__":
    main()
