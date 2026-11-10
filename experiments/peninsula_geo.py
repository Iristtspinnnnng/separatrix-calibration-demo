"""Peninsula arm 2: the root-to-tip axis, i.e. recovery as ``f(t, d)``.

The thickness experiment answers "how thin must a corridor be to be lost?".
This one adds the second coordinate of the peninsula story: a peninsula is
attached to the densely sampled main boundary, so its ROOT should survive
while its TIP is lost.

    t = corridor thickness            (separatrix.peninsula.thickness)
    d = distance from the thick core  (separatrix.peninsula.core_distance)

``d`` is computed with a distance transform from the core cells (``t >=
0.05``) of each basin separately.  It is a *Euclidean* distance; ``--validate``
compares it with a local BFS geodesic on random cells and prints the
agreement (median absolute difference 0.0009 units, 90th percentile 0.0058 =
1-6 fine cells), which is why only bins of width >= 0.02 are used: for
winding corridors the Euclidean value is a lower bound on the geodesic one.

Run
---
    python experiments/peninsula_geo.py --validate
    python experiments/peninsula_geo.py --sweep --seeds 0 1 2   # -> results/geo.csv
    python experiments/peninsula_geo.py --figure results/geo.csv
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
from separatrix.peninsula import (CAPS, CELL, XLIM, prepare_core,
                                  geodesic_core_distance, raster_train_grid)


def validate(n=150, seed=0):
    """Compare the Euclidean ``d`` with a local BFS geodesic distance."""
    strata, labels, t, d = prepare_core(seed=seed, per_cell=10, t_max=0.05)
    rng = np.random.default_rng(seed + 1)
    pts = []
    for s in strata:
        pts.extend(list(zip(s["xy"], s["label"])))
    if not pts:
        print("no cells")
        return
    sel = rng.choice(len(pts), min(n, len(pts)), replace=False)
    ratios, diffs = [], []
    for i in sel:
        xy, own = pts[i]
        xg = int(round((xy[0] - XLIM[0]) / CELL))
        yg = int(round((xy[1] - XLIM[0]) / CELL))
        geo = geodesic_core_distance(labels, t, yg, xg, int(own))
        if not np.isfinite(geo) or geo <= 0:
            continue
        euc = float(d[yg, xg])
        ratios.append(euc / geo)
        diffs.append(abs(euc - geo))
    ratios = np.array(ratios)
    diffs = np.array(diffs)
    print(f"compared {len(ratios)} cells (t < 0.05)")
    print(f"  Euclidean/geodesic ratio: median {np.median(ratios):.2f}, "
          f"10-90% [{np.percentile(ratios, 10):.2f}, "
          f"{np.percentile(ratios, 90):.2f}]")
    print(f"  absolute difference (units): median {np.median(diffs):.4f}, "
          f"90% {np.percentile(diffs, 90):.4f}")
    print("  -> bins of width 0.02+ are safe; treat 0.01 bins as approximate")


def run_point(n_grid, cap, seed, strata, labels=None):
    """Train one classifier and score each (t, d) stratum."""
    X, y, h = raster_train_grid(n_grid, labels, seed)
    clf = MCClassifier(hidden=CAPS[cap], random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X, y, max_iter=250)
    wall = time.perf_counter() - t0
    rows = []
    for s in strata:
        pred = clf.predict(s["xy"])
        has = s["near"] > 0
        rows.append(dict(d_bin=s["d_bin"], d_lo=s["d_lo"], d_hi=s["d_hi"],
                         t_mean=s["t_mean"], d_mean=s["d_mean"],
                         recovery=float((pred == s["label"]).mean()),
                         freeze=(float((pred[has] == s["near"][has]).mean())
                                 if has.any() else float("nan")),
                         n=s["n"]))
    return rows, h, int(len(X)), round(wall, 2)


def sweep(args):
    strata, labels, t, d = prepare_core(seed=args.seed0,
                                        per_cell=args.per_cell,
                                        t_max=args.t_max)
    print(f"{len(strata)} (t, d) strata, t<{args.t_max}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="", encoding="utf-8")
    fields = ["n_grid", "h", "cap", "seed", "d_bin", "d_lo", "d_hi", "t_mean",
              "d_mean", "recovery", "freeze", "n_cells", "n_train", "wall_s"]
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for n_grid in args.grid:
        for cap in args.caps:
            for seed in args.seeds:
                rows, h, n_train, wall = run_point(n_grid, cap, seed, strata,
                                                   labels)
                for r in rows:
                    r.update(n_grid=n_grid, h=h, cap=cap, seed=seed,
                             n_cells=r["n"], n_train=n_train, wall_s=wall)
                    w.writerow({k: r[k] for k in fields})
                f.flush()
                by_d = {}
                for r in rows:
                    by_d.setdefault(r["d_bin"], []).append(r["recovery"])
                line = "  ".join(f"d{k}:{np.mean(v):.2f}"
                                 for k, v in sorted(by_d.items()))
                print(f"grid={n_grid:>3} {cap:>5} seed={seed}  {line}  "
                      f"({wall:.1f}s)", flush=True)
    f.close()
    print(f"wrote {args.out}")


def make_figure(path=None, out=None):
    """(a) root -> tip profile, (b) capacity x position, (c) freeze."""
    path = path or str(result("geo.csv"))
    out = out or figure("fig11_peninsula_geo.png")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("t_mean", "d_mean", "d_lo", "d_hi", "recovery", "freeze"):
            r[k] = float(r[k])
        r["d_bin"] = int(r["d_bin"])
        r["n_grid"] = int(r["n_grid"])

    def group(key, sub):
        out = {}
        for r in sub:
            out.setdefault((r["cap"], r["d_bin"]), []).append(r[key])
        return {k: (np.nanmean(v), np.nanstd(v, ddof=1) if len(v) > 1 else 0.0)
                for k, v in out.items()}

    caps = sorted({r["cap"] for r in rows})
    colors = {"w32": "#95a5a6", "w128": "#2980b9", "w512": "#c0392b"}
    ng = max(r["n_grid"] for r in rows)
    sub = [r for r in rows if r["n_grid"] == ng]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) recovery vs distance from the root
    ax = axes[0]
    grec = group("recovery", sub)
    for cap in caps:
        db = sorted({r["d_bin"] for r in sub if r["cap"] == cap})
        m = [grec[(cap, k)][0] for k in db]
        s = [grec[(cap, k)][1] for k in db]
        ax.errorbar(db, m, yerr=s, marker="o", color=colors.get(cap, "k"),
                    capsize=3, label=cap)
    ax.axhline(1 / 3, color="k", lw=0.7, ls=":", label="chance")
    ax.set_xlabel("distance from the thick core (bin index: 1 = root)")
    ax.set_ylabel("recovery (thin corridors)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"(a) Root -> tip, thin corridors\n(grid {ng}x{ng})",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) heat map recovery(capacity, d)
    ax = axes[1]
    dbins = sorted({r["d_bin"] for r in sub})
    M = np.full((len(caps), len(dbins)), np.nan)
    for i, cap in enumerate(caps):
        for j, db in enumerate(dbins):
            v = grec.get((cap, db))
            if v:
                M[i, j] = v[0]
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(caps)))
    ax.set_yticklabels(caps)
    ax.set_xticks(range(len(dbins)))
    lab_by = {}
    for r in sub:
        lab_by.setdefault(r["d_bin"], f"{r['d_lo']:.2f}-{r['d_hi']:.2f}")
    ax.set_xticklabels([lab_by.get(j, str(j)) for j in dbins],
                       rotation=30, fontsize=8)
    ax.set_xlabel("distance from the thick core (plane units)")
    ax.set_title("(b) Capacity x position along the corridor", fontsize=11)
    fig.colorbar(im, ax=ax, label="recovery")

    # (c) freeze
    ax = axes[2]
    gfrz = group("freeze", sub)
    for cap in caps:
        db = sorted({r["d_bin"] for r in sub if r["cap"] == cap})
        v = [gfrz.get((cap, k), (np.nan, 0))[0] for k in db]
        ax.plot(db, v, "s-", color=colors.get(cap, "k"), label=cap)
    ax.set_xlabel("distance from the thick core (bin index)")
    ax.set_ylabel("frozen into the neighbouring basin")
    ax.set_title("(c) Neighbour takeover vs distance", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Peninsula propagation: does only the root survive?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true",
                    help="compare Euclidean and geodesic distance-from-core")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--figure", nargs="?", const="", default=None,
                    help="draw the figure from --out (or from the given CSV)")
    ap.add_argument("--grid", nargs="+", type=int, default=[150, 300, 600])
    ap.add_argument("--caps", nargs="+", default=["w32", "w128", "w512"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--seed0", type=int, default=0,
                    help="seed of the evaluation-cell draw (independent of fits)")
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--t-max", type=float, default=0.03)
    ap.add_argument("--out", default=str(result("geo.csv")))
    a = ap.parse_args()
    if a.validate:
        validate()
    if a.sweep:
        sweep(a)
    if a.figure is not None:
        make_figure(a.figure or a.out)


if __name__ == "__main__":
    main()
