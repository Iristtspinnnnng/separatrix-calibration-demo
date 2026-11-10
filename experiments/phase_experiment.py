"""phase_experiment.py
========================================================================
Seed-averaged freeze phase diagram, resumable, server-friendly.

Two families
------------
island     the p=3 Newton fractal's buried islands (islands.py).  The island
           is fixed in the plane and the training window is zoomed, so
           w/h = island width / sampling spacing sweeps 2..150 while the
           architecture and the sample budget stay put.

peninsula  a constructed landscape  V = (x^2-1)^2/4 + y^2/2
                                     - A exp(-r_x^2/2sx^2 - r_y^2/2sy^2)
           restricted to a window around the satellite well.  With
           sx >> sy the satellite's basin is a narrow channel (a peninsula);
           with sx = sy it is a compact blob (the control).  Labelling is
           real dynamics (which minimum the flow reaches); the step is a
           capped gradient step (length min(dt, |grad|)) which is stable for
           narrow wells and converges near minima.

For every configuration the script runs `--seeds` seeds, appends one row per
(family, config, seed) to a CSV **immediately**, and skips rows that already
exist -- so an interrupted run resumes where it stopped.

Usage
-----
  python phase_experiment.py island   --seeds 5 --out results/island.csv
  python phase_experiment.py peninsula --seeds 5 --out results/peninsula.csv
  python phase_experiment.py aggregate results/*.csv --fig fig8_phase.png

Quick local check (small subset):
  python phase_experiment.py island --seeds 2 --windows 200 --zooms 5 25 --out /tmp/t.csv

Runtime on one RTX 4060 (rough): island ~3 s per (config, seed) at
200x200 windows, ~8 s at 300x300; peninsula ~5 s per (config, seed).
========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from separatrix.classifier import MCClassifier
from separatrix.paths import figure, result, cache

XLIM = (-1.5, 1.5)
N_FINE = 3200
CELL = (XLIM[1] - XLIM[0]) / (N_FINE - 1)
CACHE = str(cache(f"island_labels_{N_FINE}.npy"))

FIELDS = ["family", "config", "w_over_h", "w", "h", "params", "cap", "seed",
          "recovery", "freeze", "n_train", "wall_s", "meta"]


# ======================================================================
#  I/O helpers
# ======================================================================
def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["family"], row["config"], int(row["seed"])))
    return done


def append_row(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def params_of(hidden, n_in=2, n_out=3):
    sizes = (n_in,) + tuple(hidden) + (n_out,)
    return int(sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                   for i in range(len(sizes) - 1)))


CAPS = {"w32": (32, 32), "w128": (128, 128), "w512": (512, 512)}


# ======================================================================
#  family: Newton islands (zoom parameterisation)
# ======================================================================
class IslandFamily:
    name = "island"

    def __init__(self, n_islands=4):
        from separatrix.islands import fine_labels, find_islands
        if os.path.exists(CACHE):
            self.labels = np.load(CACHE)
        else:
            self.labels = fine_labels(n=N_FINE, xlim=XLIM)
            np.save(CACHE, self.labels)
        isls = find_islands(self.labels, min_area=9, ring=2, bury=3,
                            surround_frac=0.6)
        isls.sort(key=lambda d: -d["area"])
        self.isls = isls[:n_islands]
        self.widths = [d["scale_cells"] * CELL for d in self.isls]

    def config_id(self, window, zoom, cap):
        return f"win{window}_k{zoom}_{cap}"

    def make(self, window, zoom, cap, seed):
        """Training set + ground truth for one configuration."""
        from separatrix.evaluate import grid_labels
        from separatrix.islands import island_points
        rng = np.random.default_rng(seed)
        cases = []
        for isl in self.isls:
            cx = XLIM[0] + isl["centroid"][0] * CELL
            cy = XLIM[0] + isl["centroid"][1] * CELL
            w = isl["scale_cells"] * CELL
            half = 0.5 * zoom * w
            g = np.linspace(-half, half, window)
            Xp, Yp = np.meshgrid(g, g, indexing="xy")
            xy = np.column_stack([(Xp + cx).ravel(), (Yp + cy).ravel()])
            lab = grid_labels(xy)
            keep = lab > 0
            X, y = xy[keep], lab[keep].astype(np.int8)
            if len(X) > 40_000:
                sel = rng.choice(len(X), 40_000, replace=False)
                X, y = X[sel], y[sel]
            cases.append(dict(X=X, y=y, elong=float(isl["elongation"]),
                              area=int(isl["area"]),
                              gt_xy=island_points(isl, N_FINE, xlim=XLIM),
                              gt_label=isl["label"],
                              gt_surround=isl["surrounding"],
                              w=w, h=2 * half / (window - 1)))
        return cases

    def w_over_h(self, window, zoom):
        return (window - 1) / zoom


# ======================================================================
#  family: constructed potential (peninsula / compact blob)
# ======================================================================
class PotentialFamily:
    name = "peninsula"

    def __init__(self, sx, sy, A=1.0, m=(0.55, 0.0), half=0.4):
        self.sx, self.sy, self.A, self.m, self.half = sx, sy, A, m, half

    # -- dynamics -----------------------------------------------------
    def _grad(self, x, y):
        mx, my = self.m
        rx, ry = x - mx, y - my
        g = self.A * np.exp(-rx * rx / (2 * self.sx ** 2)
                            - ry * ry / (2 * self.sy ** 2))
        return (-((x * x - 1.0) * x + g * rx / self.sx ** 2),
                -(y + g * ry / self.sy ** 2))

    def _V(self, x, y):
        mx, my = self.m
        return ((x * x - 1.0) ** 2 / 4.0 + y * y / 2.0
                - self.A * np.exp(-(x - mx) ** 2 / (2 * self.sx ** 2)
                                  - (y - my) ** 2 / (2 * self.sy ** 2)))

    def _step(self, x, y, dt):
        vx, vy = self._grad(x, y)
        g = np.hypot(vx, vy)
        f = np.minimum(1.0, dt / np.maximum(g, 1e-14))
        return x + f * vx, y + f * vy

    def minima(self):
        """Base wells and the satellite minimum (local search)."""
        from scipy.optimize import minimize
        f = lambda p: self._V(p[0], p[1])
        res = minimize(f, np.array([self.m[0], self.m[1]]), method="Nelder-Mead",
                       options=dict(xatol=1e-11, fatol=1e-15, maxiter=4000))
        return np.array([[-1.0, 0.0], [1.0, 0.0], res.x])

    def labels(self, xy, coarse=2500, dt=0.04, budget=200.0):
        """1 = left well, 2 = right well, 3 = satellite, 0 = unsettled."""
        x = xy[:, 0].copy(); y = xy[:, 1].copy()
        steps = int(min(coarse, budget / dt))
        for _ in range(steps):
            x, y = self._step(x, y, dt)
        # fine pass near the satellite (thin structures need a small step)
        s_min = min(self.sx, self.sy)
        near = np.hypot(x - self.m[0], y - self.m[1]) < 12 * max(self.sx, self.sy)
        if near.any():
            xs, ys = x[near], y[near]
            dtf = 0.25 * s_min
            for _ in range(int(min(8000, budget / dtf))):
                xs, ys = self._step(xs, ys, dtf)
            x[near], y[near] = xs, ys
        mins = self.minima()
        d = np.hypot(x[:, None] - mins[None, :, 0], y[:, None] - mins[None, :, 1])
        lab = d.argmin(axis=1) + 1
        lab[d.min(axis=1) > 2e-2] = 0
        return lab.astype(np.int8)

    # -- ground truth --------------------------------------------------
    def ground_truth(self, n_gt=700):
        from scipy import ndimage
        g = np.linspace(-self.half, self.half, n_gt)
        Xp, Yp = np.meshgrid(g + self.m[0], g + self.m[1], indexing="xy")
        lab = self.labels(np.column_stack([Xp.ravel(), Yp.ravel()])).reshape(n_gt, n_gt)
        mask = lab == 3
        comp, ncomp = ndimage.label(mask, structure=np.ones((3, 3)))
        if ncomp == 0:
            return None
        i0 = int(np.argmin(np.abs(g)))
        k = comp[i0, i0]
        if k == 0:
            sizes = ndimage.sum(mask, comp, index=np.arange(1, ncomp + 1))
            k = int(np.argmax(sizes)) + 1
        cm = comp == k
        if cm.mean() > 0.30:              # whole basin, not a localised feature
            return None
        ys, xs = np.nonzero(cm)
        dx = 2 * self.half / (n_gt - 1)
        span_long = max(np.ptp(xs), np.ptp(ys)) * dx
        w = cm.sum() * dx * dx / max(span_long, 1e-9)     # mean width along axis
        ring = ndimage.binary_dilation(cm, iterations=max(4, n_gt // 60)) & ~cm
        vals, counts = np.unique(lab[ring], return_counts=True)
        surround = int(vals[counts.argmax()])
        return dict(coords=np.column_stack([g[xs] + self.m[0],
                                            g[ys] + self.m[1]]),
                    w=float(w), area=int(cm.sum()),
                    label_int=3, surround=surround)

    def training(self, n_train, seed):
        n = int(round(np.sqrt(n_train)))
        g = np.linspace(-self.half, self.half, n)
        Xp, Yp = np.meshgrid(g + self.m[0], g + self.m[1], indexing="xy")
        xy = np.column_stack([Xp.ravel(), Yp.ravel()])
        y = self.labels(xy)
        keep = y > 0
        return xy[keep], y[keep], (2 * self.half) / (n - 1)


# ======================================================================
#  scoring + one configuration
# ======================================================================
def score(clf, gt_xy, gt_label, gt_surround):
    pred = clf.predict(gt_xy)
    return (float((pred == gt_label).mean()),
            float((pred == gt_surround).mean()))


def run_subsample_cfg(fam, window, zoom, cap, seed, fraction):
    """
    CLEAN density experiment: the window and the island are fixed; only the
    number of training points changes (random subsample of the same grid), so
    the window content is identical at every density level.

    w/h_eff is the *effective* ratio: window width = zoom * w, hence
    h_eff = zoom * w / sqrt(n_train)  =>  w/h_eff = sqrt(n_train) / zoom.
    """
    hidden = CAPS[cap]
    recs, frzs, walls, n_used = [], [], 0.0, 0
    cases = fam.make(window, zoom, cap, seed)
    for case in cases:
        X, y = case["X"], case["y"]
        n_keep = max(200, int(round(fraction * len(X))))
        if n_keep < len(X):
            sel = np.random.default_rng(seed + 977).choice(len(X), n_keep,
                                                           replace=False)
            X, y = X[sel], y[sel]
        n_used += len(X)
        clf = MCClassifier(hidden=hidden, random_state=seed)
        t0 = time.perf_counter()
        clf.fit(X, y, max_iter=250)
        walls += time.perf_counter() - t0
        r, f = score(clf, case["gt_xy"], case["gt_label"], case["gt_surround"])
        recs.append(r); frzs.append(f)
    case = cases[0]
    n_mean = n_used / len(cases)
    return dict(recovery=float(np.mean(recs)), freeze=float(np.mean(frzs)),
                n_train=int(n_mean), w=float(case["w"]), h=float(case["h"]),
                w_over_h=float(np.sqrt(n_mean) / zoom),
                wall_s=round(walls, 2),
                meta=json.dumps(dict(fraction=fraction, islands=len(recs),
                                     h_nominal=case["h"])))


def run_island_cfg(fam, window, zoom, cap, seed, per_island=False):
    """If per_island, also return one row per island (so the elongation
    question can be answered by regression over islands)."""
    hidden = CAPS[cap]
    recs, frzs, walls = [], [], 0.0
    cases = fam.make(window, zoom, cap, seed)
    for case in cases:
        clf = MCClassifier(hidden=hidden, random_state=seed)
        t0 = time.perf_counter()
        clf.fit(case["X"], case["y"], max_iter=250)
        walls += time.perf_counter() - t0
        r, f = score(clf, case["gt_xy"], case["gt_label"], case["gt_surround"])
        recs.append(r); frzs.append(f)
    case = cases[0]
    out = dict(recovery=float(np.mean(recs)), freeze=float(np.mean(frzs)),
               n_train=int(len(case["X"])), w=float(case["w"]),
               h=float(case["h"]), wall_s=round(walls, 2),
               meta=json.dumps(dict(islands=len(recs))))
    if per_island:
        out["_per_island"] = [dict(recovery=r, freeze=f, w=c["w"], h=c["h"],
                                   elong=c["elong"], area=c["area"])
                              for r, f, c in zip(recs, frzs, cases)]
    return out


def run_peninsula_cfg(fam, n_train, cap, seed):
    hidden = CAPS[cap]
    gt = fam.gt
    X, y, h = fam.training(n_train, seed)
    clf = MCClassifier(hidden=hidden, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X, y, max_iter=250)
    wall = time.perf_counter() - t0
    r, f = score(clf, gt["coords"], gt["label_int"], gt["surround"])
    return dict(recovery=r, freeze=f, n_train=int(len(X)), w=float(gt["w"]),
                h=float(h), wall_s=round(wall, 2),
                meta=json.dumps(dict(sx=fam.sx, sy=fam.sy, area=gt["area"])))


# ======================================================================
#  drivers
# ======================================================================
def run_island(args):
    fam = IslandFamily()
    done = load_done(args.out)
    print(f"[island] {len(fam.isls)} islands, widths "
          f"{[round(w, 5) for w in fam.widths]}", flush=True)
    total = len(args.windows) * len(args.zooms) * len(args.caps) * len(args.seeds)
    k = 0
    for window in args.windows:
        for zoom in args.zooms:
            for cap in args.caps:
                cid = fam.config_id(window, zoom, cap)
                for seed in args.seeds:
                    k += 1
                    if (fam.name, cid, seed) in done:
                        continue
                    t0 = time.perf_counter()
                    res = run_island_cfg(fam, window, zoom, cap, seed,
                                         per_island=args.per_island)
                    per = res.pop("_per_island", None)
                    row = dict(family=fam.name, config=cid,
                               w_over_h=fam.w_over_h(window, zoom), cap=cap,
                               params=params_of(CAPS[cap]), seed=seed,
                               **res)
                    append_row(args.out, row)
                    if per:
                        for j, d in enumerate(per):
                            append_row(args.out, dict(
                                family=fam.name,
                                config=f"{cid}_isl{j}",
                                w_over_h=d["w"] / d["h"], w=d["w"], h=d["h"],
                                params=params_of(CAPS[cap]), cap=cap,
                                seed=seed, recovery=d["recovery"],
                                freeze=d["freeze"], n_train=res["n_train"],
                                wall_s=0.0,
                                meta=json.dumps(dict(elong=d["elong"],
                                                     area=d["area"]))))
                    print(f"  [{k}/{total}] {cid} seed={seed} "
                          f"w/h={row['w_over_h']:.1f} rec={row['recovery']:.3f} "
                          f"frz={row['freeze']:.3f} "
                          f"({time.perf_counter()-t0:.1f}s)", flush=True)


def run_subsample(args):
    """Clean density axis: fixed window + fixed island, only n_train changes."""
    fam = IslandFamily()
    done = load_done(args.out)
    print(f"[subsample] windows={args.windows} zooms={args.zooms} "
          f"fractions={args.fractions}", flush=True)
    for window in args.windows:
        for zoom in args.zooms:
            for frac in args.fractions:
                for cap in args.caps:
                    cid = f"sub_win{window}_k{zoom}_f{frac:g}_{cap}"
                    for seed in args.seeds:
                        if (fam.name, cid, seed) in done:
                            continue
                        t0 = time.perf_counter()
                        res = run_subsample_cfg(fam, window, zoom, cap, seed,
                                               frac)
                        row = dict(family="subsample", config=cid, cap=cap,
                                   params=params_of(CAPS[cap]), seed=seed,
                                   **res)
                        append_row(args.out, row)
                        print(f"  {cid} seed={seed} n={row['n_train']:>6} "
                              f"w/h={row['w_over_h']:5.2f} "
                              f"rec={row['recovery']:.3f} "
                              f"frz={row['freeze']:.3f} "
                              f"({time.perf_counter()-t0:.1f}s)", flush=True)


def run_peninsula(args):
    done = load_done(args.out)
    print(f"[peninsula] sigmas={args.sigmas} densities={args.densities}",
          flush=True)
    for sx in args.sigmas:
        sy = sx / args.aspect
        fam = PotentialFamily(sx=sx, sy=sy)
        fam.gt = fam.ground_truth()
        if fam.gt is None:
            print(f"  sx={sx} sy={sy}: no localised feature, skipped", flush=True)
            continue
        print(f"  sx={sx} sy={sy}: width w={fam.gt['w']:.4f} "
              f"area={fam.gt['area']} surround={fam.gt['surround']}", flush=True)
        for n_train in args.densities:
            h = 2 * fam.half / (int(round(np.sqrt(n_train))) - 1)
            for cap in args.caps:
                cid = f"sx{sx}_ar{args.aspect}_{n_train}_{cap}"
                for seed in args.seeds:
                    if (fam.name, cid, seed) in done:
                        continue
                    res = run_peninsula_cfg(fam, n_train, cap, seed)
                    row = dict(family=fam.name, config=cid,
                               w_over_h=fam.gt["w"] / h, cap=cap,
                               params=params_of(CAPS[cap]), seed=seed, **res)
                    append_row(args.out, row)
                    print(f"  {cid} seed={seed} w/h={row['w_over_h']:.2f} "
                          f"rec={row['recovery']:.3f} frz={row['freeze']:.3f} "
                          f"({row['wall_s']:.1f}s)", flush=True)


# ======================================================================
#  aggregation + figure
# ======================================================================
def read_rows(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for key in ("w_over_h", "w", "h", "params", "recovery",
                            "freeze", "n_train", "wall_s"):
                    r[key] = float(r[key])
                r["seed"] = int(r["seed"])
                rows.append(r)
    return rows


def aggregate(rows):
    """group by (family, config) -> mean/std over seeds."""
    groups = {}
    for r in rows:
        key = (r["family"], r["config"])
        groups.setdefault(key, []).append(r)
    out = []
    for (family, config), rs in groups.items():
        rec = np.array([r["recovery"] for r in rs])
        frz = np.array([r["freeze"] for r in rs])
        base = dict(rs[0])
        base.update(n=len(rs), recovery_mean=rec.mean(), recovery_std=rec.std(ddof=1)
                    if len(rs) > 1 else 0.0,
                    freeze_mean=frz.mean(), freeze_std=frz.std(ddof=1)
                    if len(rs) > 1 else 0.0)
        out.append(base)
    return out


def crossing(xs, ys, level=0.5):
    """Smallest x where a monotone envelope of ys reaches level."""
    order = np.argsort(xs)
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    env = np.maximum.accumulate(ys)
    if env.max() < level:
        return np.nan
    i = int(np.argmax(env >= level))
    if i == 0:
        return float(xs[0])
    x0, x1, y0, y1 = np.log(xs[i - 1]), np.log(xs[i]), env[i - 1], env[i]
    t = 0.0 if y1 == y0 else (level - y0) / (y1 - y0)
    return float(np.exp(x0 + t * (x1 - x0)))


def run_aggregate(args):
    rows = read_rows(args.inputs)
    # per-island rows are for the elongation regression, not for w_c
    if not getattr(args, "include_per_island", False):
        n0 = len(rows)
        rows = [r for r in rows if "_isl" not in r["config"]]
        if len(rows) != n0:
            print(f"(ignoring {n0-len(rows)} per-island rows; "
                  f"use --include-per-island to keep them)")
    if not rows:
        print("no rows found"); return
    agg = aggregate(rows)
    print(f"{len(rows)} rows -> {len(agg)} configurations")
    fams = sorted({r["family"] for r in agg})
    fig, axes = plt.subplots(1, 2 if len(fams) > 1 else 1,
                             figsize=(7 * (2 if len(fams) > 1 else 1), 5),
                             squeeze=False)
    colors = {"w32": "#95a5a6", "w128": "#2980b9", "w512": "#c0392b"}
    for ax, fam in zip(axes[0], fams):
        sub = [r for r in agg if r["family"] == fam]
        for cap in sorted({r["cap"] for r in sub}):
            pts = sorted([r for r in sub if r["cap"] == cap],
                         key=lambda r: r["w_over_h"])
            x = [r["w_over_h"] for r in pts]
            m = [r["recovery_mean"] for r in pts]
            s = [r["recovery_std"] for r in pts]
            ax.errorbar(x, m, yerr=s, marker="o", color=colors.get(cap, "k"),
                        capsize=3, lw=1.4, ms=5, label=cap)
            wc = crossing(x, m)
            if not np.isnan(wc):
                ax.axvline(wc, color=colors.get(cap, "k"), ls=":", lw=1, alpha=0.7)
        ax.axhline(0.5, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("w / h  (feature width / sampling spacing)")
        ax.set_ylabel("recovery (mean ± std over seeds)")
        ax.set_title(f"{fam}: freeze transition", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Seed-averaged freeze phase diagram", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=150, bbox_inches="tight")
    print(f"wrote {args.fig}")

    # boundary summary
    for fam in fams:
        print(f"\n{fam}: w_c (recovery mean = 0.5)")
        for cap in sorted({r["cap"] for r in agg if r["family"] == fam}):
            pts = sorted([r for r in agg if r["family"] == fam and r["cap"] == cap],
                         key=lambda r: r["w_over_h"])
            wc = crossing([r["w_over_h"] for r in pts],
                          [r["recovery_mean"] for r in pts])
            n = max(r["n"] for r in pts)
            print(f"  {cap:>5} params={pts[0]['params']:>7.0f}  "
                  f"w_c={wc:6.2f}  (n={n} seeds)")


# ======================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("island")
    pi.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    pi.add_argument("--windows", nargs="+", type=int, default=[100, 200, 300])
    pi.add_argument("--zooms", nargs="+", type=int,
                    default=[2, 3, 5, 8, 14, 25, 50, 70, 100])
    pi.add_argument("--caps", nargs="+", default=["w32", "w128", "w512"])
    pi.add_argument("--out", default=str(result("island.csv")))
    pi.add_argument("--per-island", action="store_true",
                    help="also write one row per island (elongation analysis)")

    pp = sub.add_parser("peninsula")
    pp.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    pp.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.04, 0.02, 0.01, 0.005])
    pp.add_argument("--aspect", type=float, default=6.0,
                    help="sx/sy: >1 gives a narrow channel (peninsula)")
    pp.add_argument("--densities", nargs="+", type=int,
                    default=[10000, 40000, 90000])
    pp.add_argument("--caps", nargs="+", default=["w32", "w128", "w512"])
    pp.add_argument("--out", default=str(result("peninsula.csv")))

    ps = sub.add_parser("subsample")
    ps.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ps.add_argument("--windows", nargs="+", type=int, default=[300])
    ps.add_argument("--zooms", nargs="+", type=int, default=[7])
    ps.add_argument("--fractions", nargs="+", type=float,
                    default=[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    ps.add_argument("--caps", nargs="+", default=["w32", "w128", "w512"])
    ps.add_argument("--out", default=str(result("subsample.csv")))

    pa = sub.add_parser("aggregate")
    pa.add_argument("inputs", nargs="+")
    pa.add_argument("--fig", default=figure("fig8_phase.png"))
    pa.add_argument("--include-per-island", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "island":
        run_island(args)
    elif args.cmd == "subsample":
        run_subsample(args)
    elif args.cmd == "peninsula":
        run_peninsula(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
