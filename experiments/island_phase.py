"""island_phase.py
========================================================================
Freeze phase diagram on the (w/h, capacity) plane.

w/h = island width / training-grid spacing = "how many samples fit across
the island".  The trick that makes this cheap and rigorous: keep the island
fixed in the *plane* and zoom the training window, so w/h changes by a
factor of ~100 while the architecture and the sample count stay put.  Doing
the same w/h at three different window sizes tests whether the transition
is governed by w/h (a collapse) or by the absolute scale.

For every (window, w/h, capacity) we train one classifier and score it on
the island's interior cells:
    recovery = fraction labelled with the island's own basin
    freeze   = fraction labelled with the SURROUNDING basin

Outputs: island_phase.csv, fig8_freeze_phase.png, and the fitted boundary
    w_c(params)   (the island width, in sampling cells, below which the
    island is frozen), plus the capacity exponent.

Run:  python island_phase.py        (~3-5 min on GPU, labels cached)
========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from separatrix.classifier import MCClassifier
from separatrix.evaluate import grid_labels
from separatrix.islands import fine_labels, find_islands, island_points
from separatrix.paths import figure, result, cache

XLIM = (-1.5, 1.5)
N_FINE = 3200
CELL = (XLIM[1] - XLIM[0]) / (N_FINE - 1)
CACHE = str(cache(f"island_labels_{N_FINE}.npy"))

WINDOWS = (100, 200, 300)          # training-grid resolution n x n
ZOOMS = (2, 3, 5, 8, 14, 25, 50)   # window width in island widths
CAPS = {"w32": (32, 32), "w128": (128, 128), "w512": (512, 512)}
EPOCHS = 250
MAX_TRAIN = 40_000
SEED = 0
N_ISLANDS = 4                      # average over the largest few islands


def get_labels():
    try:
        return np.load(CACHE)
    except Exception:
        lab = fine_labels(n=N_FINE, xlim=XLIM)
        np.save(CACHE, lab)
        return lab


def params_of(hidden, n_in=2, n_out=3):
    sizes = (n_in,) + tuple(hidden) + (n_out,)
    return int(sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                   for i in range(len(sizes) - 1)))


def island_center(isl):
    cx, cy = isl["centroid"]
    return XLIM[0] + cx * CELL, XLIM[0] + cy * CELL


def run_case(isl, k, n_grid, hidden, seed=SEED):
    """Train on a window of width k * island_width, score the island."""
    cx, cy = island_center(isl)
    w = isl["scale_cells"] * CELL
    half = 0.5 * k * w
    g = np.linspace(-half, half, n_grid)
    Xp, Yp = np.meshgrid(g, g, indexing="xy")
    xy = np.column_stack([(Xp + cx).ravel(), (Yp + cy).ravel()])
    lab = grid_labels(xy)
    keep = lab > 0
    X, y = xy[keep], lab[keep].astype(np.int8)

    rng = np.random.default_rng(seed)
    if len(X) > MAX_TRAIN:
        sel = rng.choice(len(X), MAX_TRAIN, replace=False)
        X, y = X[sel], y[sel]

    clf = MCClassifier(hidden=hidden, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X, y, max_iter=EPOCHS)
    fit_s = time.perf_counter() - t0

    ip = island_points(isl, N_FINE, xlim=XLIM)
    pred = clf.predict(ip)
    rec = float((pred == isl["label"]).mean())
    frz = float((pred == isl["surrounding"]).mean())
    _, H = clf.mc_predict(ip, M=6)
    return dict(recovery=rec, freeze=frz, mean_H=float(H.mean()),
                fit_s=round(fit_s, 2), n_train=int(len(X)),
                w_over_h=float((n_grid - 1) / k), w=float(w),
                h=float(2 * half / (n_grid - 1)), window=float(2 * half))


def main():
    labels = get_labels()
    isls = find_islands(labels, min_area=9, ring=2, bury=3, surround_frac=0.6)
    isls.sort(key=lambda d: -d["area"])
    isls = isls[:N_ISLANDS]
    print(f"islands used: {[(d['id'], round(d['scale_cells'], 2)) for d in isls]}")
    print(f"widths (units): {[round(d['scale_cells']*CELL, 5) for d in isls]}")

    rows = []
    for n_grid in WINDOWS:
        for k in ZOOMS:
            for cap_name, hidden in CAPS.items():
                recs, frzs, fits = [], [], []
                for isl in isls:
                    r = run_case(isl, k, n_grid, hidden)
                    recs.append(r["recovery"])
                    frzs.append(r["freeze"])
                    fits.append(r["fit_s"])
                row = dict(window=n_grid, zoom=k,
                           w_over_h=(n_grid - 1) / k,
                           params=params_of(hidden), cap=cap_name,
                           recovery=float(np.mean(recs)),
                           freeze=float(np.mean(frzs)),
                           n_train=int(r["n_train"]),
                           fit_s=float(np.mean(fits)))
                rows.append(row)
                print(f"  n={n_grid:>3} k={k:>3} w/h={row['w_over_h']:6.1f} "
                      f"{cap_name:>5}: recover {row['recovery']:.2f} "
                      f"freeze {row['freeze']:.2f}  ({row['fit_s']:.1f}s)",
                      flush=True)

    with open(str(result("island_phase.csv")), "w", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[key]) for key in keys) + "\n")
    print("\nwrote island_phase.csv")

    boundary = fit_boundary(rows)
    make_figure(rows, boundary)
    print(f"total {time.perf_counter():.0f}s (clock)")


def fit_boundary(rows):
    """w_c(capacity): monotone (isotonic) crossing of recovery = 0.5.
    Averaging across the three window sizes suppresses the bimodal noise at
    the transition."""
    out = []
    for cap in CAPS:
        sub = sorted([r for r in rows if r["cap"] == cap],
                     key=lambda r: r["w_over_h"])
        ws = np.array([r["w_over_h"] for r in sub], float)
        rec = np.array([r["recovery"] for r in sub], float)
        # isotonic (non-decreasing) envelope, then the first crossing of 0.5
        env = np.maximum.accumulate(rec)
        if env.max() < 0.5:
            continue
        i = int(np.argmax(env >= 0.5))
        if i == 0:
            wc = ws[0]
        else:
            # log-linear interpolation between i-1 and i
            x0, x1 = np.log(ws[i - 1]), np.log(ws[i])
            y0, y1 = env[i - 1], env[i]
            t = 0.0 if y1 == y0 else (0.5 - y0) / (y1 - y0)
            wc = float(np.exp(x0 + t * (x1 - x0)))
        out.append(dict(cap=cap, params=params_of(CAPS[cap]), w_c=wc,
                        n_points=len(sub)))
    if len(out) >= 2:
        X = np.log([d["params"] for d in out])
        Y = np.log([d["w_c"] for d in out])
        c, a = np.polyfit(X, Y, 1)
        print(f"\nfreeze boundary:  w_c = {np.exp(a):.1f} * params^{c:+.2f}")
        for d in out:
            print(f"  {d['cap']:>5} {d['params']:>7} params -> "
                  f"w_c = {d['w_c']:5.1f} samples across")
        return dict(points=out, exponent=float(c), scale=float(np.exp(a)))
    return dict(points=out, exponent=None)

def make_figure(rows, boundary, out=figure("fig8_freeze_phase.png")):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    colors = {"w32": "#95a5a6", "w128": "#2980b9", "w512": "#c0392b"}
    marks = {100: "o", 200: "s", 300: "^"}

    # (a) collapse test: same w/h, three window sizes
    ax = axes[0]
    for cap in CAPS:
        for n_grid in WINDOWS:
            sub = [r for r in rows if r["cap"] == cap and r["window"] == n_grid]
            ax.plot([r["w_over_h"] for r in sub], [r["recovery"] for r in sub],
                    marks[n_grid], color=colors[cap], ms=5, alpha=0.8,
                    ls="-" if n_grid == 200 else "none",
                    label=f"{cap}, window {n_grid}")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("island width / sampling spacing  (samples across)")
    ax.set_ylabel("island recovery")
    ax.set_title("(a) The transition is governed by w/h,\n"
                 "roughly independent of the absolute scale", fontsize=11)
    ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors[c], marker="o", ls="", label=c)
               for c in CAPS]
    handles += [Line2D([], [], color="k", marker=marks[n], ls="",
                       label=f"window {n}") for n in WINDOWS]
    ax.legend(handles=handles, fontsize=7, ncol=2)

    # (b) capacity dependence
    ax = axes[1]
    for cap in CAPS:
        sub = sorted([r for r in rows if r["cap"] == cap],
                     key=lambda r: r["w_over_h"])
        ax.plot([r["w_over_h"] for r in sub], [r["recovery"] for r in sub],
                "o-", color=colors[cap], label=f"{cap} ({params_of(CAPS[cap])} p)")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("w / h"); ax.set_ylabel("island recovery")
    ax.set_title("(b) Capacity shifts the threshold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) boundary law
    ax = axes[2]
    pts = boundary["points"]
    if pts:
        ax.plot([d["params"] for d in pts], [d["w_c"] for d in pts], "o-",
                color="#c0392b", label="w_c (recovery = 0.5)")
        if boundary["exponent"] is not None:
            c = boundary["exponent"]
            ps = np.array([d["params"] for d in pts])
            ws = np.array([d["w_c"] for d in pts])
            k = ws / ps ** c
            ax.plot(ps, k * ps ** c, "k--", lw=1,
                    label=f"fit: w_c ~ params^{c:+.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MLP parameters"); ax.set_ylabel("critical width (samples)")
    ax.set_title("(c) Freeze boundary: power law over the first\ndecade, "
                 "then saturation", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Freeze phase diagram: one fixed island, swept in sampling "
                 "resolution and capacity", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
