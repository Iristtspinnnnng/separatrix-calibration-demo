"""analyse_island_server.py
========================================================================
Final analysis of the server run (results/island.csv):
81 configurations (3 windows x 9 zooms x 3 capacities) x 5 seeds.

Produces
  * seed-mean recovery vs w/h per capacity, with std over seeds
  * the freeze boundary w_c(capacity) from a monotone-envelope crossing,
    with a bootstrap CI over seeds
  * the capacity exponent of w_c
  * a collapse test: w_c per (capacity, window) -- is the transition
    governed by w/h alone, or does the absolute window matter?
  * the sampling-density effect at matched w/h (interpolated curves)
  * fig12_island_phase_final.png

Run: python analyse_island_server.py results/island.csv
========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from separatrix.paths import figure, result
from separatrix.stats import (CAPS, COLORS, MARKERS, bootstrap_wc,
                              ci, crossing, curve, interp_envelope,
                              load_phase_csv)

def main(path):
    rows = load_phase_csv(path)
    print(f"{len(rows)} rows, seeds {sorted({r['seed'] for r in rows})}, "
          f"windows {sorted({r['window'] for r in rows})}, "
          f"{len({(r['cap'], r['window'], r['zoom']) for r in rows})} configs")

    # ---------------- w_c per capacity (pooled over windows) ----------
    print("\nfreeze boundary: w_c where seed-mean recovery = 0.5")
    print(f"{'cap':>5} {'params':>8} {'w_c':>6} {'95% CI (seeds)':>20}")
    wcs = {}
    for cap in CAPS:
        params = int([r["params"] for r in rows if r["cap"] == cap][0])
        ux, mean, _ = curve(rows, "recovery", cap)
        wc = crossing(ux, mean)
        c = ci(bootstrap_wc(rows, cap))
        wcs[cap] = (params, wc, c)
        print(f"{cap:>5} {params:>8} {wc:>6.2f}   [{c[0]:.2f}, {c[1]:.2f}]")

    ps = np.array([wcs[c][0] for c in CAPS], float)
    ws = np.array([wcs[c][1] for c in CAPS], float)
    ok = np.isfinite(ws)
    slope, intercept = np.polyfit(np.log(ps[ok]), np.log(ws[ok]), 1)
    print(f"\npower law: w_c = {np.exp(intercept):.2f} * params^{slope:+.3f}")
    print(f"  capacity gain 1.3k -> 266k params: {ws[0]/ws[-1]:.2f}x  "
          f"(w_c moves {ws[0]:.2f} -> {ws[-1]:.2f})")

    # ---------------- collapse test: w_c per (cap, window) ------------
    windows = sorted({r["window"] for r in rows})
    print("\ncollapse test: w_c per (capacity, window)  [bootstrap 95% CI]")
    print(f"{'cap':>5} " + " ".join(f"{'win'+str(w):>20}" for w in windows))
    detail = {}
    for cap in CAPS:
        cells = []
        for w in windows:
            ux, mean, _ = curve(rows, "recovery", cap, window=w)
            wc = crossing(ux, mean)
            c = ci(bootstrap_wc(rows, cap, window=w))
            detail[(cap, w)] = (wc, c)
            cells.append((wc, c))
        print(f"{cap:>5} " + " ".join(
            f"{wc:7.2f} [{c[0]:5.2f},{c[1]:5.2f}]" for wc, c in cells))

    # ---------------- density effect at matched w/h -------------------
    targets = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0])
    print("\nsampling density: seed-mean recovery at matched w/h")
    print("  w/h targets: " + " ".join(f"{t:g}" for t in targets))
    dens = {}
    for cap in CAPS:
        rows_w = []
        print(f"  {cap}:")
        for w in windows:
            ux, mean, _ = curve(rows, "recovery", cap, window=w)
            v = interp_envelope(ux, mean, targets)
            rows_w.append(v)
            print(f"    window {w:>3}: " + " ".join(f"{val:>6.2f}" for val in v))
        dens[cap] = np.array(rows_w)
        sp = np.nanmax(dens[cap], axis=0) - np.nanmin(dens[cap], axis=0)
        print("    spread  : " + " ".join(
            f"{0.0 if np.isnan(val) else val:>6.2f}" for val in sp))

    make_figure(rows, wcs, slope, intercept, detail, dens, targets, windows)


def make_figure(rows, wcs, slope, intercept, detail, dens, targets, windows,
                out=figure("fig12_island_phase_final.png")):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) seed-mean curves
    ax = axes[0, 0]
    for cap in CAPS:
        ux, mean, std = curve(rows, "recovery", cap)
        ax.errorbar(ux, mean, yerr=std, marker=MARKERS[cap], color=COLORS[cap],
                    capsize=2, lw=1.3, ms=3.5,
                    label=f"{cap} ({int(wcs[cap][0])} p)")
        if np.isfinite(wcs[cap][1]):
            ax.axvline(wcs[cap][1], color=COLORS[cap], ls=":", lw=1)
    ax.axhline(0.5, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("w / h_eff  (island width / effective spacing)")
    ax.set_ylabel("island recovery (mean ± std, 5 seeds)")
    ax.set_title("(a) Freeze transition, seed-averaged", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) w_c vs capacity
    ax = axes[0, 1]
    xs, ys, lo, hi = [], [], [], []
    for cap in CAPS:
        p, wc, c = wcs[cap]
        xs.append(p); ys.append(wc)
        lo.append(max(0.0, wc - c[0])); hi.append(max(0.0, c[1] - wc))
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", color="#c0392b", capsize=4,
                label="w_c (bootstrap 95% CI)")
    pp = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 50)
    ax.plot(pp, np.exp(intercept) * pp ** slope, "k--", lw=1,
            label=f"fit: w_c ~ params^{slope:+.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MLP parameters"); ax.set_ylabel("critical w/h")
    ax.set_title("(b) Capacity barely moves the boundary", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) collapse test
    ax = axes[1, 0]
    for cap in CAPS:
        ws_ = [detail[(cap, w)][0] for w in windows]
        cs = [detail[(cap, w)][1] for w in windows]
        lo = [max(0.0, v - c[0]) for v, c in zip(ws_, cs)]
        hi = [max(0.0, c[1] - v) for v, c in zip(ws_, cs)]
        ax.errorbar(windows, ws_, yerr=[lo, hi], marker=MARKERS[cap],
                    color=COLORS[cap], capsize=3, label=cap)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training grid n (window resolution)")
    ax.set_ylabel("w_c (samples across)")
    ax.set_title("(c) Collapse test: w_c is not scale-free", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) density comparison at matched w/h
    ax = axes[1, 1]
    styles = {100: "--", 200: "-", 300: ":"}
    for cap in CAPS:
        for i, w in enumerate(windows):
            ax.plot(targets, dens[cap][i], styles.get(w, "-"),
                    marker=MARKERS[cap], ms=3, color=COLORS[cap],
                    alpha=0.45 + 0.25 * i,
                    label=f"{cap}, window {w}")
    ax.axhline(0.5, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("matched w / h")
    ax.set_ylabel("recovery")
    ax.set_title("(d) Sampling density at matched w/h", fontsize=11)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle("Island freezing: 81 configurations x 5 seeds (server run)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(result("island.csv")))
