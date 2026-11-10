"""analyse_subsample.py
========================================================================
Final analysis of the CLEAN density experiment (results/subsample.csv):
one fixed window (300x300 grid, zoom = 20 island widths), only the number of
training points changes (random subsample of the same grid), 7 densities x 3
capacities x 5 seeds.

Because the window content is identical at every density level, this removes
the confound that spoiled the window-based diagrams (where a wider window at
matched w/h also brought more competing structure).

Outputs
  * seed-mean recovery / freeze vs w/h_eff, per capacity
  * w_c per capacity with a bootstrap CI over seeds
  * a comparison against the window-based run (results/island.csv) at matched
    w/h: the gap between them is the "window content" effect
  * fig13_density.png

Run: python analyse_subsample.py results/subsample.csv [results/island.csv]
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
                              ci, crossing, curve, load_phase_csv,
                              load_simple_csv)

def main(sub_path, isl_path=None):
    rows = load_simple_csv(sub_path)
    print(f"{len(rows)} rows, {len({(r['cap'], r['n_train']) for r in rows})} "
          f"configs, seeds {sorted({r['seed'] for r in rows})}")
    print(f"w/h values: {sorted({round(r['w_over_h'], 2) for r in rows})}")

    print("\nseed-mean recovery / freeze by w/h (fixed window content)")
    hdr = f"{'cap':>5} " + " ".join(f"{w:>13}" for w in
                                    sorted({round(r['w_over_h'], 2) for r in rows}))
    print(hdr)
    wcs = {}
    for cap in CAPS:
        ux, mean, std = curve(rows, "recovery", cap)
        _, frz, _ = curve(rows, "freeze", cap)
        print(f"{cap:>5} " + " ".join(f"{m:5.2f}±{s:4.2f}"
                                      for m, s in zip(mean, std)))
        print(f"{'':>5} " + " ".join(f"{f:>13.2f}" for f in frz) + "   <- freeze")
        boot = bootstrap_wc(rows, cap, n_boot=2000)
        wc = crossing(ux, mean)
        wcs[cap] = (float(np.mean([r["params"] for r in rows
                                   if r["cap"] == cap])), wc, ci(boot))
        c = wcs[cap][2]
        print(f"{'':>5} w_c = {wc:.2f}  [95% CI {c[0]:.2f}, {c[1]:.2f}]\n")

    ps = np.array([wcs[c][0] for c in CAPS])
    ws = np.array([wcs[c][1] for c in CAPS])
    ok = np.isfinite(ws)
    if ok.sum() >= 2:
        slope, intercept = np.polyfit(np.log(ps[ok]), np.log(ws[ok]), 1)
        print(f"capacity law: w_c = {np.exp(intercept):.2f} * params^{slope:+.3f}"
              f"   (1.3k -> 266k: {ws[0]/ws[-1]:.2f}x)")

    # ---------------- comparison with the window-based run -------------
    comp = None
    if isl_path:
        isl = load_phase_csv(isl_path)
        print("\nsame w/h, different window content: subsample vs window run")
        comp = {}
        for cap in CAPS:
            ux_s, m_s, _ = curve(rows, "recovery", cap)
            entry = {"sub": (ux_s, m_s)}
            for win in sorted({r["window"] for r in isl}):
                sub = [r for r in isl if r["cap"] == cap and r["window"] == win]
                if not sub:
                    continue
                ux, m, _ = curve(sub, "recovery", cap)
                entry[f"win{win}"] = (ux, m)
            comp[cap] = entry
            # compare by interpolating the window curves at the subsample
            # grid points (their w/h grids do not coincide exactly)
            line = []
            for key, (ux, m) in entry.items():
                if key == "sub":
                    continue
                lx, ly = np.log(np.asarray(ux, float)), np.asarray(m, float)
                o = np.argsort(lx)
                lx, ly = lx[o], np.maximum.accumulate(ly[o])
                tgt = np.log(np.asarray(ux_s, float))
                v = np.interp(tgt, lx, ly, left=np.nan, right=np.nan)
                ok2 = np.isfinite(v)
                if ok2.any():
                    gap = np.nanmean(np.asarray(m_s)[ok2] - v[ok2])
                    line.append(f"{key}: {gap:+.2f} (n={ok2.sum()})")
            print(f"  {cap}: recovery(subsample) - recovery(window)  "
                  + "  ".join(line))

    make_figure(rows, wcs, comp)


def make_figure(rows, wcs, comp, out=figure("fig13_density.png")):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) recovery vs w/h at fixed window content
    ax = axes[0, 0]
    for cap in CAPS:
        ux, mean, std = curve(rows, "recovery", cap)
        ax.errorbar(ux, mean, yerr=std, marker=MARKERS[cap], color=COLORS[cap],
                    capsize=3, lw=1.5, label=cap)
        if np.isfinite(wcs[cap][1]):
            ax.axvline(wcs[cap][1], color=COLORS[cap], ls=":", lw=1)
    ax.axhline(0.5, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("w / h_eff  (island width / effective spacing)")
    ax.set_ylabel("island recovery (mean ± std, 5 seeds)")
    ax.set_title("(a) Density only: window content fixed", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) freeze
    ax = axes[0, 1]
    for cap in CAPS:
        ux, frz, std = curve(rows, "freeze", cap)
        ax.errorbar(ux, frz, yerr=std, marker=MARKERS[cap], color=COLORS[cap],
                    capsize=3, lw=1.5, label=cap)
    ax.set_xscale("log")
    ax.set_xlabel("w / h_eff"); ax.set_ylabel("frozen into the neighbour")
    ax.set_title("(b) The failure mode", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) w_c per capacity
    ax = axes[1, 0]
    xs = [wcs[c][0] for c in CAPS]; ys = [wcs[c][1] for c in CAPS]
    elo = [max(0.0, y - wcs[c][2][0]) for c, y in zip(CAPS, ys)]
    ehi = [max(0.0, wcs[c][2][1] - y) for c, y in zip(CAPS, ys)]
    ax.errorbar(xs, ys, yerr=[elo, ehi], marker="o", color="#8e44ad",
                capsize=4, label="w_c (95% CI, clean run)")
    for c, x, y in zip(CAPS, xs, ys):
        ax.annotate(c, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=COLORS[c])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MLP parameters"); ax.set_ylabel("critical w/h")
    ax.set_title("(c) Capacity still moves the boundary", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) content comparison
    ax = axes[1, 1]
    if comp:
        styles = {"win100": ":", "win200": "--", "win300": "-"}
        for cap in CAPS:
            e = comp[cap]
            ux, m = e["sub"]
            ax.plot(ux, m, marker=MARKERS[cap], color=COLORS[cap], lw=2,
                    label=f"{cap}: fixed content")
            for key, (ux2, m2) in e.items():
                if key == "sub":
                    continue
                ax.plot(ux2, m2, styles.get(key, "-"), color=COLORS[cap],
                        lw=0.9, alpha=0.6,
                        label=f"{cap}: {key}")
        ax.axhline(0.5, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("w / h_eff"); ax.set_ylabel("recovery")
        ax.set_title("(d) Window content, not density, was the confound",
                     fontsize=11)
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    else:
        ax.axis("off")

    fig.suptitle("Clean density experiment: 7 densities x 3 capacities x 5 "
                 "seeds, window fixed", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(result("subsample.csv")),
         sys.argv[2] if len(sys.argv) > 2 else
         (str(result("island.csv")) if __import__("os").path.exists(str(result("island.csv")))
          else None))
