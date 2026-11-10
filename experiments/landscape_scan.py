"""The constructed testbed: a scale-tunable island, and why it is not used.

Two wells plus a shallow satellite well,

    V(x, y) = (x^2 - 1)^2 / 4 + y^2 / 2 - A exp(-r^2 / (2 sigma^2)),

gives a third attractor whose basin size is set by ``sigma`` -- the cheapest way
to sweep an island's absolute scale with ground truth that is not a fractal.
The labels are real dynamics ("which minimum does the flow reach"), so the
separatrix is a smooth closed curve and scale is isolated as a variable.

It works as a *scale* sweep, but not as a clean island at large ``sigma``: the
satellite's basin grows into a tongue of the surrounding basin (a peninsula),
and for narrow wells the basin assignment becomes sensitive to the integrator
step (:func:`separatrix`-independent; see ``peninsula_v2.py`` for the variant
that puts the well on the separatrix).  This is why the island and peninsula
experiments are run on the paper's own fractal instead, where the ground truth
is exact and free.

Run
---
    python experiments/landscape_scan.py        # ~5 s, writes fig7
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from island_landscape import island_component, label_grid
from separatrix.paths import figure

SIGMAS = (0.30, 0.20, 0.12, 0.07, 0.04, 0.02)
N = 300
STEPS = 1600
CMAP = ListedColormap(["#333333", "#3b4cc0", "#e0b000", "#c0392b"])


def main():
    rows = []
    grids = {}
    for sigma in SIGMAS:
        lab, gx, gy = label_grid(N, sx=sigma, sy=sigma, n_steps=STEPS)
        comp = island_component(lab)
        area = int((lab == 3).sum())
        r_eq = float(np.sqrt(area * (3.2 / (N - 1)) * (2.4 / (N - 1)) /
                             np.pi)) if area else 0.0
        rows.append(dict(sigma=sigma, area=area, r_eq=r_eq,
                         enclosed=comp is not None,
                         enclosed_area=0 if comp is None else comp["area"]))
        grids[sigma] = (lab, gx, gy)
        print(f"sigma={sigma:5.2f}  basin-3 area={area:6d} cells  "
              f"r_eq={r_eq:8.5f} units  "
              f"enclosed island: {'yes' if comp else 'no'}")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3)

    # ---- the morphology at three scales ------------------------------
    for k, sigma in enumerate((0.30, 0.20, 0.12)):
        ax = fig.add_subplot(gs[0, k])
        lab, gx, gy = grids[sigma]
        ax.imshow(lab, origin="lower", cmap=CMAP, vmin=-0.5, vmax=3.5,
                  extent=(gx[0], gx[-1], gy[0], gy[-1]), interpolation="nearest")
        ax.set_title(f"sigma = {sigma:.2f}   (basin 3 = red)\n"
                     f"third-basin area {rows[SIGMAS.index(sigma)]['area']} cells",
                     fontsize=10)
        ax.set_xlabel("x")
        if k == 0:
            ax.set_ylabel("y")

    # ---- area vs sigma ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    s = np.array([r["sigma"] for r in rows])
    a = np.array([r["area"] for r in rows])
    ax.loglog(s, a, "o-", color="#c0392b", label="third-basin area (cells)")
    enc = np.array([r["enclosed_area"] for r in rows])
    ax.loglog(s, np.where(enc > 0, enc, np.nan), "s--", color="#2980b9",
              label="enclosed component only")
    ax.set_xlabel("satellite width sigma (units)")
    ax.set_ylabel("area (grid cells)")
    ax.set_title("(d) The satellite basin grows with sigma", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # ---- equivalent radius vs sigma ---------------------------------
    ax = fig.add_subplot(gs[1, 1])
    r = np.array([r["r_eq"] for r in rows])
    ax.loglog(s, r, "o-", color="#c0392b")
    for n_grid, ls in ((200, ":"), (400, "--")):
        cell = 3.2 / (n_grid - 1)
        ax.axhline(cell, color="#7f8c8d", ls=ls, lw=0.8)
        ax.annotate(f"1 cell @ {n_grid}$^2$", (s[0], cell), fontsize=7,
                    color="#7f8c8d", va="bottom")
    ax.set_xlabel("satellite width sigma (units)")
    ax.set_ylabel("equivalent radius (units)")
    ax.set_title("(e) Scale is tunable over one decade", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    # ---- why this is a dead end -------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join([
        "Why this is not the island testbed:",
        "",
        "* large sigma: the satellite owns a tongue",
        "  of its neighbour, not a closed island;",
        "* small sigma: the tongue is a slow channel",
        "  and the basin label depends on the",
        "  integrator step (arc-length descent with",
        "  min(dt, |grad|) is stable, but the",
        "  assignment still drifts);",
        "* the usable range is about one decade,",
        "  and the boundary is smooth, so there is",
        "  no resolution threshold to find.",
        "",
        "The island and peninsula experiments are",
        "therefore run on the paper's own Newton",
        "fractal (islands.py, peninsula.py), where",
        "the ground truth is exact and free.",
    ]), va="top", ha="left", fontsize=9.5, family="monospace")

    fig.suptitle("Constructed testbed: a scale-tunable third basin, and its "
                 "limits", fontsize=13)
    fig.tight_layout()
    out = figure("fig7_island_landscape.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
