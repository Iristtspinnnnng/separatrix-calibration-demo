"""Figures for the demo: the paper's Fig. 1, and the metric-population trap.

``figure_basins_and_entropy`` reproduces the four-panel idea of
arXiv:2608.14743 Fig. 1 on the Newton fractal: the classifier's basin map and
the MC-dropout entropy field, with the 85th-percentile contour that the
acquisition step would threshold.

``figure_calibration_collapse`` is this repository's own diagnostic. It puts
the in-sample metrics (whose population is re-drawn every round) next to the
same models scored on a fixed independent population, and shows the
training-set composition, the calibration error inside the band, and the
entropy separability. The name is historical: the "collapse" it was built to
show does not happen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from separatrix.classifier import MCClassifier
from separatrix.dataset import make_dataset
from separatrix.refine import run_iterative_refinement, true_boundary_mask
from separatrix.paths import figure

CMAP = ListedColormap(["#3b4cc0", "#e0b000", "#c0392b"])  # 3 basins


def _grid_mesh(xg, yg):
    Xp, Yp = np.meshgrid(xg, yg, indexing="xy")
    return np.column_stack([Xp.ravel(), Yp.ravel()]), Xp, Yp


# ----------------------------------------------------------------------
def figure_basins_and_entropy(X, y, grid, out=figure("fig1_basins_and_entropy.png")):
    xg, yg = grid
    xy, Xp, Yp = _grid_mesh(xg, yg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    # ---- left: basins of attraction (paper Fig.1 top-left) ----------
    # (same backend as the refinement loop: torch/GPU when available)
    clf = MCClassifier(hidden=(256, 256), max_iter=150).fit(X, y)
    print(f"[plotting] classifier: {clf.describe()}, "
          f"{clf.n_epochs_} epochs")
    pred = clf.predict(xy).reshape(Xp.shape)
    ax = axes[0]
    ax.pcolormesh(xg, yg, pred.T, cmap=CMAP, vmin=0.5, vmax=3.5,
                  shading="auto", rasterized=True)
    ax.set_aspect("equal")
    ax.set_title("Basins of attraction\n(classifier prediction, p=3 Newton)",
                 fontsize=12)
    ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")

    # ---- right: MC-dropout entropy = separatrix proxy (Fig.1 top-right)
    _, H = clf.mc_predict(xy, M=12)          # one MC loop -> entropy
    H = H.reshape(Xp.shape)
    ax = axes[1]
    im = ax.pcolormesh(xg, yg, np.log1p(H).T, cmap="inferno",
                       shading="auto", rasterized=True)
    ax.contour(Xp, Yp, H.T, levels=[np.percentile(H, 85)],
               colors="white", linewidths=1.0, alpha=0.8)
    fig.colorbar(im, ax=ax, label="log(1 + entropy H)")
    ax.set_aspect("equal")
    ax.set_title("High-entropy region = approximate separatrix\n"
                 "(MC-dropout uncertainty, 85th-percentile contour)",
                 fontsize=12)
    ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")

    fig.suptitle("Reproduction of arXiv:2608.14743, Fig.1  (Newton fractal)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("[plotting] wrote", out)
    return fig


# ----------------------------------------------------------------------
def figure_calibration_collapse(log, out=figure("fig2_calibration_collapse.png")):
    rounds = [d["round"] for d in log]
    bacc = [d["boundary_acc"] for d in log]
    sep = [d["entropy_separability"] for d in log]
    bfrac = [100 * d.get("boundary_fraction", np.nan) for d in log]
    ece = [d.get("expected_calibration_error", np.nan) for d in log]

    fair = "fair_acc_band1" in log[0]
    if fair:
        f_b1 = [d["fair_acc_band1"] for d in log]
        f_b8 = [d["fair_acc_band8"] for d in log]
        f_int = [d["fair_acc_interior"] for d in log]
        f_ece = [d["fair_ece_band1"] for d in log]
        f_sep = [d.get("fair_sep_auc_hard", np.nan) for d in log]

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.0))

    # (a) the measurement trap: in-sample metric vs fixed population
    ax = axes[0]
    if fair:
        ax.plot(rounds, f_b1, "o-", color="#c0392b", lw=2,
                label="band-1 accuracy, FIXED population")
        ax.plot(rounds, f_b8, "s-", color="#e67e22", lw=2,
                label="band-8 accuracy, FIXED population")
        ax.plot(rounds, f_int, "^-", color="#3b4cc0", lw=1.6,
                label="interior accuracy, FIXED population")
    ax.plot(rounds, bacc, "o--", color="#7f8c8d", lw=1.8,
            label="boundary acc., in-sample (population shifts)")
    ax.set_xlabel("refinement round")
    ax.set_ylabel("accuracy")
    ax.set_xticks(rounds)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("(a) Same models, two populations: the in-sample\n"
                 "metric falls while the real accuracy rises", fontsize=11)
    ax.grid(alpha=0.3)

    # (b) composition of the training set (the oversampling driver)
    ax = axes[1]
    uni = [100 * (1 - d.get("boundary_fraction", np.nan)) for d in log]
    ax.bar(rounds, uni, color="#3b4cc0", alpha=0.8,
           label="uniform base points")
    ax.bar(rounds, bfrac, bottom=uni, color="#8e44ad", alpha=0.85,
           label="oversampled boundary points")
    ax.set_xlabel("refinement round")
    ax.set_ylabel("share of the training set (%)")
    ax.set_xticks(rounds); ax.set_ylim(0, 100)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("(b) Boundary points crowd out\nuniform coverage", fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # (c) calibration error on the separatrix band
    ax = axes[2]
    if fair:
        ax.plot(rounds, f_ece, "o-", color="#16a085", lw=2,
                label="ECE in band 1 (fixed population)")
    ax.plot(rounds, ece, "s--", color="#7f8c8d", lw=1.6,
            label="in-sample |conf - correct|")
    ax.set_xlabel("refinement round"); ax.set_ylabel("calibration error")
    ax.set_xticks(rounds)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("(c) Calibration inside the band\nimproves, not collapses",
                 fontsize=11)
    ax.grid(alpha=0.3)

    # (d) entropy separability (AUROC)
    ax = axes[3]
    if fair:
        ax.plot(rounds, f_sep, "o-", color="#16a085", lw=2,
                label="band-1 vs the surrounding halo (fixed)")
    ax.plot(rounds, sep, "s--", color="#7f8c8d", lw=1.6,
            label="vs the wide dynamical proxy (in-sample)")
    ax.set_xlabel("refinement round")
    ax.set_ylabel("AUROC of MC entropy")
    ax.set_xticks(rounds); ax.set_ylim(0.5, 1.01)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("(d) Entropy separability:\nthe oracle decides the story",
                 fontsize=11)
    ax.grid(alpha=0.3)

    fig.suptitle("Iterative boundary refinement re-examined: population "
                 "shift, not calibration collapse", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("[plotting] wrote", out)
    return fig


if __name__ == "__main__":
    X, y, grid = make_dataset(n=300)
    fig = figure_basins_and_entropy(X, y, grid)

    log = run_iterative_refinement(X, y, grid[0], grid[1], rounds=5)
    figure_calibration_collapse(log)

    # also save a high-res true-boundary reference for comparison
    xy, Xp, Yp = _grid_mesh(grid[0], grid[1])
    bm = true_boundary_mask(xy).reshape(Xp.shape)
    plt.figure(figsize=(6.2, 6))
    plt.pcolormesh(grid[0], grid[1], bm.T, cmap="gray_r",
                   shading="auto", rasterized=True)
    plt.gca().set_aspect("equal")
    plt.title("Oracle: true separatrix\n(Newton fractal, p=3)")
    plt.xlabel("Re(z)"); plt.ylabel("Im(z)")
    plt.tight_layout()
    plt.savefig(figure("fig0_true_separatrix.png"), dpi=160, bbox_inches="tight")
    print("[plotting] wrote", figure("fig0_true_separatrix.png"))
