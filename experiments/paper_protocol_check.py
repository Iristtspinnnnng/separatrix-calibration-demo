"""What does the paper's own protocol actually see of the buried islands?

Protocol, from Sec. 3.1 of arXiv:2608.14743:

* 40 000 uniformly sampled initial conditions (a 200x200 grid), integrated with
  Newton's method and labelled by the root they reach;
* a classifier with MC-dropout entropy on top of them;
* evaluation on a finer 160 000-point (400x400) grid, of which the top **10 %**
  by entropy (16 000 points) are kept as "the separatrix" and used to train the
  score-based generative model;
* the generative step is scored by Chamfer and sliced Wasserstein distance
  *between that 16k set and the SGM's samples*.

That last line is the point of this script: measure how much buried-island
structure ever enters the 16k set, and how much of the island structure the
classifier recovers at those settings.

Measured (see ``results/paper_protocol.json``):

    training points inside islands            4 of 39 986
    islands touched by the training grid      4 of 68
    "separatrix" points inside islands        22 of 16 000  (0.14 %)
    island cells classified correctly         9.1 % of the 60 buried islands

Run
---
    python experiments/paper_protocol_check.py        # ~2 min on a 4060
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np

from separatrix.classifier import MCClassifier
from separatrix.islands import cached_fine_labels, find_islands, island_points
from separatrix.paths import result

XLIM = (-1.5, 1.5)
N_FINE = 3200


def grid(n, p=3):
    """Coordinates and labels of an ``n x n`` uniform grid, from the dynamics."""
    from separatrix.evaluate import grid_labels
    g = np.linspace(*XLIM, n)
    xp, yp = np.meshgrid(g, g, indexing="xy")
    xy = np.column_stack([xp.ravel(), yp.ravel()])
    return xy, grid_labels(xy), g


def cell_index(xy, labels):
    """Fine-grid indices of arbitrary points."""
    cell = (XLIM[1] - XLIM[0]) / (labels.shape[0] - 1)
    return (np.rint((xy[:, 0] - XLIM[0]) / cell).astype(int),
            np.rint((xy[:, 1] - XLIM[0]) / cell).astype(int))


def points_in_islands(xy, islands, labels):
    """Boolean mask: which points fall on an island cell."""
    xs, ys = cell_index(xy, labels)
    n = labels.shape[0]
    ok = (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
    m = np.zeros(len(xy), dtype=bool)
    for isl in islands:
        y0, x0 = isl["patch_origin"]
        patch = isl["patch_mask"]
        my = ((ys >= y0) & (ys < y0 + patch.shape[0]) &
              (xs >= x0) & (xs < x0 + patch.shape[1]))
        if my.any():
            m[my] |= patch[ys[my] - y0, xs[my] - x0]
    return m & ok


def main():
    labels = cached_fine_labels(N_FINE, XLIM)
    islands = find_islands(labels, min_area=9, ring=2, bury=3,
                           surround_frac=0.6)
    islands.sort(key=lambda d: -d["area"])
    buried = [d for d in islands if d["dist_to_main"] >= 10]
    area_frac = 100 * sum(d["area"] for d in islands) / labels.size
    print(f"islands: {len(islands)} total, {len(buried)} buried; "
          f"island area = {area_frac:.4f}% of the plane")

    # ---- training set: the paper's 200x200 grid ------------------------
    xy_tr, lab_tr, _ = grid(200)
    keep = lab_tr > 0
    Xtr, ytr = xy_tr[keep], lab_tr[keep]
    print(f"training set: {len(Xtr)} points (200x200 grid)")

    m_tr = points_in_islands(Xtr, islands, labels)
    # per island: is any training point on it, and how many of its cells are
    # covered
    xs_tr, ys_tr = cell_index(Xtr, labels)
    touched = cells_total = cells_hit = 0
    for isl in islands:
        y0, x0 = isl["patch_origin"]
        patch = isl["patch_mask"]
        my = ((ys_tr >= y0) & (ys_tr < y0 + patch.shape[0]) &
              (xs_tr >= x0) & (xs_tr < x0 + patch.shape[1]))
        hit = np.zeros_like(patch)
        if my.any():
            hit[ys_tr[my] - y0, xs_tr[my] - x0] = True
        touched += int((hit & patch).any())
        cells_total += int(patch.sum())
        cells_hit += int((hit & patch).sum())
    print(f"  training points inside islands: {int(m_tr.sum())}")
    print(f"  islands touched by the training grid: {touched}/{len(islands)}")
    print(f"  island cells covered by a training point: {cells_hit}/"
          f"{cells_total} ({100 * cells_hit / max(1, cells_total):.2f}%)")

    # ---- evaluation grid: 400x400, keep the top 10 % by entropy --------
    clf = MCClassifier(hidden=(256, 256), random_state=0).fit(Xtr, ytr)
    xy_ev, lab_ev, _ = grid(400)
    _, H = clf.mc_predict(xy_ev, M=12)
    sel = H >= np.percentile(H, 90)
    m_ev = points_in_islands(xy_ev, islands, labels)
    n_sel_in = int((sel & m_ev).sum())

    # islands with at least one *selected* point
    xs_ev, ys_ev = cell_index(xy_ev[sel], labels)
    sele_islands = 0
    for isl in islands:
        y0, x0 = isl["patch_origin"]
        patch = isl["patch_mask"]
        my = ((ys_ev >= y0) & (ys_ev < y0 + patch.shape[0]) &
              (xs_ev >= x0) & (xs_ev < x0 + patch.shape[1]))
        if my.any() and patch[ys_ev[my] - y0, xs_ev[my] - x0].any():
            sele_islands += 1

    print(f"evaluation grid 400x400 = {len(xy_ev)} points; "
          f"top 10% by entropy = {int(sel.sum())} points")
    print(f"  evaluation points inside islands: {int(m_ev.sum())} "
          f"({100 * m_ev.mean():.4f}%)")
    print(f"  of the {int(sel.sum())} 'separatrix' (SGM training) points: "
          f"{n_sel_in} inside islands "
          f"({100 * n_sel_in / max(1, int(sel.sum())):.4f}%)")
    print(f"  islands with at least one selected point: "
          f"{sele_islands}/{len(islands)}")

    tot = rec = 0
    for d in buried:
        p = clf.predict(island_points(d, N_FINE, xlim=XLIM))
        tot += len(p)
        rec += int((p == d["label"]).sum())
    print(f"  island cells predicted as their own basin: {rec}/{tot} "
          f"({100 * rec / max(1, tot):.2f}%) on the {len(buried)} buried islands")

    payload = dict(n_islands=len(islands), n_buried=len(buried),
                   island_area_pct=round(area_frac, 4),
                   n_train=int(len(Xtr)),
                   train_points_in_islands=int(m_tr.sum()),
                   train_islands_touched=int(touched),
                   train_cells_covered=int(cells_hit),
                   train_cells_total=int(cells_total),
                   n_eval=int(len(xy_ev)), n_selected=int(sel.sum()),
                   eval_points_in_islands=int(m_ev.sum()),
                   selected_in_islands=n_sel_in,
                   islands_with_selected=sele_islands,
                   island_cells_recovered=rec, island_cells_evaluated=tot,
                   island_recovery_pct=round(100 * rec / max(1, tot), 2))
    with open(result("paper_protocol.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print("wrote", result("paper_protocol.json"))


if __name__ == "__main__":
    main()
