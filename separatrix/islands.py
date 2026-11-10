"""Buried islands, freeze-in, and the acquisition blind spot.

The Newton map of ``z**3 - 1`` has a fractal separatrix, and *buried* pieces
of one basin can sit inside another basin ("islands").  A uniform grid can
miss them entirely, a smooth classifier will label them like their
surroundings, and an entropy-driven sampler can only visit them if a sample
ever lands there.

This module builds the ground truth for that experiment:

1. :func:`fine_labels` -- vectorised Newton labelling of an ``n x n`` grid:
   ``1..p`` = basin of that root, ``0`` = escaped, ``-1`` = not converged
   within ``n_max`` steps (the Julia set itself, numerically of measure zero).
   The implementation is :func:`separatrix.dynamics.basin_image`; the
   :func:`cached_fine_labels` wrapper keeps the 3200x3200 field on disk
   because every island, peninsula and protocol experiment needs it.

2. :func:`find_islands` -- connected components of a basin label that are
   (a) big enough to be more than pixel noise, (b) *enclosed*: the ring just
   outside them is dominated by a DIFFERENT basin, and (c) buried: far from
   the main body of their own basin.

3. :func:`island_metrics` -- per island, given a classifier: the fraction of
   its interior labelled correctly ("recovery"), labelled as the surrounding
   basin ("freeze"), or as a third basin; plus the island's entropy rank
   within the sampler's own reference distribution.

"Freeze rate" is the failure mode of interest: the island is frozen into its
neighbourhood, i.e. the model asserts a confident but wrong basin there.
"""
import numpy as np
from scipy import ndimage

from .dynamics import basin_image
from .paths import cache

__all__ = ["fine_labels", "cached_fine_labels", "find_islands",
           "describe_islands", "eval_grid_full", "island_points",
           "island_metrics", "neighbour_pressure"]


# ----------------------------------------------------------------------
def fine_labels(n=1600, xlim=(-1.5, 1.5), p=3, n_max=40, tol=1e-6,
                escape=1e3):
    """Basin labels on an ``n x n`` grid: the ground truth of this repository.

    Thin wrapper around :func:`separatrix.dynamics.basin_image`, which holds
    the vectorised Newton iteration.  ``1..p`` are the basins, ``0`` marks
    orbits that escaped, ``-1`` orbits still unconverged after ``n_max``
    steps (numerically: the Julia set).
    """
    return basin_image(n=n, xlim=xlim, p=p, n_max=n_max, tol=tol,
                       escape=escape)


def cached_fine_labels(n=3200, xlim=(-1.5, 1.5), p=3):
    """Basin image at ``n x n``, cached in ``.cache/``.

    The 3200x3200 field takes a few seconds to build and is shared by every
    island, peninsula and protocol experiment; caching it keeps those runs
    cheap and the island catalogue reproducible.
    """
    path = cache(f"island_labels_{n}.npy")
    if path.exists():
        return np.load(path)
    lab = basin_image(n=n, xlim=xlim, p=p)
    np.save(path, lab)
    return lab


# ----------------------------------------------------------------------
def find_islands(labels, min_area=25, ring=3, bury=6, surround_frac=0.8,
                 labels_to_track=(1, 2, 3)):
    """
    Connected components of each basin that are enclosed by another basin.

    min_area      : ignore components smaller than this many cells
    ring          : width (cells) of the ring used to read the surroundings
    bury          : component must be at least this far (cells) from the main
                    body of its own basin, so boundary speckle is excluded
    surround_frac : fraction of the ring that must agree on the surrounding
                    label for the component to count as enclosed
    """
    n = labels.shape[0]
    out = []
    objs = None
    for lab in labels_to_track:
        mask = labels == lab
        comp, ncomp = ndimage.label(mask)          # 4-connectivity
        if ncomp == 0:
            continue
        sizes = ndimage.sum(mask, comp, index=np.arange(1, ncomp + 1))
        # "main body" = all big components of this basin merged, so that
        # bury distance is measured to the bulk, not to another island
        big = (np.arange(1, ncomp + 1))[sizes >= max(0.02 * mask.sum(),
                                                     10 * min_area)]
        main = np.isin(comp, big)
        dist_to_main = (ndimage.distance_transform_edt(~main)
                        if main.any() else np.zeros_like(labels, float))

        struct = ndimage.generate_binary_structure(2, 2)
        objs = ndimage.find_objects(comp)
        pad = ring + 1
        for k in range(1, ncomp + 1):
            area = int(sizes[k - 1])
            if area < min_area or objs[k - 1] is None:
                continue
            sl = objs[k - 1]
            y0 = max(0, sl[0].start - pad); y1 = min(n, sl[0].stop + pad)
            x0 = max(0, sl[1].start - pad); x1 = min(n, sl[1].stop + pad)

            win = labels[y0:y1, x0:x1]
            cm_w = comp[y0:y1, x0:x1] == k
            dist_w = dist_to_main[y0:y1, x0:x1]
            ys, xs = np.nonzero(cm_w)
            if dist_w[cm_w].min() < bury:
                continue                            # part of the main body
            ring_m = ndimage.binary_dilation(
                cm_w, structure=struct, iterations=ring) & ~cm_w
            if ring_m.sum() == 0:
                continue
            vals, counts = np.unique(win[ring_m], return_counts=True)
            top = vals[counts.argmax()]
            if top == lab or counts.max() / counts.sum() < surround_frac:
                continue                            # not enclosed by one basin
            interior = ndimage.binary_erosion(cm_w, structure=struct,
                                              iterations=1)
            if interior.sum() < 5:
                interior = cm_w
            iy_w, ix_w = np.nonzero(interior)
            ys_full, xs_full = ys + y0, xs + x0
            # elongation from the second moments of the component
            cov = np.cov(np.vstack([xs_full, ys_full])) if area > 2 else np.eye(2)
            ev = np.linalg.eigvalsh(cov)
            elong = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
            out.append(dict(id=len(out), label=int(lab),
                            surrounding=int(top),
                            area=area, scale_cells=float(np.sqrt(area)),
                            centroid=(float(xs_full.mean()),
                                      float(ys_full.mean())),
                            bbox=(int(ys_full.min()), int(ys_full.max()),
                                  int(xs_full.min()), int(xs_full.max())),
                            patch_origin=(int(y0), int(x0)),
                            patch_mask=cm_w,
                            elongation=elong,
                            exterior_fraction=float(1 - counts.max() /
                                                    counts.sum()),
                            dist_to_main=float(dist_w[cm_w].min()),
                            interior_idx=(iy_w + y0, ix_w + x0)))
    return out


# ----------------------------------------------------------------------
def _points_in_patch(isl, grid_mask, n):
    """Does a coarse grid mask have any True inside the island patch?"""
    y0, x0 = isl["patch_origin"]
    patch = isl["patch_mask"]
    sub = grid_mask[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]]
    return (sub & patch).any()


# ----------------------------------------------------------------------
def describe_islands(labels, islands, xlim=(-1.5, 1.5), n_grid=None,
                     mask300=None):
    """Human-readable per-island table (no classifier needed)."""
    n = labels.shape[0]
    cell = (xlim[1] - xlim[0]) / (n - 1)
    rows = []
    for isl in islands:
        x, y = isl["centroid"]
        zx = xlim[0] + x * cell
        zy = xlim[0] + y * cell
        rows.append(dict(id=isl["id"], label=isl["label"],
                         surrounding=isl["surrounding"],
                         area_cells=isl["area"],
                         scale_cells=round(isl["scale_cells"], 2),
                         scale_units=round(isl["scale_cells"] * cell, 5),
                         center=(round(zx, 4), round(zy, 4)),
                         elongation=round(isl["elongation"], 2),
                         buried_by=round(isl["dist_to_main"] * cell, 4),
                         sampled_by_initial_grid=bool(
                             _points_in_patch(isl, mask300, labels.shape[0]))
                         if mask300 is not None else None))
    return rows


# ----------------------------------------------------------------------
def eval_grid_full(clf, xy, mc_samples=12, chunk=250_000):
    """Predictions + MC entropy on a big grid, in chunks."""
    n = len(xy)
    pred = np.empty(n, dtype=np.int8)
    H = np.empty(n, dtype=np.float32)
    conf = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        mean_p, h = clf.mc_predict(xy[sl], M=mc_samples)
        pred[sl] = clf.predict(xy[sl]).astype(np.int8)
        H[sl] = h
        conf[sl] = mean_p.max(axis=1)
    return pred, H, conf


def island_points(isl, n, xlim=(-1.5, 1.5), sample=400, rng=None):
    """Coordinates of (a subsample of) an island's interior cells."""
    iy, ix = isl["interior_idx"]
    if len(iy) == 0:
        return np.zeros((0, 2))
    if rng is not None and len(iy) > sample:
        sel = rng.choice(len(iy), sample, replace=False)
        iy, ix = iy[sel], ix[sel]
    cell = (xlim[1] - xlim[0]) / (n - 1)
    return np.column_stack([xlim[0] + ix * cell, xlim[0] + iy * cell])


def island_metrics(clf, islands, n, xlim=(-1.5, 1.5), H_train=None, thr_H=None,
                   sample=400, seed=0, mc_samples=12):
    """
    Per-island recovery statistics, evaluated directly on the island cells.

    recover_rate : interior cells the model labels as the island's own basin
    freeze_rate  : interior cells the model labels as the SURROUNDING basin
                   (the "frozen into its neighbourhood" failure)
    entropy_pct  : percentile of the island's max entropy within the training
                   set's entropy distribution (the sampler's own reference)
    selected     : does the island exceed the sampler's threshold at all?
    """
    rng = np.random.default_rng(seed)
    out = []
    for isl in islands:
        xy = island_points(isl, n, xlim=xlim, sample=sample, rng=rng)
        if len(xy) == 0:
            continue
        mean_p, H = clf.mc_predict(xy, M=mc_samples)
        p = clf.predict(xy)
        h = np.asarray(H, dtype=float)
        pct = (100.0 * (H_train < h.max()).mean()) if H_train is not None else np.nan
        out.append(dict(id=isl["id"], label=isl["label"],
                        surrounding=isl["surrounding"],
                        scale_cells=isl["scale_cells"], area=isl["area"],
                        elongation=isl["elongation"],
                        centroid=isl["centroid"],
                        n_eval=int(len(h)),
                        recover_rate=float((p == isl["label"]).mean()),
                        freeze_rate=float((p == isl["surrounding"]).mean()),
                        other_rate=float(((p != isl["label"]) &
                                          (p != isl["surrounding"])).mean()),
                        mean_H=float(h.mean()), max_H=float(h.max()),
                        max_entropy_pct=float(pct),
                        selected=bool(thr_H is not None and h.max() >= thr_H)))
    return out


def neighbour_pressure(islands, X, n, xlim=(-1.5, 1.5), radius_scale=3.0):
    """
    H3 proxy: how outnumbered is an island by its surroundings *in the
    training set*?  Counts training points inside the island (n_in) and
    inside an annulus of radius `radius_scale` x island scale (n_near).
    """
    cell = (xlim[1] - xlim[0]) / (n - 1)
    out = []
    for isl in islands:
        cx, cy = isl["centroid"]
        ccx = xlim[0] + cx * cell
        ccy = xlim[0] + cy * cell
        r = radius_scale * isl["scale_cells"] * cell
        n_out = int((np.hypot(X[:, 0] - ccx, X[:, 1] - ccy) <= r).sum())
        n_in = 0
        y0, x0 = isl["patch_origin"]
        patch = isl["patch_mask"]
        xs = np.rint((X[:, 0] - xlim[0]) / cell).astype(int)
        ys = np.rint((X[:, 1] - xlim[0]) / cell).astype(int)
        my = ((ys >= y0) & (ys < y0 + patch.shape[0]) &
              (xs >= x0) & (xs < x0 + patch.shape[1]))
        if my.any():
            n_in = int(patch[ys[my] - y0, xs[my] - x0].sum())
        out.append(dict(id=isl["id"], scale_cells=isl["scale_cells"],
                        n_in=n_in, n_near=n_out,
                        pressure=(n_out / n_in) if n_in > 0 else np.inf))
    return out