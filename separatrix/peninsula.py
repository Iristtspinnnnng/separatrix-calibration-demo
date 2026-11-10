"""Peninsula geometry: corridor thickness and distance from the thick core.

Why the peninsula experiment runs on the paper's own fractal
------------------------------------------------------------
In a smooth 2-D gradient flow the basin boundary is a smooth curve, so a
third attractor placed *on* the boundary owns a strip and one placed *inside*
a basin owns a buried island -- neither is a finger (see
``experiments/peninsula_v2.py`` and ``experiments/island_landscape.py`` for
the two attempts).  Genuine peninsulas need a folded or fractal boundary,
which the Newton fractal of the paper's benchmark provides, with ground truth
we can compute exactly.

Two coordinates
---------------
For a basin cell ``x``, its local corridor thickness is

    t(x) = 2 * distance(x, nearest cell carrying another basin),

i.e. the width of the corridor that contains it.  Thin corridors (small
``t``) are the fingers that reach from the densely sampled boundary into a
basin; thick cells are basin interiors.  The second coordinate is

    d(x) = distance(x, nearest thick "core" cell of the SAME basin),

which measures how far down a finger a cell sits: ``d`` small = root (close
to the dense main boundary), ``d`` large = tip.

Both are *Euclidean* distances on the 3200x3200 label raster.  For winding
corridors the Euclidean value is a lower bound on the geodesic one;
:func:`geodesic_core_distance` computes a local BFS reference, and
``experiments/peninsula_geo.py --validate`` reports the agreement (median
absolute difference 0.0009 units, 90th percentile 0.0058, i.e. 1-6 fine
cells), which is why only bins of width >= 0.02 are used.
"""

import numpy as np
from scipy import ndimage

from .islands import cached_fine_labels

__all__ = ["XLIM", "N_FINE", "CELL", "CAPS", "THICKNESS_BINS",
           "DISTANCE_BINS", "T_THIN", "T_CORE", "thickness", "bin_id",
           "neighbour_label", "neighbour_of", "core_distance",
           "geodesic_core_distance", "prepare_thickness", "prepare_core",
           "raster_train_grid"]

XLIM = (-1.5, 1.5)
N_FINE = 3200
CELL = (XLIM[1] - XLIM[0]) / (N_FINE - 1)
CAPS = {"w32": (32, 32), "w128": (128, 128), "w512": (512, 512)}

# corridor-thickness bins in plane units (0-4e-3 ... 3.2e-2+)
THICKNESS_BINS = [(0.0, 0.004), (0.004, 0.008), (0.008, 0.016),
                  (0.016, 0.032), (0.032, 1e9)]
# distance-from-core bins for the thin corridors (t < 0.03)
DISTANCE_BINS = (0.0, 0.01, 0.03, 0.08, 0.25, 1e9)
T_THIN = 0.02
T_CORE = 0.05


# ----------------------------------------------------------------------
def thickness(labels):
    """``t = 2 * distance to the nearest cell of another basin``, and the
    boundary mask it is built from.  Units: plane units."""
    lab = labels
    bnd = np.zeros_like(lab, dtype=bool)
    bnd[:-1, :] |= lab[:-1, :] != lab[1:, :]
    bnd[1:, :] |= lab[1:, :] != lab[:-1, :]
    bnd[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    bnd[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    d = ndimage.distance_transform_edt(~bnd)
    return 2.0 * d * CELL, bnd


def bin_id(t, bins=None):
    """Index of the thickness bin each cell falls in (``-1`` if outside)."""
    bins = THICKNESS_BINS if bins is None else bins
    out = np.full(t.shape, -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(bins):
        out[(t >= lo) & (t < hi)] = i
    return out


def neighbour_label(labels, cells, k=5):
    """Dominant basin near each cell, EXCLUDING the cell's own basin.

    ``cells`` is a sequence of ``(y, x, own_label)``.  Without the exclusion
    the "freeze" rate would be equal to the recovery rate: the dominant label
    around a cell is almost always the cell's own basin.
    """
    lab = labels
    out = []
    for y, x, own in cells:
        y0, y1 = max(0, y - k), min(lab.shape[0], y + k + 1)
        x0, x1 = max(0, x - k), min(lab.shape[1], x + k + 1)
        vals, counts = np.unique(lab[y0:y1, x0:x1], return_counts=True)
        keep = (vals > 0) & (vals != own)
        if not keep.any():
            out.append(-1)          # no neighbouring basin in the window
            continue
        vals, counts = vals[keep], counts[keep]
        out.append(int(vals[counts.argmax()]))
    return np.array(out)


def neighbour_of(labels, y, x, own, k=5):
    """Same as :func:`neighbour_label` for one cell."""
    y0, y1 = max(0, y - k), min(labels.shape[0], y + k + 1)
    x0, x1 = max(0, x - k), min(labels.shape[1], x + k + 1)
    vals, counts = np.unique(labels[y0:y1, x0:x1], return_counts=True)
    keep = (vals > 0) & (vals != own)
    if not keep.any():
        return -1
    vals, counts = vals[keep], counts[keep]
    return int(vals[counts.argmax()])


# ----------------------------------------------------------------------
def core_distance(labels, t, core_t=T_CORE):
    """Euclidean distance (plane units) from each cell to the nearest core
    cell of its OWN basin (``t >= core_t``).  Cells without a core: ``inf``."""
    d_out = np.full(labels.shape, np.inf, dtype=np.float32)
    for k in (1, 2, 3):
        basin = labels == k
        core = basin & (t >= core_t)
        if not core.any():
            continue
        d = ndimage.distance_transform_edt(~core) * CELL
        m = basin & np.isfinite(d)
        d_out[m] = np.minimum(d_out[m], d[m])
    return d_out


def geodesic_core_distance(labels, t, y, x, own, radius=150, core_t=T_CORE):
    """True geodesic distance from the core, by BFS on a local window.

    Used only to validate :func:`core_distance`; for winding corridors the
    Euclidean estimate underestimates the geodesic one.
    """
    n = labels.shape[0]
    y0, y1 = max(0, y - radius), min(n, y + radius + 1)
    x0, x1 = max(0, x - radius), min(n, x + radius + 1)
    sub = labels[y0:y1, x0:x1]
    tsub = t[y0:y1, x0:x1]
    basin = sub == own
    core = basin & (tsub >= core_t)
    if not core.any():
        return np.nan
    dist = np.full(sub.shape, -1, np.int32)
    dist[core] = 0
    frontier = core.copy()
    struct = np.ones((3, 3), bool)
    d = 0
    while frontier.any() and d < 2 * radius:
        d += 1
        grow = ndimage.binary_dilation(frontier, structure=struct,
                                       mask=basin) & (dist < 0)
        if not grow.any():
            break
        dist[grow] = d
        frontier = grow
    v = dist[y - y0, x - x0]
    return float(v) * CELL if v >= 0 else np.nan


# ----------------------------------------------------------------------
def prepare_thickness(labels=None, t=None, seed=0, per_bin=1200):
    """Evaluation cells stratified by corridor thickness.

    Returns ``(strata, labels)`` where each stratum is a dict with the bin
    range, the sampled coordinates, their true labels, the dominant
    neighbouring basin, ``t`` and the cell count.
    """
    labels = cached_fine_labels(N_FINE, XLIM) if labels is None else labels
    t = thickness(labels)[0] if t is None else t
    b = bin_id(t)
    rng = np.random.default_rng(seed)
    g = np.linspace(*XLIM, N_FINE)
    flat_lab = labels.ravel()
    strata = []
    for i, (lo, hi) in enumerate(THICKNESS_BINS):
        flat = np.nonzero((b == i).ravel() & (flat_lab > 0))[0]
        if len(flat) == 0:
            continue
        sel = rng.choice(flat, min(per_bin, len(flat)), replace=False)
        ys, xs = np.divmod(sel, N_FINE)
        xy = np.column_stack([g[xs], g[ys]])
        near = neighbour_label(labels, list(zip(ys, xs, flat_lab[sel])))
        strata.append(dict(bin=i, lo=lo, hi=hi, xy=xy, label=flat_lab[sel],
                           near=near, t=t.ravel()[sel], n=len(sel)))
    return strata, labels


def prepare_core(labels=None, t=None, seed=0, per_cell=100, t_max=0.03):
    """Evaluation cells stratified by (thickness, distance from the core).

    Only cells with ``t < t_max`` (thin corridors) are used, because that is
    where peninsulas live.
    """
    labels = cached_fine_labels(N_FINE, XLIM) if labels is None else labels
    t = thickness(labels)[0] if t is None else t
    d = core_distance(labels, t)
    rng = np.random.default_rng(seed)
    g = np.linspace(*XLIM, N_FINE)
    thin = (t < t_max) & (labels > 0) & np.isfinite(d)
    ys, xs = np.nonzero(thin)
    dv = d[ys, xs]
    strata = []
    for i in range(len(DISTANCE_BINS) - 1):
        m = (dv >= DISTANCE_BINS[i]) & (dv < DISTANCE_BINS[i + 1])
        if m.sum() < 50:
            continue
        sel = rng.choice(np.nonzero(m)[0], min(per_cell, int(m.sum())),
                         replace=False)
        gy, gx = ys[sel], xs[sel]
        own = labels[gy, gx]
        near = np.array([neighbour_of(labels, y, x, o)
                         for y, x, o in zip(gy, gx, own)])
        strata.append(dict(d_bin=i, d_lo=DISTANCE_BINS[i],
                           d_hi=DISTANCE_BINS[i + 1],
                           t_mean=float(t[gy, gx].mean()),
                           d_mean=float(dv[sel].mean()),
                           xy=np.column_stack([g[gx], g[gy]]),
                           label=own.astype(np.int8), near=near,
                           n=int(len(sel))))
    return strata, labels, t, d


def raster_train_grid(n, labels=None, seed=0, max_train=60_000):
    """Uniform ``n x n`` training grid whose labels are read off the fine
    label raster (no re-integration, so it is fast and consistent).

    Returns ``(X, y, h)`` with ``h`` the nominal spacing ``window/(n-1)``.
    """
    labels = cached_fine_labels(N_FINE, XLIM) if labels is None else labels
    g = np.linspace(*XLIM, n)
    xp, yp = np.meshgrid(g, g, indexing="xy")
    fy = np.rint((yp - XLIM[0]) / CELL).astype(int)
    fx = np.rint((xp - XLIM[0]) / CELL).astype(int)
    lab = labels[np.clip(fy, 0, N_FINE - 1), np.clip(fx, 0, N_FINE - 1)].ravel()
    xy = np.column_stack([xp.ravel(), yp.ravel()])
    keep = lab > 0
    xy, lab = xy[keep], lab[keep].astype(np.int8)
    if len(xy) > max_train:
        sel = np.random.default_rng(seed).choice(len(xy), max_train,
                                                 replace=False)
        xy, lab = xy[sel], lab[sel]
    return xy, lab, 3.0 / (n - 1)
