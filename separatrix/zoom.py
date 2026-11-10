"""Resolution threshold: the same island at a controlled relative scale.

At the demo's 300x300 grid, "the model smooths the island away" and "the island
was never sampled" are the same statement, because the islands are 3-6 fine
cells across.  To separate them, hold the island fixed in the plane and zoom
the training window: for an island of width ``w``, train on an ``n x n`` grid
spanning ``k*w``.  The island then spans ``(n-1)/k`` training cells, while the
domain size (in cells), the number of training points and the architecture all
stay fixed.

Sweeping ``k`` from 1.5 to 150 takes the island from "huge" to "smaller than one
sampling cell".  Repeating it for several capacities says where the ceiling
comes from.
"""

import time

import numpy as np

from .classifier import MCClassifier
from .evaluate import grid_labels
from .islands import island_points

__all__ = ["ZOOMS", "WIDTHS", "N_GRID", "XLIM", "N_FINE", "CELL",
           "island_center", "island_width", "window_of", "zoom_case"]

ZOOMS = (1.5, 3.0, 6.0, 15.0, 40.0, 150.0)      # window width / island width
WIDTHS = ((32, 32), (128, 128), (512, 512))
N_GRID = 200                                     # training points per axis
XLIM = (-1.5, 1.5)
N_FINE = 3200
CELL = (XLIM[1] - XLIM[0]) / (N_FINE - 1)
MAX_SAMPLES = 60_000


def island_center(isl):
    """Centroid of an island dict (fine-grid indices) in plane units."""
    cx, cy = isl["centroid"]
    return XLIM[0] + cx * CELL, XLIM[0] + cy * CELL


def island_width(isl):
    """Island width in plane units (from its area)."""
    return isl["scale_cells"] * CELL


def window_of(isl, k):
    """Window ``(centre_x, centre_y, half_width)`` for zoom factor ``k``."""
    cx, cy = island_center(isl)
    return cx, cy, 0.5 * k * island_width(isl)


def zoom_case(isl, k, hidden, n_grid=N_GRID, epochs=200, seed=0, n_field=0):
    """Train one classifier on a window of width ``k * island width``.

    Returns a dict with

    * ``recover`` / ``freeze`` -- fractions of the island's interior cells that
      the model labels with the island's own / the surrounding basin;
    * ``acc_far`` -- accuracy on training points further than ``2w`` from the
      island, i.e. the honest "everything else is fine" control;
    * ``island_cells`` -- island width in training cells, the x-axis of
      ``fig6`` (``(n_grid - 1) / k``);
    * optionally ``field`` -- the predicted label field over the window, which
      is what the interactive demo displays.
    """
    cx, cy, half = window_of(isl, k)
    rng = np.random.default_rng(seed)

    g = np.linspace(-half, half, n_grid)
    xp, yp = np.meshgrid(g, g, indexing="xy")
    xy = np.column_stack([(xp + cx).ravel(), (yp + cy).ravel()])
    lab = grid_labels(xy)
    keep = lab > 0
    X, y = xy[keep], lab[keep].astype(np.int8)
    if len(X) > MAX_SAMPLES:
        sel = rng.choice(len(X), MAX_SAMPLES, replace=False)
        X, y = X[sel], y[sel]

    clf = MCClassifier(hidden=hidden, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X, y, max_iter=epochs)
    fit_s = time.perf_counter() - t0

    interior = island_points(isl, N_FINE, xlim=XLIM)
    pred_in = clf.predict(interior)
    recover = float((pred_in == isl["label"]).mean())
    freeze = float((pred_in == isl["surrounding"]).mean())

    w = island_width(isl)
    d = np.hypot(X[:, 0] - cx, X[:, 1] - cy)
    far = d > 2.0 * w
    acc_far = float((clf.predict(X[far]) == y[far]).mean()) if far.any() else float("nan")

    field = None
    if n_field:
        gf = np.linspace(-half, half, n_field)
        xf, yf = np.meshgrid(gf + cx, gf + cy, indexing="xy")
        fxy = np.column_stack([xf.ravel(), yf.ravel()])
        field = clf.predict(fxy).reshape(n_field, n_field).astype(np.uint8)

    return dict(zoom=k, hidden=list(hidden),
                island_cells=(n_grid - 1) / k,
                island_units=w, window_units=2 * half,
                n_train=int(len(X)), recover=recover, freeze=freeze,
                acc_far=acc_far, epochs=int(clf.n_epochs_),
                fit_s=round(fit_s, 2), window=(cx, cy, half),
                field=field)
