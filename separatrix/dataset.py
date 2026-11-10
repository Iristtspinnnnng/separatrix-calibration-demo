"""Datasets: labelled initial conditions for the Newton-fractal benchmark.

This is stage 1 of the paper's pipeline (arXiv:2608.14743, Algorithm 1, input
``D1``): draw initial conditions on a uniform grid over the complex plane,
push each one through the Newton map, and record which root it converges to.

The grid geometry matches the DynamicalSystems.jl example
(``ignoredimension`` / ``range(-1.5, 1.5, length=400)``) and the paper's
Sec. 3.1 (a 200x200 mesh grid, i.e. 40 000 points).
"""

import numpy as np

from .dynamics import newton_labels

__all__ = ["make_dataset", "grid_xy", "grid_dataset"]


def make_dataset(xlim=(-1.5, 1.5), n=400, p=3, seed=0, n_max=40):
    """Uniform grid of initial conditions labelled by their basin.

    Parameters
    ----------
    xlim : tuple of float
        The plane window; the grid is ``n x n`` over ``xlim x xlim``.
    n : int
        Number of grid points per axis (``n**2`` initial conditions).
    seed : int
        Unused for the (deterministic) grid; kept so that callers can change
        the sampling scheme without changing the signature.

    Returns
    -------
    X : ndarray, shape (m, 2)   -- retained initial conditions ``(Re z, Im z)``
    y : ndarray, shape (m,)     -- basin label ``1..p``
    grid : tuple (xg, yg)       -- the 1D coordinate vectors

    Initial conditions that do not converge to a root (``y == 0``) are
    dropped: they are numerically rare, and a pure classifier demo cannot use
    them.  The escape cut is deliberately large (``1e8``) so that points which
    Newton's map flings out to ``|z| ~ 1e3`` and then brings back are still
    labelled -- that neighbourhood of the origin is genuinely fractal.
    """
    del seed                                     # deterministic grid
    g = np.linspace(*xlim, n)
    xy = grid_xy(g)
    lab = newton_labels(xy, p=p, n_max=n_max, escape=1e8)
    keep = lab > 0
    return xy[keep], lab[keep].astype(np.int8), (g, g)


def grid_xy(g, gy=None):
    """Coordinates of a grid, flattened in ``meshgrid(..., indexing="xy")``
    order (row-major over ``y``, then ``x``)."""
    gy = g if gy is None else gy
    xp, yp = np.meshgrid(g, gy, indexing="xy")
    return np.column_stack([xp.ravel(), yp.ravel()])


def grid_dataset(n=300, xlim=(-1.5, 1.5), p=3, n_max=40, escape=1e3):
    """Labelled uniform grid of training points, from the dynamics directly.

    Unlike :func:`make_dataset` no point is dropped: the label array keeps
    ``0`` for cells that do not converge, and callers decide (the refinement
    loop drops them, the island experiments keep them as a separate class).
    """
    g = np.linspace(*xlim, n)
    xy = grid_xy(g)
    lab = newton_labels(xy, p=p, n_max=n_max, escape=escape)
    return xy, lab.astype(np.int8), (g, g)
