"""Basin dynamics: Newton's method for ``f(z) = z**p - 1`` (default ``p=3``).

This is the benchmark used in Crabtree, Giovanis, Georgiou, Datseris and
Kevrekidis (2026), *Generative Learning of Separatrices*, arXiv:2608.14743,
Fig. 1, and it is the example of the "Discrete system" section of the
DynamicalSystems.jl documentation for ``basins_of_attraction``.  The Julia
original is a 15-line snippet; everything below is its line-for-line Python
translation, plus a vectorised labeller for whole grids.

The Newton map is

    z <- z - f(z) / f'(z) = z - (z**p - 1) / (p * z**(p - 1)),

whose attractors are the ``p`` roots of unity.  The basin boundaries are a
fractal (the Julia set of the map) of Lebesgue measure zero, which is what
makes "accuracy on the separatrix" a subtle quantity -- see
:mod:`separatrix.evaluate`.

Attractor ids used everywhere in this repository:

===== ==========================================================
``1`` root ``+1``
``2`` root ``exp(+2*pi*i/p)``
``3`` root ``exp(-2*pi*i/p)``
``0`` not classified (escaped, or ``|z|`` blew up)
``-1`` not converged within ``n_max`` iterations (grid labeller only)
===== ==========================================================

References
----------
* G. Datseris (2022), "Effortless estimation of basins of attraction",
  Chaos 32, 023104, doi:10.1063/5.0076568.
* E. R. Crabtree, D. G. Giovanis, A. Georgiou, G. Datseris, I. G. Kevrekidis
  (2026), "Generative Learning of Separatrices", arXiv:2608.14743, Sec. 3.1.
"""

import numpy as np

__all__ = [
    "newton_f",
    "newton_df",
    "newton_step",
    "newton_root",
    "newton_labels",
    "basin_image",
]

# |z| beyond this is treated as "escaped" by the vectorised labeller.  The
# scalar reference integrator uses a much larger cut because it follows one
# orbit at a time; the two agree everywhere except within a few cells of the
# origin, where Newton's map sends z ~ 0 to z ~ 1/(p z**(p-1)) and back.
DEFAULT_ESCAPE = 1e3


def newton_f(z, p=3):
    """``f(z) = z**p - 1``.  Its roots are the attractors of the map."""
    return z**p - 1.0


def newton_df(z, p=3):
    """``f'(z) = p z**(p-1)``."""
    return p * z ** (p - 1)


def newton_step(z, p=3):
    """One Newton iteration: ``z <- z - f(z)/f'(z)``."""
    return z - newton_f(z, p) / newton_df(z, p)


def newton_root(z0, p=3, n_max=40, tol=1e-6, r_max=1e8):
    """Iterate the Newton map from a single initial condition ``z0``.

    Returns the id of the attractor (``1..p``) or ``0`` if the orbit did not
    converge to a root in ``n_max`` steps, blew up, or reached ``|z| > r_max``.

    The roots lie on the unit circle; the id is read off as the nearest root
    once ``|z - root| < tol``, which is robust right up to the fractal
    boundary (a point can be arbitrarily close to two roots without being
    "on" the separatrix).
    """
    z = z0
    roots = np.exp(2j * np.pi * np.arange(p) / p)

    for _ in range(n_max):
        z = newton_step(z, p)
        if not np.isfinite(z):
            return 0
        if abs(z) > r_max:
            return 0
        for k, r in enumerate(roots, start=1):
            if abs(z - r) < tol:
                return k
    return 0


def _label_field(z, p=3, n_max=40, tol=1e-6, escape=DEFAULT_ESCAPE):
    """Vectorised Newton labeller on a complex array of any shape.

    Returns an ``int8`` array of the same shape: ``1..p`` for the basin,
    ``0`` for orbits that escape (``|z| > escape`` or non-finite), ``-1`` for
    orbits still unconverged after ``n_max`` steps (numerically these are the
    Julia set itself, i.e. a set of measure zero).
    """
    z = np.asarray(z, dtype=np.complex128)
    z = np.where(np.abs(z) < 1e-12, 1e-12 + 0j, z)     # f'(0) = 0
    escaped = np.zeros(z.shape, dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(n_max):
            z = z - (z**p - 1.0) / (p * z ** (p - 1))
            escaped |= (np.abs(z) > escape) | ~np.isfinite(z)
            z = np.where(escaped | ~np.isfinite(z), 0.0, z)
        roots = np.exp(2j * np.pi * np.arange(p) / p)
        d = np.abs(z[:, :, None] - roots[None, None, :])
        lab = d.argmin(axis=2) + 1
        ok = d.min(axis=2) < tol
    out = np.full(z.shape, -1, dtype=np.int8)
    out[ok] = lab[ok]
    out[escaped] = 0
    return out


def newton_labels(xy, p=3, n_max=40, tol=1e-6, escape=DEFAULT_ESCAPE):
    """Basin labels for a cloud of points ``xy`` of shape ``(n, 2)``.

    Same conventions as :func:`_label_field`; ``0`` and ``-1`` both mean
    "not a basin cell".  For a 400x400 grid this is ~0.1 s instead of the
    ~30 s a Python loop over :func:`newton_root` needs.
    """
    xy = np.asarray(xy, dtype=np.float64)
    z = (xy[:, 0] + 1j * xy[:, 1])[:, None]
    return _label_field(z, p=p, n_max=n_max, tol=tol, escape=escape).ravel()


def basin_image(n=1600, xlim=(-1.5, 1.5), p=3, n_max=40, tol=1e-6,
                escape=DEFAULT_ESCAPE):
    """Basin labels on an ``n x n`` uniform grid over ``xlim`` (an image).

    Cell ``[j, i]`` corresponds to ``x = linspace(xlim)[i]``,
    ``y = linspace(xlim)[j]`` -- the same convention as
    ``numpy.meshgrid(..., indexing="xy")``, so the array can be shown
    directly with ``imshow(origin="lower")``.
    """
    g = np.linspace(*xlim, n)
    xp, yp = np.meshgrid(g, g, indexing="xy")
    return _label_field(xp + 1j * yp, p=p, n_max=n_max, tol=tol, escape=escape)
