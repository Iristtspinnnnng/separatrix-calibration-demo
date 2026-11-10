"""evaluate.py
========================================================================
"How do you measure accuracy on a fractal separatrix?"

The separatrix of the p=3 Newton map is the Julia set of z -> z - (z^3-1)/(3z^2):
a fractal with Lebesgue measure ZERO.  You can therefore never evaluate
accuracy *on* it -- any finite sample of it has probability 0.  What you
can measure is accuracy inside a BAND of width w around it, and the
number you get depends on w.  This module makes that explicit:

  1. labels on an independent uniform grid (vectorised Newton iteration,
     so a 401x401 grid takes ~0.1 s instead of a Python loop);
  2. the fractal band as a *geometric* object: a grid cell belongs to the
     band of width k cells if it is within Chebyshev distance k of a cell
     whose basin label differs from a 4-neighbour's (i.e. of a point where
     the labelled field is discontinuous).  `band_cells=1` is the thinnest
     band the grid can resolve; `band_cells=8` is a genuinely thick halo
     that includes the "slow", nearly-fractal approach to the separatrix;
  3. the same metrics the demo prints, evaluated on a grid that is NOT the
     training grid (different resolution, so the classifier cannot just
     memorise it).

The old oracle (`separatrix.refine.true_boundary_mask`) is kept and compared
against the geometric band: it is a *dynamical* proxy (a point is "near
the separatrix" if its Newton iterate escapes the unit disc within 6
steps), cheap and useful for caching, but its width is set implicitly by
the 6-step threshold rather than by a distance.
========================================================================
"""
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from .dynamics import newton_labels, newton_root
from .refine import true_boundary_mask

__all__ = ["grid_labels", "make_eval_grid", "evaluate_on_grid",
           "expected_calibration_error", "params_of"]


# ----------------------------------------------------------------------
def grid_labels(xy, p=3, n_max=40, tol=1e-6):
    """Basin labels for many points at once (vectorised Newton map).

    Thin alias of :func:`separatrix.dynamics.newton_labels`, kept because the
    island and phase scripts call it on arbitrary point clouds.  ``0`` and
    ``-1`` both mean "not a basin cell" (escaped / still unconverged).
    """
    return newton_labels(xy, p=p, n_max=n_max, tol=tol)


# ----------------------------------------------------------------------
def make_eval_grid(n=401, xlim=(-1.5, 1.5), bands=(1, 2, 4, 8),
                   interior_band=8):
    """
    Independent evaluation grid + fractal-band masks.

    Returns dict with
        xy        (n*n, 2) coordinates
        labels    (n*n,)   basin label, 0 = unconverged
        bands     {k: bool mask "within k cells of a label discontinuity"}
        interior  bool mask "further than `interior_band` cells from it"
        escape    bool mask from the old dynamical proxy (for comparison)
    """
    g = np.linspace(*xlim, n)
    Xp, Yp = np.meshgrid(g, g, indexing="xy")
    xy = np.column_stack([Xp.ravel(), Yp.ravel()])
    lab_flat = grid_labels(xy)
    # 0 == "not a basin cell" (escaped or unconverged) is its own class, so
    # that the frontier around it is counted like any other discontinuity.
    lab = np.where(lab_flat > 0, lab_flat, 0).reshape(n, n)

    # cell is on the "frontier" if a 4-neighbour has a different label
    # (unconverged cells count as their own class)
    frontier = np.zeros_like(lab, dtype=bool)
    frontier[:-1, :] |= lab[:-1, :] != lab[1:, :]
    frontier[1:, :] |= lab[1:, :] != lab[:-1, :]
    frontier[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    frontier[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    frontier |= lab == 0

    struct = np.ones((3, 3), dtype=bool)
    bands_m = {}
    for k in bands:
        m = frontier.copy()
        for _ in range(int(k) - 1):
            m = ndimage.binary_dilation(m, structure=struct)
        bands_m[int(k)] = m

    outer = frontier.copy()
    for _ in range(int(interior_band)):
        outer = ndimage.binary_dilation(outer, structure=struct)

    return dict(xy=xy, labels=lab.ravel(), n=n,
                bands={k: v.ravel() for k, v in bands_m.items()},
                interior=~outer.ravel(),
                escape=true_boundary_mask(xy),
                grid=g)


# ----------------------------------------------------------------------
def expected_calibration_error(conf, correct, n_bins=10):
    """Standard binned ECE."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def params_of(hidden, n_in=2, n_out=3):
    sizes = (n_in,) + tuple(hidden) + (n_out,)
    return int(sum(sizes[i] * sizes[i + 1] + sizes[i + 1]
                   for i in range(len(sizes) - 1)))


# ----------------------------------------------------------------------
def evaluate_on_grid(clf, ev, mc_samples=12, bands=(1, 2, 4, 8)):
    """
    All metrics on the independent grid, for the band widths in `bands`.

    Returns a flat dict, e.g.
        acc_band1, acc_band4, acc_interior, conf_band1, ece_band1,
        overconf_wrong_band1  (fraction of band points with conf>0.9 and a
        wrong label -- the "confidently wrong on the separatrix" number)
        sep_auc1  (AUROC of entropy separating band1 from interior)
    """
    xy = ev["xy"]
    mean_proba, H = clf.mc_predict(xy, M=mc_samples)
    pred = clf.predict(xy)
    correct = pred == ev["labels"]
    conf = mean_proba.max(axis=1)

    out = dict(acc_all=float(correct.mean()),
               ece=expected_calibration_error(conf, correct),
               mean_conf=float(conf.mean()),
               n_eval=int(len(xy)))

    interior = ev["interior"]
    out["acc_interior"] = float(correct[interior].mean())
    out["conf_interior"] = float(conf[interior].mean())
    out["ece_interior"] = expected_calibration_error(conf[interior],
                                                     correct[interior])

    for k in bands:
        m = ev["bands"][k]
        out[f"acc_band{k}"] = float(correct[m].mean())
        out[f"conf_band{k}"] = float(conf[m].mean())
        out[f"ece_band{k}"] = expected_calibration_error(conf[m], correct[m])
        out[f"frac_band{k}"] = float(m.mean())
        out[f"n_band{k}"] = int(m.sum())
        # confidently wrong ON the separatrix band
        out[f"overconf_wrong_band{k}"] = float(
            ((conf[m] > 0.9) & (~correct[m])).mean())

    # entropy separability: band1 vs interior is almost trivial (AUROC ~1),
    # so also report the HARD contrast: the thinnest band vs the rest of the
    # halo.  "Can the entropy still rank the exact frontier above points that
    # are only a few cells away?"
    lo_b, hi_b = min(bands), max(bands)
    try:
        sub = ev["bands"][lo_b] | interior
        out["sep_auc1"] = float(roc_auc_score(
            ev["bands"][lo_b][sub].astype(int), H[sub]))
    except Exception:
        out["sep_auc1"] = float("nan")
    try:
        mid = ev["bands"][hi_b] & ~ev["bands"][lo_b]
        sub = ev["bands"][lo_b] | mid
        y_bin = ev["bands"][lo_b][sub].astype(int)
        out["sep_auc_hard"] = float(roc_auc_score(y_bin, H[sub]))
    except Exception:
        out["sep_auc_hard"] = float("nan")

    # agreement between the old dynamical proxy and the geometric band1
    b1, esc = ev["bands"][min(bands)], ev["escape"]
    inter = (b1 & esc).sum()
    out["escape_precision"] = float(inter / max(1, esc.sum()))
    out["escape_recall"] = float(inter / max(1, b1.sum()))
    return out


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    ev = make_eval_grid()
    print(f"eval grid {ev['n']}x{ev['n']} = {len(ev['xy'])} points "
          f"({time.perf_counter()-t0:.2f}s)")
    for k, m in ev["bands"].items():
        print(f"  band {k:>2} cells: {100*m.mean():5.2f}% of the grid")
    print(f"  interior (>{8} cells): {100*ev['interior'].mean():5.2f}%")
    print(f"  old dynamical proxy : {100*ev['escape'].mean():5.2f}%")
    b1 = ev["bands"][1]
    print(f"  proxy vs band1: precision "
          f"{(b1 & ev['escape']).sum()/ev['escape'].sum():.3f}, recall "
          f"{(b1 & ev['escape']).sum()/b1.sum():.3f}")

    # sanity: the vectorised labeller must agree with the scalar one
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ev["xy"]), 3000, replace=False)
    scal = np.array([newton_root(complex(*ev["xy"][i]), p=3) for i in idx])
    print("labeller agreement with newton_root:",
          f"{(grid_labels(ev['xy'][idx]) == scal).mean():.4f}")
