"""Statistics for the phase diagrams: seed averaging, freeze boundary, CIs.

Everything that turns a CSV of per-(configuration, seed) measurements into
the numbers quoted in the README lives here, so that the analysis scripts are
thin wrappers and the numbers can be re-derived (and tested) independently
of formatting and plotting.

Vocabulary
----------
``w/h``
    island width divided by the sampling spacing, i.e. "how many samples fit
    across the feature".  Acute: for the window-based island phase diagram
    the *effective* spacing is ``window / sqrt(n_train)`` when the 40k training
    cap subsamples the grid (:func:`load_phase_csv` does the correction).

``w_c``
    the freeze boundary: the smallest ``w/h`` at which the seed-mean recovery
    reaches 0.5, obtained from the monotone envelope of the curve and
    interpolated in ``log(w/h)``.  ``w_c`` is where a feature stops being
    recoverable.

``bootstrap CI``
    resample *seeds* with replacement, recompute ``w_c``, take the 2.5/97.5
    percentiles.  Seeds, not points: the seed is the independent repetition.
"""

import csv

import numpy as np

__all__ = ["crossing", "curve", "interp_envelope", "bootstrap_wc", "ci",
           "power_law", "load_phase_csv", "load_simple_csv"]

CAPS = ("w32", "w128", "w512")
COLORS = {"w32": "#95a5a6", "w128": "#2980b9", "w512": "#c0392b"}
MARKERS = {"w32": "o", "w128": "s", "w512": "^"}


def crossing(x, y, level=0.5):
    """Smallest ``x`` at which the monotone envelope of ``y`` reaches ``level``.

    Monotone envelope first (recovery is a monotone function of scale up to
    noise), then a linear interpolation in ``log x``.  Returns ``nan`` if the
    curve never reaches the level.
    """
    o = np.argsort(np.asarray(x, float))
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    env = np.maximum.accumulate(y)
    if env.max() < level:
        return np.nan
    i = int(np.argmax(env >= level))
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = np.log(x[i - 1]), np.log(x[i]), env[i - 1], env[i]
    t = 0.0 if y1 == y0 else (level - y0) / (y1 - y0)
    return float(np.exp(x0 + t * (x1 - x0)))


def curve(rows, key, cap, window=None):
    """Seed mean and std (ddof=1) of ``key`` versus ``w/h``."""
    sel = [r for r in rows if r["cap"] == cap
           and (window is None or r["window"] == window)]
    xs = np.array([r["w_over_h"] for r in sel], float)
    ys = np.array([r[key] for r in sel], float)
    ux = np.unique(xs)
    mean = np.array([ys[xs == u].mean() for u in ux])
    std = np.array([ys[xs == u].std(ddof=1) if (xs == u).sum() > 1 else 0.0
                    for u in ux])
    return ux, mean, std


def interp_envelope(x, y, targets):
    """Monotone envelope of a seed-mean curve, interpolated in ``log x``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], np.maximum.accumulate(y[o])
    return np.interp(np.log(np.asarray(targets, float)), np.log(x), y,
                     left=np.nan, right=np.nan)


def bootstrap_wc(rows, cap, window=None, n_boot=400, seed=0):
    """Resample seeds with replacement and return the ``w_c`` samples."""
    rng = np.random.default_rng(seed)
    seeds = sorted({r["seed"] for r in rows})
    pool = [r for r in rows if r["cap"] == cap
            and (window is None or r["window"] == window)]
    out = []
    for _ in range(n_boot):
        pick = set(rng.choice(seeds, len(seeds), replace=True).tolist())
        agg = {}
        for r in pool:
            if r["seed"] in pick:
                agg.setdefault(r["w_over_h"], []).append(r["recovery"])
        wc = crossing(list(agg), [np.mean(v) for v in agg.values()])
        if not np.isnan(wc):
            out.append(wc)
    return np.array(out)


def ci(samples, lo=2.5, hi=97.5):
    """Percentile CI, or ``(nan, nan)`` when there are too few samples."""
    if len(samples) <= 20:
        return (np.nan, np.nan)
    return tuple(np.percentile(samples, [lo, hi]))


def power_law(params, w_c):
    """Fit ``w_c ~ params**slope``; returns ``(slope, intercept, gain)``.

    ``gain`` is the factor by which ``w_c`` moves between the smallest and the
    largest capacity in the list.
    """
    ps = np.asarray(params, float)
    ws = np.asarray(w_c, float)
    ok = np.isfinite(ws)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(np.log(ps[ok]), np.log(ws[ok]), 1)
    return float(slope), float(intercept), float(ws[0] / ws[-1])


def load_phase_csv(path, drop_per_island=True):
    """Read a phase-diagram CSV and apply the effective-spacing correction.

    The island phase diagram keeps the island fixed in the plane and zooms the
    training window.  ``phase_experiment.py`` caps the training set at 40k, so
    a 300x300 window (90k cells) is subsampled and the spacing the classifier
    actually sees is ``window / sqrt(n_train)``, not ``window / (n-1)``.
    Using the nominal ``w/h`` silently mixes the two and was the bug behind
    the first (wrong) estimate of the freeze boundary.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if drop_per_island and "_isl" in r["config"]:
                continue                      # one row per island, not per run
            parts = r["config"].split("_")
            win = int(parts[0][3:])
            w_nom, h_nom = float(r["w_over_h"]), float(r["h"])
            n_train = int(r["n_train"])
            h_eff = h_nom * (win - 1) / np.sqrt(n_train)
            rows.append(dict(cap=r["cap"], window=win,
                             zoom=int(parts[1][1:]),
                             w_over_h=w_nom * h_nom / h_eff,
                             w_over_h_nominal=w_nom,
                             n_train=n_train,
                             params=float(r["params"]),
                             seed=int(r["seed"]),
                             recovery=float(r["recovery"]),
                             freeze=float(r["freeze"])))
    return rows


def load_simple_csv(path):
    """Read a CSV whose ``w/h`` column is already the effective one."""
    with open(path, encoding="utf-8") as f:
        return [dict(cap=r["cap"], seed=int(r["seed"]),
                     w_over_h=float(r["w_over_h"]),
                     n_train=int(r["n_train"]),
                     recovery=float(r["recovery"]),
                     freeze=float(r["freeze"]),
                     params=float(r["params"]))
                for r in csv.DictReader(f)]
