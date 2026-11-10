"""Unit tests: dynamics, dataset, entropy, the band oracle, the statistics.

Run with ``python -m pytest tests -q`` from the repository root (or after
``pip install -e .`` without the sys.path shim below).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from separatrix.classifier import MCClassifier, predictive_entropy
from separatrix.dataset import grid_dataset, make_dataset
from separatrix.dynamics import (basin_image, newton_f, newton_root,
                                 newton_step, newton_labels)
from separatrix.evaluate import (expected_calibration_error, grid_labels,
                                 make_eval_grid, params_of)
from separatrix.paths import CACHE, FIGURES, RESULTS, cache, figure, result
from separatrix.stats import bootstrap_wc, ci, crossing, curve


# ----------------------------------------------------------------------
# dynamics
# ----------------------------------------------------------------------
@pytest.mark.parametrize("p", [2, 3, 4])
def test_roots_are_fixed_points(p):
    for k in range(p):
        r = np.exp(2j * np.pi * k / p)
        assert abs(newton_f(r, p)) < 1e-12
        assert abs(newton_step(r, p) - r) < 1e-12


def test_newton_root_is_deterministic_and_bounded():
    rng = np.random.default_rng(0)
    z = rng.uniform(-1.5, 1.5, 200) + 1j * rng.uniform(-1.5, 1.5, 200)
    lab = np.array([newton_root(z0) for z0 in z])
    assert set(np.unique(lab)).issubset({0, 1, 2, 3})
    assert (lab > 0).mean() > 0.9          # almost everything converges


def test_vectorised_labeller_matches_scalar():
    rng = np.random.default_rng(1)
    xy = np.column_stack([rng.uniform(-1.5, 1.5, 500),
                          rng.uniform(-1.5, 1.5, 500)])
    vec = newton_labels(xy)
    scal = np.array([newton_root(complex(*p)) for p in xy])
    # the vectorised labeller cuts at |z| > 1e3, the scalar one at 1e8, so
    # they may differ on a couple of points near the origin
    assert (vec == scal).mean() > 0.99


def test_basin_image_conventions():
    lab = basin_image(n=64)
    assert lab.shape == (64, 64)
    assert lab.dtype == np.int8
    assert set(np.unique(lab)).issubset({-1, 0, 1, 2, 3})
    # the three basins are roughly equal by symmetry
    for k in (1, 2, 3):
        assert 0.2 < (lab == k).mean() < 0.45


def test_grid_labels_is_the_same_labeller():
    xy = np.array([[0.3, 0.1], [-1.2, 0.4], [0.5, -0.7]])
    assert np.array_equal(grid_labels(xy), newton_labels(xy))


# ----------------------------------------------------------------------
# dataset and classifier
# ----------------------------------------------------------------------
def test_dataset_labels_in_range():
    X, y, grid = make_dataset(n=40)
    assert set(np.unique(y)).issubset({1, 2, 3})
    assert X.shape[0] == y.shape[0]
    assert grid[0].shape == (40,)


def test_grid_dataset_keeps_every_cell():
    xy, lab, grid = grid_dataset(n=32)
    assert len(xy) == len(lab) == 32 * 32
    assert grid[0][0] == pytest.approx(-1.5)


def test_entropy_is_high_on_mixture():
    uniform = np.full((5, 3), 1 / 3)
    peaked = np.array([[0.99, 0.005, 0.005]] * 5)
    assert predictive_entropy(uniform).mean() > predictive_entropy(peaked).mean()


def test_classifier_trains_and_entropises():
    X, y, _ = make_dataset(n=40)
    clf = MCClassifier(hidden=(32, 32), max_iter=80, backend="sklearn")
    clf.fit(X, y)
    H = clf.mc_entropy(X[:50], M=6)
    assert H.shape == (50,)
    assert np.all(np.isfinite(H)) and np.all(H >= 0)
    proba = clf.predict_proba(X[:20])
    assert proba.shape == (20, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_warm_start_continues_training():
    X, y, _ = make_dataset(n=40)
    clf = MCClassifier(hidden=(32, 32), max_iter=60, backend="sklearn")
    clf.fit(X, y)
    before = clf.n_epochs_
    clf.fit(X, y, warm_start=True, max_iter=60)
    assert clf.n_epochs_ >= before


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def test_band_masks_are_nested_and_bounded():
    ev = make_eval_grid(n=120, bands=(1, 4), interior_band=8)
    b1, b4 = ev["bands"][1], ev["bands"][4]
    assert b1.sum() > 0
    assert np.all(b4[b1])                       # band 4 contains band 1
    assert 0.01 < b1.mean() < 0.30
    assert ev["interior"].sum() > 0
    assert not np.any(ev["interior"] & b4)


def test_expected_calibration_error_endpoints():
    # perfectly confident and perfectly right -> zero error
    conf = np.array([1.0, 0.0, 1.0, 0.0])
    assert expected_calibration_error(conf, np.array([True, False, True,
                                                      False])) == 0.0
    # fully confident but half wrong -> the error is the error rate
    conf = np.array([1.0, 1.0, 1.0, 1.0])
    correct = np.array([True, False, True, False])
    assert expected_calibration_error(conf, correct) == pytest.approx(0.5)


def test_params_of_counts_weights_and_biases():
    # 2 -> 4 -> 3 : (2*4+4) + (4*3+3) = 12 + 15
    assert params_of((4,)) == 27


# ----------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------
def test_crossing_interpolates_the_envelope():
    x = [1.0, 2.0, 4.0]
    y = [0.0, 0.4, 0.9]
    wc = crossing(x, y, level=0.5)
    assert 2.0 < wc < 4.0
    assert np.isnan(crossing([1.0, 2.0], [0.0, 0.2]))


def test_curve_and_bootstrap_on_synthetic_rows():
    rows = [dict(cap="w32", seed=s, w_over_h=w, recovery=r, freeze=0.0)
            for s in range(3) for w, r in ((1.0, 0.1), (2.0, 0.5), (4.0, 0.9))]
    ux, mean, std = curve(rows, "recovery", "w32")
    assert list(ux) == [1.0, 2.0, 4.0]
    assert np.allclose(mean, [0.1, 0.5, 0.9])
    assert np.allclose(std, 0.0)
    boot = bootstrap_wc(rows, "w32", n_boot=20)
    assert len(boot) > 0
    assert np.isfinite(ci(np.arange(100.0))).all()


# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
def test_repository_relative_outputs():
    assert FIGURES.parent == RESULTS.parent == CACHE.parent
    assert figure("x.png").is_absolute() and result("x.csv").is_absolute()
    assert cache("x.npy").parent.name == ".cache"
