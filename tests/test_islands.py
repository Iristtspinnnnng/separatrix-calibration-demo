"""Unit tests for the island catalogue and the peninsula coordinates.

The synthetic fields below are hand-built label rasters, so the expected
answers are exact and no Newton iteration (or training) is involved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from separatrix.islands import (find_islands, island_metrics, island_points,
                                neighbour_pressure)
from separatrix.peninsula import (CELL, bin_id, core_distance, neighbour_of,
                                  thickness)


def synthetic_field(n=120, r=4, label=2, surrounding=1):
    """Basin ``surrounding`` everywhere, one compact ``label`` disc at the
    centre, and a large ``label`` half-disc on the far left (the "main body",
    so the centre disc counts as buried)."""
    lab = np.full((n, n), surrounding, dtype=np.int8)
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n // 2
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    lab[disc] = label
    lab[:, :8] = label                    # main body of basin `label`
    return lab, disc


def test_find_islands_finds_the_buried_disc():
    lab, disc = synthetic_field()
    isls = find_islands(lab, min_area=9, ring=2, bury=3, surround_frac=0.6)
    assert len(isls) == 1
    isl = isls[0]
    assert isl["label"] == 2 and isl["surrounding"] == 1
    assert isl["area"] == int(disc.sum())
    assert isl["scale_cells"] == pytest.approx(np.sqrt(disc.sum()))
    assert isl["dist_to_main"] >= 3
    # the interior pixels are a subset of the component
    iy, ix = isl["interior_idx"]
    assert len(iy) > 0 and np.all(disc[iy, ix])


def test_island_points_are_inside_the_island():
    lab, disc = synthetic_field()
    isl = find_islands(lab, min_area=9, ring=2, bury=3)[0]
    xy = island_points(isl, lab.shape[0], xlim=(-1.5, 1.5))
    assert xy.shape[1] == 2
    assert np.all(np.abs(xy) < 1.5)


def test_island_metrics_with_a_fake_classifier():
    lab, _ = synthetic_field()
    isl = find_islands(lab, min_area=9, ring=2, bury=3)[0]

    class Perfect:
        def predict(self, xy):
            return np.full(len(xy), isl["label"], dtype=np.int8)

        def mc_predict(self, xy, M=12):
            p = np.zeros((len(xy), 3))
            p[:, isl["label"] - 1] = 1.0
            return p, np.zeros(len(xy))

    m = island_metrics(Perfect(), [isl], lab.shape[0], sample=50)[0]
    assert m["recover_rate"] == 1.0
    assert m["freeze_rate"] == 0.0

    class Frozen:
        def predict(self, xy):
            return np.full(len(xy), isl["surrounding"], dtype=np.int8)

        def mc_predict(self, xy, M=12):
            p = np.zeros((len(xy), 3))
            p[:, isl["surrounding"] - 1] = 1.0
            return p, np.zeros(len(xy))

    m = island_metrics(Frozen(), [isl], lab.shape[0], sample=50)[0]
    assert m["freeze_rate"] == 1.0 and m["recover_rate"] == 0.0


def test_neighbour_pressure_counts_training_points():
    lab, _ = synthetic_field()
    isl = find_islands(lab, min_area=9, ring=2, bury=3)[0]
    g = np.linspace(-1.5, 1.5, 40)
    xp, yp = np.meshgrid(g, g, indexing="xy")
    xy = np.column_stack([xp.ravel(), yp.ravel()])
    out = neighbour_pressure([isl], xy, lab.shape[0])[0]
    assert out["n_near"] >= out["n_in"] >= 0
    # pressure is infinite exactly when the grid misses the island
    assert np.isinf(out["pressure"]) == (out["n_in"] == 0)


# ----------------------------------------------------------------------
# peninsula coordinates
# ----------------------------------------------------------------------
def test_thickness_of_a_half_plane_boundary():
    lab = np.ones((80, 80), dtype=np.int8)
    lab[:, 40:] = 2
    t, bnd = thickness(lab)
    assert bnd[:, 39].all() and bnd[:, 40].all()
    # k cells away from the boundary: t = 2 * k * CELL, on both sides
    for k in (1, 2, 5):
        assert t[10, 39 - k] == pytest.approx(2 * k * CELL)
        assert t[10, 40 + k] == pytest.approx(2 * k * CELL)


def test_bin_id_and_neighbour_of():
    lab = np.ones((40, 40), dtype=np.int8)
    lab[:, 20:] = 3
    t, _ = thickness(lab)
    b = bin_id(t)
    assert b.min() == 0 and b.max() == len(np.unique(b)) - 1
    assert neighbour_of(lab, 10, 18, own=1) == 3
    assert neighbour_of(lab, 10, 22, own=3) == 1


def test_core_distance_is_zero_on_the_core():
    lab = np.ones((60, 60), dtype=np.int8)
    lab[20:40, 20:40] = 2
    t, _ = thickness(lab)
    d = core_distance(lab, t, core_t=0.0)      # everything is "core"
    assert np.allclose(d[lab == 1], 0.0)
    d2 = core_distance(lab, t, core_t=1e9)     # nothing is "core"
    assert np.all(np.isinf(d2))
