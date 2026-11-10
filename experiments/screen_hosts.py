"""Can a sibling Newton map ``z**p - 1`` supply islands over a wider size range?

The narrow island-size range of the p=3 map (3.0-6.3 fine cells, i.e.
0.0028-0.0059 units) is the main limitation of the island experiment; this
script checks the cheapest possible fix first: keep the same dynamics family
and change ``p``.

Measured at 3200x3200 (cell 0.00094 units) with the same catalogue criteria
as ``island_study.py`` (``min_area=9, ring=2, bury=3, surround_frac=0.6``):

    p   n_isl  width (cells)  width (units)      max/median
    3     68    3.00-6.32     0.0028-0.0059        1.91
    4     32    3.00-4.47     0.0028-0.0042        1.45
    5      2    6.40          0.0060              1.00
    6     44    3.00-3.61     0.0028-0.0034        1.20

So no: every map in the family puts its buried islands at 0.003-0.006 units,
a 1.2-1.9x spread.  The reason is structural, not numerical -- with
``N_a(z) = z - (z**3 - a**3)/(3 z**2)`` one has ``N_a(a z) = a N_1(z)``, so
rescaling the whole map is a relabelling of the same field, and the
classifier standardises its inputs: absolute island width is a gauge choice.
The only scale variable the family offers is ``w/h`` (samples across), which
the zoom experiment already sweeps from 1.3 to 133.

A genuinely wide, fixed-content scale series needs a non-self-similar host
(standard-map island chain: island width ``~ sqrt(K)`` in a fixed phase
space, or a perturbed Newton map with a fixed-scale attractor), plus a
redefinition of the ground truth, since a chaotic sea has no basin of
attraction.

Run
---
    python experiments/screen_hosts.py [N]      # N = 3200 reproduces the table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import numpy as np

from separatrix.dynamics import basin_image
from separatrix.islands import find_islands

XLIM = (-1.5, 1.5)


def main(n=3200):
    cell = (XLIM[1] - XLIM[0]) / (n - 1)
    print(f"grid {n}x{n}, cell {cell:.5f} units, "
          f"min_area 9 cells = scale 3 fine cells = {3 * cell:.5f} units\n")
    print(f"{'p':>2} {'t(s)':>6} {'n_isl':>6} {'scale(cells) min..max':>24} "
          f"{'scale(units) min..max':>26} {'max/median':>10}")
    for p in (3, 4, 5, 6):
        t0 = time.perf_counter()
        lab = basin_image(n=n, xlim=XLIM, p=p)
        isls = find_islands(lab, min_area=9, ring=2, bury=3,
                            surround_frac=0.6,
                            labels_to_track=tuple(range(1, p + 1)))
        dt = time.perf_counter() - t0
        if not isls:
            print(f"{p:>2} {dt:6.1f} {0:>6}  (none)")
            continue
        s = np.array([d["scale_cells"] for d in isls])
        q = np.percentile(s, [5, 50, 95, 100])
        print(f"{p:>2} {dt:6.1f} {len(isls):>6} "
              f"{s.min():>10.2f}..{s.max():<10.2f} "
              f"{s.min() * cell:>10.4f}..{s.max() * cell:<12.4f} "
              f"{s.max() / np.median(s):>10.2f}")
        print(f"     percentiles(5/50/95/100) = {np.round(q, 2)} cells; "
              f"burial median "
              f"{np.median([d['dist_to_main'] for d in isls]):.0f} cells")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3200)
