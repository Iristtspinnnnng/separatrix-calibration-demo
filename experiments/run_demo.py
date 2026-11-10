"""One command to reproduce the calibration-demo numbers.

    python experiments/run_demo.py            # 300x300 grid (~20 s on a 4060)
    python experiments/run_demo.py 150        # smaller grid, ~5 s

What it does
------------
1. labels a uniform 300x300 grid of initial conditions with the Newton map
   (the DynamicalSystems.jl example, and the paper's Sec. 3.1 benchmark);
2. trains the MC-dropout classifier and draws basins + entropy
   (a reproduction of arXiv:2608.14743, Fig. 1);
3. runs five rounds of entropy-driven refinement, scoring every round twice:
   in-sample (the confounded number) and on a fixed, independent 400x400
   population with the geometric fractal bands (the number to trust);
4. writes ``figures/fig1_basins_and_entropy.png``,
   ``figures/fig2_calibration_collapse.png`` and
   ``results/demo_diagnostics.json``.

The JSON is what the interactive demo in ``docs/`` reads, so re-running this
script updates the demo's numbers too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import time

# the diagnostic banners are ASCII, but the console encoding on Windows is not
try:                                            # pragma: no cover
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from separatrix.classifier import backend_report
from separatrix.dataset import make_dataset
from separatrix.evaluate import make_eval_grid
from separatrix.paths import figure, result
from separatrix.plotting import (figure_basins_and_entropy,
                                 figure_calibration_collapse)
from separatrix.refine import print_report, run_iterative_refinement


def main(n_grid=300):
    t_all = time.perf_counter()
    label, detail = backend_report()

    print("=" * 78)
    print(" demo: arXiv:2608.14743 on the Newton fractal, plus the")
    print("       in-sample vs fixed-population measurement trap")
    print(f" classifier backend: {label}  ({detail})")
    print("=" * 78)

    t0 = time.perf_counter()
    print(f"\n[1/4] generating dataset (Newton f(z)=z^3-1, {n_grid}x{n_grid} "
          f"grid) ...")
    X, y, grid = make_dataset(n=n_grid)
    print(f"      -> {len(X)} labelled points ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    print("\n[2/4] plotting basins + entropy (paper Fig.1 reproduction) ...")
    figure_basins_and_entropy(X, y, grid)
    print(f"      -> step 2 done ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    print("\n[3/4] running 5 rounds of iterative refinement ...")
    # fixed, independent population for the fair metrics: the separatrix has
    # measure zero, so accuracy is measured inside bands of 1 and 8 cells
    ev = make_eval_grid(n=400, bands=(1, 2, 4, 8), interior_band=16)
    print(f"      eval grid: {ev['n']}x{ev['n']} = {len(ev['xy'])} points; "
          f"band1 = {100 * ev['bands'][1].mean():.1f}% of it")
    log = run_iterative_refinement(X, y, grid[0], grid[1], rounds=5,
                                   eval_set=ev)
    print(f"      -> step 3 done ({time.perf_counter() - t0:.1f}s)")

    print("\n[4/4] printing diagnostic tables ...")
    print_report(log)

    figure_calibration_collapse(log)
    with open(result("demo_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(f"\nDONE in {time.perf_counter() - t_all:.1f}s.  files written:")
    for p in (figure("fig1_basins_and_entropy.png"),
              figure("fig2_calibration_collapse.png"),
              result("demo_diagnostics.json")):
        print(f"    {os.path.relpath(p, Path(__file__).resolve().parents[1])}")


if __name__ == "__main__":
    main(n_grid=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
