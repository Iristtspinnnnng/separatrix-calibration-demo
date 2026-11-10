# Detailed results

Companion to the [README](../README.md): the full tables, the three times this
repository corrected itself, and the numbers behind every figure. All values are
reproduced by the commands in the README; the raw outputs are in
[`results/`](../results).

Contents
1. [The paper's protocol, in numbers](#1-the-papers-protocol-in-numbers)
2. [Metric definitions and their sensitivity](#2-metric-definitions-and-their-sensitivity)
3. [The five measurement traps](#3-the-five-measurement-traps)
4. [Capacity](#4-capacity)
5. [Islands](#5-islands)
6. [Peninsulas](#6-peninsulas)
7. [The three corrections](#7-the-three-corrections)
8. [Cost of the loop](#8-cost-of-the-loop)
9. [Limitations](#9-limitations)

---

## 1. The paper's protocol, in numbers

What the paper does with the Newton fractal (Sec. 3.1), and what this repository
found when it ran the same protocol on the same map
([`experiments/paper_protocol_check.py`](../experiments/paper_protocol_check.py)):

| step | paper's setting | this repository |
|---|---|---|
| training data | 40 000 uniform initial conditions (200×200), integrated 50 Newton steps | same, 39 986 converged points |
| classifier | NN + dropout, entropy `H(p̄)` (Alg. 1) | MLP (256,256) + MC dropout, 12 stochastic passes |
| acquisition set D₂ | top 10 % of a 400×400 = 160 000-point evaluation grid by entropy | same: 16 000 points |
| generative step | score-based model trained on D₂, 16 000 samples | **not run** — see the inference note in the README |
| evaluation | Chamfer 0.0115, sliced Wasserstein 0.0493 between D₂ and the SGM samples | replaced by a geometric band oracle with ground truth |

How much buried-island structure the protocol ever sees:

```
training points inside islands:      4 of 39 986
islands touched by the training grid: 4 of 68
islands with a selected point:       22 of the 16 000 "separatrix" points (0.14 %)
island cells classified correctly:   63 of 692 (9.10 %) on the 60 buried islands
evaluation points inside islands:    28 of 160 000 (0.0175 %)
```

The last two lines are the ones that matter for the paper's claim: the generative
model is trained *only* on a set in which 22 points sit inside islands, and the
metric compares that set with samples of its own density. Structure that is missing
from the 16k set cannot move Chamfer/Wasserstein, so the metric cannot see the loss.

---

## 2. Metric definitions and their sensitivity

The separatrix has Lebesgue measure zero, so "accuracy on the separatrix" is always
"accuracy inside a band". Three independent choices move the number:

**Band width** (one model, 256×256 evaluation grid, round 5):

```
band (cells)     1      2      4      8     16
fraction      6.1%  10.6%  16.6%  25.2%  36.6%
accuracy     0.503  0.669  0.788  0.860  0.904
```

**Evaluation resolution** (one model, band 1): 0.458 on 400², 0.460 on 401² and
0.461 on 402² — a ±0.3-point effect, since the 1-cell band covers 6.12 % of the grid
in all three cases. Resolution is therefore *second order* for this metric; the band
width (first table) is the choice that matters, and the oracle is the other large
lever: the dynamical proxy covers 29 % of the plane and scores 0.19 precision
against band 1.

**The oracle itself**: the dynamical proxy used by the original loop ("escapes
`|z| < 1.4` within 6 steps") covers 29 % of the plane; against the 1-cell geometric
band it has precision 0.19 and recall 0.90, i.e. it is a 6–8 cell halo. Numbers
computed with it look stable and are not comparable to band-1 numbers.

Entropy separability needs the same care: band 1 vs interior saturates at AUROC
1.000, so the informative contrast is the **hard** one, band 1 vs the rest of the
halo (band 1 vs band 8), which is what `evaluate_on_grid` reports as
`sep_auc_hard`. In the demo it rises 0.870 → 0.926 over five rounds.

---

## 3. The five measurement traps

All five were found by measurement, not by inspection, and each one changed a
conclusion.

### 3.1 Population shift in the in-sample metric (the important one)

The original loop scored `boundary_acc` on the **current training set**, whose
composition changes every round (0 % → 55 % boundary points). With one oracle held
fixed and only the population changed:

```
escape-proxy accuracy on the CHANGING training set : 0.828 -> 0.776  (-0.053)
escape-proxy accuracy on a FIXED uniform population: 0.829 -> 0.892  (+0.063)
```

The same models, the same oracle, opposite signs. Corrected (fixed, independent
400×400 population, geometric bands):

```
round   acc band1   acc band8   acc interior   ECE in band1   sep_AUC(hard)
    1       0.458       0.827          1.000          0.116           0.851
    5       0.518       0.869          1.000          0.077           0.923
```

There is **no calibration collapse** in this 2-D example: accuracy inside the band
rises, ECE falls, and entropy separability rises. The initial "sampling ever finer
breaks the confidence" reading was an artefact of the population drift.

### 3.2 Band-width dependence

See §2. Any statement about separatrix accuracy that does not name the band is not
reproducible.

### 3.3 Sampling density confounded with window content

The first phase diagram zoomed the *window* to sweep `w/h`, which also changed how
much competing structure the window contained. At matched `w/h`, the fixed-content
experiment beats the windowed one by **+0.03 / +0.23 / +0.17** (w32, corresponding to
windows 100 / 200 / 300): "bigger windows are harder" is a *content* effect, not a
density effect.

### 3.4 A training cap silently breaks the horizontal axis

`phase_experiment.py` caps the training set at 40 000 points, but a 300×300 window
has 90 000 cells, so the grid is subsampled and the spacing the classifier actually
sees is `window/√n_train`, not `window/(n−1)`. Recomputing the axis with the
effective spacing (`h_eff = h·(n−1)/√n_train`) is what made windows 200 and 300 agree.
Every `w/h` in this repository now means the effective one.

### 3.5 A single seed is noise in the transition region

The same configuration can give recovery 0.40 with one seed and 0.85 with another
(the outcome is bimodal: the model either carves the island out or it does not).
Single-seed estimates of `w_c` therefore moved by a factor of 4 (12.0 vs 3.1).
Everything reported here is 3–5 seeds with the spread, and `w_c` carries a bootstrap
CI over seeds.

---

## 4. Capacity

`experiments/sweep_capacity.py` (`fig3`): one width/depth sweep, band-1 accuracy
round 1 → round 5 on a fixed 401×401 population. Raw table in
[`results/capacity_sweep.csv`](../results/capacity_sweep.csv).

| config | params | band-1 r1 → r5 | band-8 r1 → r5 | ECE@band1 r1 → r5 |
|---|---|---|---|---|
| (16,16) | 371 | 0.356 → 0.372 | 0.599 → 0.651 | 0.173 → 0.155 |
| (32,32) | 1 251 | 0.359 → 0.387 | 0.628 → 0.685 | 0.163 → 0.139 |
| (64,64) | 4 547 | 0.370 → 0.450 | 0.655 → 0.809 | 0.187 → 0.092 |
| (96,96) | 9 891 | 0.373 → 0.476 | 0.670 → 0.838 | 0.186 → 0.068 |
| (128,128) | 17 283 | 0.385 → 0.484 | 0.691 → 0.848 | 0.164 → 0.066 |
| (192,192) | 38 211 | 0.394 → 0.502 | 0.733 → 0.860 | 0.167 → 0.062 |
| (256,256) | 67 331 | 0.408 → 0.503 | 0.764 → 0.860 | 0.151 → 0.059 |
| (512,512) | 265 731 | 0.423 → 0.505 | 0.801 → 0.864 | 0.148 → 0.071 |
| (256,) | 1 539 | 0.352 → 0.372 | 0.576 → 0.639 | 0.219 → 0.188 |
| (256,256,256) | 133 123 | 0.450 → 0.537 | 0.825 → 0.877 | 0.153 → 0.082 |
| control (64,64), no oversampling | 4 547 | 0.370 → 0.404 | 0.655 → 0.729 | 0.187 → 0.126 |
| control (256,256), no oversampling | 67 331 | 0.408 → 0.460 | 0.764 → 0.833 | 0.151 → 0.102 |

Two controls (`experiments/sweep_matched.py`, `fig4`) ask whether the capacity trend
is just undertraining:

| regime | band-1 acc, round 1 | band-1 acc, round 5 | refinement gain |
|---|---|---|---|
| fixed epoch | 0.356 (371 p) → 0.450 (deep 133k p) | 0.372 … 0.537 | +0.015 … +0.099 |
| train to plateau | 0.363 → 0.461, i.e. 0.446 across widths | 0.375 … **0.551** | +0.012 … +0.090 |
| matched FLOPs | 0.366 → 0.428 (the 512² net, at 38 epochs, gives 0.370) | 0.376 … 0.523 | +0.010 … +0.106 |

* the trend survives both controls, so it is **expressivity**, and it *widens* when
  nobody is undertrained;
* parameters are not free: at matched compute the widest net (512², 38 epochs/round)
  drops to 0.465 with ECE 0.128;
* the best ECE@band1 is 0.035 for the deep net at plateau.

---

## 5. Islands

Catalogue at 3200×3200 (`experiments/island_study.py`, `fig5`; criteria `min_area=9,
ring=2, bury=3, surround_frac=0.6`):

```
islands                    68   (60 buried by more than 10 fine cells)
width                      0.0028 .. 0.0059 units  (3.0 .. 6.3 fine cells)
total area                 0.0088 % of the plane
burial distance            median 62 cells (0.058 units), max 377
elongation                 1.37 .. 4.64 (median 2.3)
```

This is 2–4× below the 0.01 spacing of the demo's 300×300 grid: a 300×300 grid
puts a point on 4 of the 68 islands, 150×150 on 2, 400×400 on 28 and 1200×1200
on 60 (`separatrix/fractal.js` and `experiments/make_demo_data.py` compute this
for any density).

Five arms, same classifier and budget, statistics over the 60 genuinely buried
islands (`results/island_study.json`):

| arm | freeze r1 | freeze r5 | recovered r1 | recovered r5 | above entropy threshold |
|---|---|---|---|---|---|
| jitter @150 | 0.550 | 0.768 | 0.08 | 0.17 | 0.78 |
| jitter @300 | 0.655 | **0.800** | 0.10 | 0.12 | **0.92** |
| jitter @600 | 0.667 | 0.834 | 0.10 | 0.07 | 0.95 |
| pool @300 | 0.655 | 0.775 | 0.10 | 0.13 | 0.92 |
| oracle @300 | 0.655 | 0.799 | 0.10 | **0.07** | 0.92 |

Hypotheses, as tested:

* **H1 (acquisition blind spot): confirmed for the sampler, with a caveat about which
  sampler.** In this repository's `jitter` arm the support *is* the training set
  (`X_hi = X[hi]`), so an island without a training point inside is unreachable
  regardless of its entropy. The paper's single-shot protocol is different and weaker
  in its blind spot only by a factor: D₂ is a quantile of the 400×400 *evaluation*
  grid (0.0075-unit spacing), so it *can* contain island points — measured, 22 of the
  16 000 (0.14 %). The `pool` arm has full-plane support and still recovers 0.10, which
  is why H5 carries the mechanism at these scales.
* **H2 (island entropy never beats the main band): rejected.** 92 % of islands already
  exceed the 85th percentile in round 1. Entropy is not the limiting factor; a
  per-point threshold cannot convert high entropy at an unsampled location into a
  sample.
* **H3 (freezing / neighbour dominance): confirmed as the main outcome**, and it grows
  with refinement (0.66 → 0.80) in every arm.
* **H4 (peninsulas / elongation): not supported for islands.** Spearman(freeze,
  elongation) = −0.02 (p = 0.90); the islands found are compact (1.4–4.6, median 2.3),
  so elongation is tested in the peninsula experiment instead.
* **H5 (feature below the capacity's spatial resolution): confirmed and quantified.**
  See the zoom threshold below.

Zoom experiment (`fig6`), same island, architecture and sample count fixed, only the
relative scale changes:

| island width in training cells | (32,32) | (128,128) | (512,512) |
|---|---|---|---|
| 132.7 | 1.00 | 1.00 | 1.00 |
| 66.3 | 1.00 | 1.00 | 1.00 |
| 33.2 | 1.00 | 1.00 | 1.00 |
| 13.3 | 1.00 | 1.00 | 1.00 |
| 5.0 | **0.00** | 0.10 | 1.00 |
| 1.33 | **0.00** | **0.00** | **0.00** |

and the accuracy away from the island stays at 0.91–0.98 throughout: the failure is
invisible in every aggregate metric.

Freeze phase diagram (5 seeds, 81 configurations, `fig12`,
`results/island.csv`): `w_c = 4.46 / 3.49 / 2.74` in *effective* `w/h` units, 95 %
bootstrap CI `[4.17,4.65] / [3.41,3.58] / [2.00,2.93]`; `w_c ~ params^−0.091`, a gain
of only 1.63× over a 212× parameter range. Clean density experiment (fixed window
content, `fig13`, `results/subsample.csv`): `w_c = 3.12 / 2.78 / 2.35`,
`params^−0.053`, gain 1.33×.

Collapse test (is `w/h` really the only variable?): per (capacity, window),
`w_c = 3.58 / 5.34 / 5.22` (w32), `2.91 / 3.80 / 3.97` (w128), `2.71 / 2.51 / 3.05`
(w512) for windows 100 / 200 / 300. Windows 200 and 300 agree once the axis is
corrected; the tightest window is systematically easier, presumably because it
contains almost no competing structure. So `w/h` is *not* the only variable: how much
other structure shares the domain matters too (up to 0.4 in recovery at matched
`w/h`).

---

## 6. Peninsulas

Definitions (`separatrix/peninsula.py`):

```
t(x) = 2 · dist(x, nearest cell of another basin)          corridor thickness
d(x) = dist(x, nearest thick "core" cell of the same basin) distance from the root
```

Corridor thickness, 3 seeds, 600² grid, `h = 0.005`, `results/thin.csv`:

| t (units) | t/h | w32 | w128 | w512 | frozen (w32 / w128 / w512) |
|---|---|---|---|---|---|
| 0.0013 | 0.25 | 0.347 ± 0.003 | 0.365 ± 0.020 | 0.408 ± 0.003 | 0.345 / 0.342 / 0.307 |
| 0.0060 | 1.21 | 0.389 ± 0.006 | 0.446 ± 0.011 | 0.513 ± 0.017 | 0.305 / 0.258 / 0.226 |
| 0.0119 | 2.38 | 0.467 ± 0.008 | 0.551 ± 0.013 | 0.658 ± 0.016 | ~0.29 / 0.28 / 0.22 |
| 0.0237 | 4.74 | 0.580 ± 0.014 | 0.697 ± 0.019 | 0.849 ± 0.018 | — |
| 0.828 | 165 | 0.980 ± 0.001 | 0.988 ± 0.002 | 0.998 ± 0.001 | — |

Root → tip, thin corridors `t < 0.03`, 3 seeds, 600², `results/geo.csv`:

| distance from core | w32 | w128 | w512 | frozen (w32 / w128 / w512) |
|---|---|---|---|---|
| 0.01–0.03 (root) | 0.663 ± 0.013 | 0.742 ± 0.019 | 0.908 ± 0.019 | 0.173 / 0.086 / 0.037 |
| 0.03–0.08 | 0.333 ± 0.007 | 0.275 ± 0.000 | 0.346 ± 0.007 | 0.314 / 0.395 / 0.367 |
| 0.08–0.25 | 0.038 ± 0.013 | 0.054 ± 0.014 | 0.079 ± 0.007 | 0.554 / 0.524 / 0.500 |
| > 0.25 | 0.000 | 0.000 | 0.000 | 0.436 / 0.467 / 0.467 |

Validation of the `d` coordinate (`--validate`): Euclidean vs local BFS geodesic,
median absolute difference 0.0009 units, 90th percentile 0.0058 (1–6 fine cells),
24 % median relative error, so bins of width ≥ 0.02 are safe and the 0.01 bin is
approximate.

Interpretation: the finger is *sampled* — its root is inside the densely sampled
boundary — so more samples do not help (`×4` density: at most +0.07 in any
thickness bin, and +0.001…+0.01 in the thinnest ones). Capacity helps
at the root and does not propagate down the corridor. The two failure modes of the
README are therefore genuinely different, and the cures are different too.

---

## 7. The three corrections

The numbers in this repository changed three times. The trajectory is kept because
it is the honest version of the result.

```
single seed:                  w_c = 12.0 / 3.5 / 3.4      exponent -0.23
5 seeds + effective axis:     w_c =  4.46 / 3.49 / 2.74   exponent -0.09
clean density experiment:     w_c =  3.12 / 2.78 / 2.35   exponent -0.05
```

1. **Single seed → 5 seeds.** Near the transition the outcome is bimodal, so a single
   seed's `w_c` is noise-dominated (12.0 vs 4.46 for the same architecture).
2. **Nominal → effective spacing.** The 40k training cap subsamples wide windows, so
   `w/h` computed from `window/(n−1)` is wrong; `h_eff = h·(n−1)/√n_train` fixes the
   axis and makes windows 200 and 300 agree.
3. **Windowed → fixed-content density.** The window-based diagram varies density and
   content together; the subsample run varies only the number of training points.
   That is the run that gives `w_c ≈ 3.1` and the flat `params^−0.05` law.

The first estimate over-stated the capacity lever by a factor of ~4 in `w_c` and by a
factor of ~5 in the exponent.

---

## 8. Cost of the loop

The original loop was not buggy, just wasteful: it re-fit from scratch every round on
a training set growing 30 % per round (90k → 117k → 152k → 198k → 257k,
total training volume 9.04× a single fit, 150–230 epochs per round).

Fixes: warm start, a bounded per-round budget (fixed 60k with a growing boundary
share), and one MC loop instead of three (`mc_predict` returns `(mean_proba, H)`).

| step | CPU scikit-learn (24 BLAS threads) | torch/GPU, optimised |
|---|---|---|
| dataset | 1.6 s | 1.4 s |
| fig. 1 | 10–30 min | 4.5 s |
| 5 refinement rounds | 1.5–3 h | 11–13 s |
| total | hours | ~18 s |

On a hybrid-core laptop `OPENBLAS_NUM_THREADS=8` alone took a CPU epoch at n = 90k
from 11.2 s to 2.95 s.

`experiments/compare_loops.py 300` rebuilds the *original* loop and runs both, on the
same GPU, at the same grid:

| loop | 5 rounds, wall clock | boundary acc r1 → r5 | sep-AUC r1 → r5 |
|---|---|---|---|
| original (unbounded, from scratch, 228/152/175/156/221 epochs) | 60 s | 0.884 → 0.797 | 0.939 → 0.857 |
| bounded + warm start (120 extra epochs) | 11 s | 0.869 → 0.767 | 0.939 → 0.879 |

Same diagnostic table, ~5× less wall clock, and 9.04× less training volume
(90k + 117k + 152k + 198k + 257k points versus 5 × 60k).

---

## 9. Limitations

* **2-D toy**: `p = 3` Newton map. No claim about the paper's 3-D Lorenz system or
  about high-dimensional problems.
* **Narrow island scale range**: 3.0–6.3 fine cells at 3200². The *form* of
  `w_c(params)` is only meaningful over the decade the threshold moves;
  `experiments/screen_hosts.py` shows siblings `p = 4, 5, 6` do not widen it, because
  the family is self-similar under `z ↦ az` and the classifier standardises inputs.
* **Euclidean `d`** for the peninsula axis, validated to ~1–6 fine cells; it
  underestimates for winding corridors.
* **No score-based generative model** in the loop: support limitation is an argument,
  not a measurement.
* **Capacity at three widths** (1.3k / 17k / 266k) for the phase diagrams, with a
  fixed epoch budget (mitigated by the plateau control).
* **The bounded-budget schedule** is this repository's design choice; freezing grows
  with rounds partly because the boundary share grows, and the original unbounded
  loop never drops uniform points.
* **Residual variance**: in the transition region `w_c` is good to about ±0.2.
