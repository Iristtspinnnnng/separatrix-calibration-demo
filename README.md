# Separatrix calibration demo

An independent, reproducible **audit of one step** in

> E. R. Crabtree, D. G. Giovanis, A. Georgiou, **G. Datseris**, I. G. Kevrekidis (2026),
> *Generative Learning of Separatrices*, [arXiv:2608.14743](https://arxiv.org/abs/2608.14743), Sec. 3.1

run on the paper's own Newton-fractal benchmark (`f(z) = z³ − 1`). The step under
scrutiny is the *entropy-driven refinement*: train a classifier on labelled basin
data, read its MC-dropout entropy, keep the high-entropy points, train a generative
model on them, resample, repeat.

**What we find.** Two failure modes of that chain, with different mechanisms,
different cures, and both invisible to the metric the paper uses for this benchmark:

| | **islands** (buried basins) | **peninsulas** (thin basins) |
|---|---|---|
| mechanism | **not reached** — samples come from a fixed grid (or, in a jitter-style loop, from the training set itself), so an island below that spacing is essentially never proposed | **reached but not resolved** — sampled densely at the root, lost along the corridor |
| criterion (measured) | `w/h ≈ 2.4–3.1` (feature width / sample spacing) | `t/h ≲ 2` **and** distance from the thick core `> 0.03` |
| effect of a 4× denser grid | the axis *is* `w/h` | ≤ `+0.07` in every thickness bin |
| effect of 200× more parameters | `w_c` moves by 1.3× only (`w_c ~ params^−0.05`) | lifts the root from 0.66 to 0.91, the tip not at all |
| what is actually missing | a sample in the right place **and** enough resolution to represent it (both are needed — see below) | local *resolution* to represent it |

Both conditions are needed; neither alone is enough (§[Necessary vs sufficient](#necessary-vs-sufficient-conditions)).

---

## What this is, and what it is not

* It **is** an audit of the sampling → classifier → generative chain, using the
  paper's benchmark, the paper's protocol, and metrics with known ground truth.
  The criticism lands on *"before you add iteration, look at these two failure
  modes"*, not on *"your current loop is wrong"*.
* It **is not** a claim that the paper's results are wrong. Its own wording already
  anticipates most of what is quantified here: the note after Algorithm 2 lists
  expressivity *"at the scale of interest"*, class balance, and strategic
  initial-condition selection as conditions for the approximation to be reliable.
* It **is not** a test of a score-based generative model. There is no SGM in this
  loop; it is replaced by three substitute arms (jitter, plane-wide candidate pool,
  oracle). Everything about *"the generative model's support"* is therefore an
  **inference from a density-model argument, not a measurement** — see
  [Measured vs inferred](#measured-vs-inferred).
* It **is** a set of measured facts about how much structure a 2-D classifier can
  recover at a given sampling density and capacity, and how easy it is to report the
  wrong sign while doing it.

The most useful single number: **22 of the paper's 16 000 "separatrix" points
(0.14 %) fall inside a buried island**, and the classifier recovers **9.1 %** of the
buried-island cells — while the paper's Fig. 1 (reproduced as
[`figures/fig1_basins_and_entropy.png`](figures/fig1_basins_and_entropy.png)) looks
like a success on every panel.

---

## Headline results

1. **A metric trap, not a calibration collapse.** The refinement loop really does
   make the training set boundary-dominated (0 % → 55 % of points). Scored
   *in-sample*, separatrix accuracy falls from 0.869 to 0.779 — the intuitive
   "oversampling destroys the model" story. Scored on a **fixed independent
   population**, the same models *improve*: band-1 accuracy 0.458 → 0.518, band-8
   0.827 → 0.869, calibration error inside the band 0.116 → 0.077. With one oracle
   held fixed, the sign flips exactly (`−0.053` in-sample vs `+0.063` fixed
   population). See [`experiments/metric_bias.py`](experiments/metric_bias.py).
2. **"Separatrix accuracy" is always a band-width statement.** The separatrix has
   Lebesgue measure zero, so accuracy is measured inside a band of `k` cells. On one
   model, round 5: `k = 1, 2, 4, 8, 16` cells → the band covers 6.1 / 10.6 / 16.6 /
   25.2 / 36.6 % of the plane and the accuracy is 0.503 / 0.669 / 0.788 / 0.860 /
   0.904. Evaluation resolution is second order by comparison (same model, band 1:
   0.458 / 0.460 / 0.461 on 400², 401², 402²), but the band width is not: a
   "boundary accuracy" quoted without `k` is not a number.
3. **Buried islands are unreachable, and entropy does not help.** 68 buried islands
   exist at 3200², all tiny (0.0028–0.0059 units, 3.0–6.3 fine cells, 0.0088 % of
   the plane). **92 % of them already exceed the 85th entropy percentile** in round
   1, yet only 4 of 68 contain a training point at 300² (2 at 150², 28 at 400²). The sampler
   picks the top-*k* **training points** by entropy and jitters them, so its support
   *is* the training set.
4. **Even an oracle cannot fix it.** Putting new samples deliberately *inside* the
   islands leaves recovery at 0.10 → 0.07 over five rounds, while freezing rises
   0.66 → 0.80 in every arm. At this scale the binding constraint is the model's
   spatial resolution, not where the samples are.
5. **The resolution criterion is `w/h ≈ 2.4–3.1`, and capacity barely moves it.**
   With window content fixed and only the sample count varied: `w_c = 3.12 / 2.78 /
   2.35` samples across the island for 1.3k / 17k / 266k parameters
   (`w_c ~ params^−0.053`, i.e. ×212 parameters buys ×1.33 in `w_c`).
6. **Peninsulas fail differently.** Thin fingers are already sampled at the root, so
   density does almost nothing (`×4` → ≤ `+0.07`); capacity lifts the root
   (0.663 → 0.908) but not one bin further out (0.333 → 0.346). Recovery drops to
   chance as soon as the corridor is thinner than `t/h ≈ 2` **and** further than
   ≈0.03 units from the thick core.
7. **A 9× cost reduction is orthogonal to all of the above.** The original loop was
   not wrong, just wasteful: growing the training set 30 % per round (90k → 257k
   points, 9.04× the training volume of a single fit, 150–230 epochs each) and
   re-fitting from scratch. Warm starts + a bounded per-round budget + one MC loop
   instead of three: **1.5–3 h on CPU → 11 s on a GPU** for the five rounds, with
   the same diagnostic table.

---

## Measured vs inferred

| claim | status | evidence |
|---|---|---|
| the in-sample metric moves opposite to the fixed-population metric | **measured** | `experiments/metric_bias.py`, Table A vs B, [`results/demo_diagnostics.json`](results/demo_diagnostics.json) |
| band-width and grid-resolution dependence of "boundary accuracy" | **measured** | `separatrix/evaluate.py`, `experiments/sweep_capacity.py` |
| islands are missed by a uniform grid, and stay missed under refinement | **measured** | `experiments/island_study.py`, `paper_protocol_check.py` |
| 92 % of islands exceed the entropy threshold | **measured** | same, fig. 5c |
| oracle placement does not rescue them | **measured** | same, five-arm table |
| `w/h` is the governing scale, `w_c ~ params^−0.05` | **measured** (5 seeds, bootstrap CI) | `experiments/phase_experiment.py` + `analyse_*.py` |
| peninsulas need `t/h ≳ 2` and root proximity | **measured** (3 seeds) | `experiments/peninsula_thin.py`, `peninsula_geo.py` |
| **the SGM's support is limited to the D₂ neighbourhood** | **inferred, not measured** | D₂ is a per-point quantile of a *fixed* 400² evaluation grid (16 000 of 160 000 points, of which 22 lie on an island); the SGM sees only D₂, so it has essentially no training mass where an island is. Substituted here by jitter / pool / oracle arms. |
| **the Wasserstein/Chamfer numbers of Table 1 are self-referential** | **inferred from the protocol** | both sets are drawn from the same band; missing structure cannot move either marginal. Measuring it needs the geometric oracle in `separatrix/evaluate.py`. |

### Open questions (PRs welcome)

* Replace the three substitute arms with a **real SGM** (score-based model on D₂,
  resample, evaluate on the geometric band oracle). This is the one inference above
  that a small amount of code would turn into a measurement.
* The island catalogue of the `p = 3` map has a **narrow size range** (3.0–6.3 fine
  cells); `experiments/screen_hosts.py` shows sibling maps (`p = 4, 5, 6`) do not
  widen it, because the family is self-similar under `z ↦ a z` and the classifier
  standardises its inputs. A **non-self-similar host** (standard-map island chain,
  `w ~ √K`; or a Hénon window) is needed for a scale law spanning a decade.
* `w_c` was measured with a **bounded-budget** loop whose boundary share grows to
  60 % by round 5. Whether the criterion itself changes under the paper's unbounded
  schedule is not measured here (`experiments/compare_loops.py` checks the diagnostic
  table, not `w_c`).
* Everything is 2-D (`p = 3` Newton); no claim is made about the paper's 3-D Lorenz
  case or about high-dimensional systems.

---

## Quick start

```bash
pip install -r requirements.txt          # numpy scipy scikit-learn matplotlib
pip install torch                        # optional: GPU backend, 10x faster
python experiments/run_demo.py           # ~20 s on a 4060, ~2 min on CPU
python -m pytest tests -q                # 25 unit tests, ~3 s
```

The demo prints both diagnostic tables, writes
`figures/fig1_basins_and_entropy.png` (a reproduction of the paper's Fig. 1),
`figures/fig2_calibration_collapse.png` (the metric trap), and
`results/demo_diagnostics.json`. No compilation, no Julia, no downloads.

Requires Python ≥ 3.9. The committed numbers and figures were produced with
**Python 3.13.14** — numpy 2.3.2, scipy 1.15.2, scikit-learn 1.7.0,
matplotlib 3.10.1, torch 2.9.1+cu130, Pillow 11.1.0, pytest 9.2.0, on an
RTX 4060 laptop GPU. On a machine where `python` is older than 3.13, run the
same commands with an explicit interpreter (`python3.13 …`).

`classifier.MCClassifier(backend="torch" | "sklearn" | "auto")` picks the GPU when
torch+CUDA are present; the scikit-learn fallback applies the same Bernoulli mask to
the last hidden layer (MC dropout, Gal & Ghahramani 2016), so entropies are
comparable across backends.

### Interactive demo

[`docs/index.html`](docs/index.html) is a single page, no build step, no server, no
external libraries: open it in a browser (or serve `docs/` on GitHub Pages). It
shows the live Newton fractal at any zoom, the 68 buried islands, a uniform sampling
grid you can densify (and how many islands it actually hits), the classifier's
prediction at a controlled `w/h`, and the frozen-vs-recovered readouts. Numbers come
from the committed measurement files; the fractal and the sampling grid are computed
in the browser. See [`docs/README.md`](docs/README.md).

---

## The measurement problem

The separatrix of the `p = 3` Newton map is the Julia set of
`z ↦ z − (z³−1)/(3z²)`: a fractal of **Lebesgue measure zero**. No finite sample
lands on it, so any "accuracy on the separatrix" is accuracy inside a band, and the
band width and the evaluation resolution must be quoted or the number is not
comparable.

Two oracles live in this repository, and mixing them is the classic way to get a
wrong sign:

| oracle | definition | effective width |
|---|---|---|
| `separatrix.refine.true_boundary_mask` | a point is "near the separatrix" if its Newton iterate leaves `|z| < 1.4` within 6 steps | implicit, covers **29 %** of the plane; vs the 1-cell band: precision 0.19, recall 0.90 → a **6–8 cell halo** |
| `separatrix.evaluate.make_eval_grid` | label a grid by the dynamics, then call a cell "band `k`" if it is within `k−1` cells of a label change | explicit: `k = 1, 2, 4, 8, 16` → 6.1, 10.6, 16.6, 25.2, 36.6 % |

All numbers in this README are computed with the geometric band; the dynamical proxy
is kept only for the cacheable in-sample diagnostic that produced the trap.

---

## Result 1 — the refinement loop and the metric trap

Five rounds of the paper's step on a 300×300 = 90k-point grid (60k per round after
the bounded-budget fix). Same models, two populations:

Table A — **in-sample**, population re-drawn every round (confounded):

```
 round  n_train    bnd%   balance boundary_acc interior_acc confidence      gap  sep_AUC
     1    60000     0.0      0.95        0.869       0.993      0.942    0.073    0.939
     5    60000    55.0      0.95        0.779       0.959      0.823    0.045    0.876
```

Table B — **fixed, independent 400×400 population**, geometric bands:

```
 round    acc band1    acc band8   acc interior    ECE in band1      sep_AUC
     1        0.458        0.827          1.000           0.116        0.851
     5        0.518        0.869          1.000           0.077        0.923
```

Isolated with one oracle held fixed (`experiments/metric_bias.py`):

```
escape-proxy accuracy on the CHANGING training set : 0.828 -> 0.776  (-0.053)
escape-proxy accuracy on a FIXED uniform population: 0.829 -> 0.892  (+0.063)
```

The training set really does become boundary-dominated, and the model really does
get better at the separatrix; only the *population* of the in-sample metric changes.
A calibration collapse was the hypothesis; the measurement says otherwise, and the
measured ECE *inside the band* falls from 0.116 to 0.077.

### Capacity is a floor, and half the gain is just more training

`experiments/sweep_capacity.py` (`fig3`), band-1 accuracy round 1 → round 5 on the
fixed population:

| config | params | band-1 r1 → r5 | band-8 r1 → r5 | ECE@band1 r1 → r5 |
|---|---|---|---|---|
| (16,16) | 371 | 0.356 → 0.372 | 0.599 → 0.651 | 0.173 → 0.155 |
| (64,64) | 4 547 | 0.370 → 0.450 | 0.655 → 0.809 | 0.187 → 0.092 |
| (128,128) | 17 283 | 0.385 → 0.484 | 0.691 → 0.848 | 0.164 → 0.066 |
| (256,256) | 67 331 | 0.408 → 0.503 | 0.764 → 0.860 | 0.151 → 0.059 |
| (512,512) | 265 731 | 0.423 → 0.505 | 0.801 → 0.864 | 0.148 → 0.071 |
| (256,256,256) | 133 123 | 0.450 → 0.537 | 0.825 → 0.877 | 0.153 → 0.082 |
| control (64,64), no oversampling | 4 547 | 0.370 → 0.404 | 0.655 → 0.729 | — |
| control (256,256), no oversampling | 67 331 | 0.408 → 0.460 | 0.764 → 0.833 | — |

* capacity sets the floor (0.356 → 0.423 for 371 → 266k parameters), and depth is
  the efficient direction at equal parameter count;
* the refinement gain peaks at *moderate* capacity (+0.108 at ~10–38k parameters)
  and shrinks for very wide nets — a large net already fits much of the fine
  structure;
* half of the band-1 gain is **not** oversampling: the same epoch budget without ever
  adding boundary points gives +0.034 (64²) / +0.052 (256²), versus +0.080 / +0.095;
* the sign flips with band width (`fig3c`): on the 16-cell halo the gain shrinks to
  +0.04…+0.05. Any claim of the form "refinement improves/damages separatrix
  accuracy" must name the band.

`experiments/sweep_matched.py` (`fig4`) repeats it under two stricter budgets
(train-to-validation-plateau, and equal `epochs · samples · params`):
the capacity trend is **expressivity, not undertraining** (the width spread *widens*
from 0.356 → 0.423 to 0.363 → 0.446 at round 1, best result 0.551 from
(256,256,256)), but parameters are
not free — at matched compute the widest net (512², forced to 38 epochs/round) drops
to 0.465 with ECE 0.128. The sweet spot is ~1e4–1e5 parameters.

---

## Result 2 — islands: the acquisition never arrives

`islands.py` finds buried islands as connected components of one basin whose
surrounding ring is dominated by a different basin and which sit far from their own
basin's main body. At 3200×3200 (`fig5`, `fig12`):

```
island width   0.0028 .. 0.0059 units     (3.0 .. 6.3 fine-grid cells)
total area     0.0088 % of the plane
burial         median 62 cells (0.058 units), max 377 cells
count          68 (60 buried by more than 10 cells)
```

That is 2–4× *below* the 0.01 spacing of the demo's 300×300 training grid, at which
density only 4 of the 68 islands contain a training point. Three sampling arms, same
classifier, same 60k budget, statistics over the 60 buried islands (`fig5`):

| arm (initial grid) | freeze r1 | freeze r5 | recovered r1 | recovered r5 | above entropy threshold |
|---|---|---|---|---|---|
| jitter @150 | 0.55 | 0.77 | 0.08 | 0.17 | 0.78 |
| jitter @300 (demo) | 0.66 | **0.80** | 0.10 | 0.12 | **0.92** |
| jitter @600 | 0.67 | 0.83 | 0.10 | 0.07 | 0.95 |
| pool @300 | 0.66 | 0.78 | 0.10 | 0.13 | 0.92 |
| **oracle @300** | 0.66 | 0.80 | 0.10 | **0.07** | 0.92 |

Three readings:

1. **92 % of the buried islands already exceed the entropy threshold** in round 1, so
   "the island's entropy was not high enough" (H2) is rejected. The sampler picks the
   top-15 % of *training points* by entropy and jitters them; `X_hi = X[hi]` means an
   island with no training point inside is unreachable **by construction**.
2. **Freezing is the dominant outcome, and it gets worse with refinement**
   (0.66 → 0.80) in every arm, including the oracle.
3. **The oracle does not rescue them** (0.10 → 0.07): at this scale the bottleneck is
   the representation, not where the samples are.

### The criterion: samples across the feature

The zoom experiment (`fig6`) holds the island fixed in the plane and zooms the
training window, so the island spans a controlled number of cells while the
architecture and sample count stay put. Same island, recovered perfectly while it
spans **≥ 13.3 cells** ((32,32) and (128,128)) or **≥ 5.0 cells** ((512,512)), frozen
solid below that; a 212× increase in parameters moves the threshold only 2.7×, and
"accuracy far from the island" stays at 0.91–0.98 the whole time.

The clean density experiment (`fig13`: fixed window content, only the number of
training points changes, 5 seeds, bootstrap CI) gives the criterion and its capacity
dependence:

| capacity | params | `w_c` | 95 % CI |
|---|---|---|---|
| (32,32) | 1 251 | **3.12** | [3.03, 3.17] |
| (128,128) | 17 283 | **2.78** | [2.69, 2.89] |
| (512,512) | 265 731 | **2.35** | [2.12, 2.66] |

`w_c ~ params^−0.053`: ×212 parameters buys ×1.33 in threshold. The curve is steep —
`w/h = 1.8 → 0.00 / 0.08 / 0.24`, `2.5 → 0.05 / 0.34 / 0.56`,
`3.5 → 0.72 / 0.86 / 0.86` (32², 128², 512²) — so the failure is confined to a narrow
band of relative scales, which is exactly what makes it dangerous: the aggregate
accuracy is unaffected everywhere else.

The earlier window-based phase diagram (`fig8`, `fig12`) gave `w_c = 12.0 / 3.5 / 3.4`
with one seed and a **mis-specified axis**; with 5 seeds and the effective spacing
(`h_eff = h·(n−1)/√n_train`, because the 40k cap subsamples a 300² window) it is
4.46 / 3.49 / 2.74 and the exponent drops from −0.23 to −0.09. That correction is
part of the story: see [`docs/RESULTS.md`](docs/RESULTS.md#7-the-three-corrections).

### What the paper's own protocol sees

`experiments/paper_protocol_check.py` runs the paper's Sec. 3.1 settings — 40 000
uniform initial conditions (200×200), evaluate 160 000 (400×400), keep the top 10 %
by entropy = 16 000 points for the generative model:

```
training points inside islands:            4 of 39 986
islands touched by the training grid:     4 of 68
"separatrix" (top-10 %) points inside an island: 22 of 16 000  (0.14 %)
islands with at least one selected point: 22 of 68
island cells predicted as their own basin: 9.1 %  (63 of 692, on the 60 buried islands)
```

The paper evaluates the generative step by Chamfer (0.0115) and sliced Wasserstein
(0.0493) distance **between that 16k set and the SGM samples**. Both sets live on the
same main band, so structure that is missing from the 16k set cannot move either
number: the metric is blind by construction, not by accident.

---

## Result 3 — peninsulas: sampled, but not resolved

A constructed smooth flow cannot supply a peninsula: in a 2-D gradient flow the
basin boundary is a smooth curve, so a third attractor placed *on* it owns a strip
and one placed *inside* a basin owns a buried island (both attempts are in the
repository, `island_landscape.py` / `peninsula_v2.py`, with the reason documented).
Genuine fingers need a folded or fractal boundary, so this experiment runs on the
paper's own label field with a parameter-free geometric definition:

```
t(x) = 2 · distance(x, nearest cell of another basin)      corridor thickness
d(x) = distance(x, nearest "core" cell of the same basin)  position along the finger
```

`experiments/peninsula_thin.py` (`fig10`), 3 seeds, 600² grid, `h = 0.005`:

| t (units) | t/h | w32 | w128 | w512 | frozen into neighbour (w32 / w128 / w512) |
|---|---|---|---|---|---|
| 0.0013 | 0.25 | 0.347 | 0.365 | 0.408 | 0.345 / 0.342 / 0.307 |
| 0.0060 | 1.21 | 0.389 | 0.446 | 0.513 | 0.305 / 0.258 / 0.226 |
| 0.0119 | 2.38 | 0.467 | 0.551 | 0.658 | ~0.29 / 0.28 / 0.22 |
| 0.0237 | 4.74 | 0.580 | 0.697 | 0.849 | (no neighbour in the window) |
| 0.828 | 165 | 0.980 | 0.988 | 0.998 | — |

(`recovery = 1/3` is chance for three basins.)

`experiments/peninsula_geo.py` (`fig11`) adds the root→tip axis, thin corridors
(`t < 0.03`) binned by distance from the thick core:

| distance from core (units) | w32 | w128 | w512 | frozen (w32 / w128 / w512) |
|---|---|---|---|---|
| 0.01–0.03 (**root**) | 0.663 | 0.742 | **0.908** | 0.173 / 0.086 / **0.037** |
| 0.03–0.08 | 0.333 | 0.275 | 0.346 | 0.314 / 0.395 / 0.367 |
| 0.08–0.25 | 0.038 | 0.054 | 0.079 | 0.554 / 0.524 / 0.500 |
| > 0.25 | 0.000 | 0.000 | 0.000 | 0.436 / 0.467 / 0.467 |

* **only the root survives**; one bin further out the classifier is at chance and the
  neighbouring basin takes 31–40 % of the cells;
* **capacity sharpens the root but does not propagate it**: 0.663 → 0.908 at the root,
  0.333 → 0.346 one bin out;
* **density helps mostly at the root** (grid 150 → 600: +0.03…+0.10 in the root bin,
  while beyond the root it is within ±0.06 and mostly ≤ 0.03, and exactly 0 in the
  tip bins), because the finger's root is sampled by construction — there is no
  reachability problem to fix.

`d` is a Euclidean distance; `peninsula_geo.py --validate` compares it with a local
BFS geodesic: median absolute difference 0.0009 units, 90th percentile 0.0058
(1–6 fine cells), which is why only bins ≥ 0.02 wide are used.

---

## Necessary vs sufficient conditions

The island result is easy to mis-state. "Send samples to the right place" is the cure
that first suggests itself for an unreachable feature, yet the oracle arm does exactly
that and does not help. Both statements are right once the two conditions are
separated:

| | islands | peninsulas |
|---|---|---|
| **necessary** | a sampling support **finer than the feature**: the paper's D₂ comes from a 400² grid (0.0075 units) and already contains 22 island points, while a jitter loop's support is the 40k training set. A candidate pool or an oracle removes the limit entirely | *(already satisfied)* the root is densely sampled |
| **sufficient** | local resolution `w/h ≳ 2.4–3.1`, i.e. capacity matched to the feature scale | capacity / representation: `t/h ≳ 2` **and** root proximity `d ≲ 0.03` |
| what the measurement shows | oracle ⇒ necessary ✓, sufficient ✗ → still frozen (0.10 → 0.07) | necessary ✓ by construction, sufficient ✗ → lost along the corridor |
| practical reading | a reachability fix without a capacity fix changes nothing measurable at these scales | a capacity fix works only near the root; there is no density fix |

So "send samples to the right place" is a *precondition*, not a cure; the sentence to
quote is this table, not the earlier shorthand.

---

## What this suggests for the pipeline

Framed as suggestions to the authors, offered with the measurements that motivate
each:

1. **Give the fractal example a ground-truth metric.** Replace (or supplement) the
   D₂-vs-SGM comparison with coverage / Chamfer / Hausdorff against a high-resolution
   label field, i.e. the geometric band oracle in [`separatrix/evaluate.py`](separatrix/evaluate.py).
   Minimum effort, and it makes the island/peninsula failures visible.
2. **Let the acquisition reach the islands.** In the paper's protocol D₂ is a quantile of
   the 400² *evaluation* grid — 16 000 of 160 000 points — and an island is 3–6 cells of
   a 3200² raster, i.e. below that grid's 0.0075-unit spacing: measured, 22 of the
   16 000 selected points (0.14 %) lie on an island. A jitter-style iterative version
   (`X_hi = X[hi]`) has a strictly smaller support — the training set — and cannot
   reach an island at all. Scoring the classifier's entropy on a dense candidate pool
   (the `pool` arm) restores reachability at almost no cost; the caveat is that at
   these island scales it still does not help, which is what points to the resolution
   limit rather than to the acquisition.
3. **Do not allocate the acquisition budget by per-point quantile.** Top-10 % by
   *point count* gives an island covering 1e-4 of the plane single-digit samples even
   when 100 % of its points exceed the threshold (measured: 22 of 16 000). Allocate
   over connected components of the high-entropy level set instead (or apply
   non-maximum suppression).
4. **Report the effective resolution.** `w_c ~ params^−0.05` for islands and
   `t/h ≳ 2` for peninsulas are cheap, measurable resolution criteria; "finite
   networks smooth out detail" is not.
5. **Report coverage per connected component, not only overall.** The paper already
   has ground truth for the Lorenz case (SCIGMA) and reports coverage, Chamfer and
   Hausdorff; a global coverage fraction asks what part of the true manifold lies
   near the prediction, so a small, far-away component can be missing while the
   headline coverage stays high. Splitting the manifold into components and
   reporting coverage for each closes that hole cheaply.
6. **Name the two failures differently.** "Cannot be reached" and "cannot be
   resolved" have different diagnoses and different cures, and the paper's
   limitations section would be sharper if it said which one applies.

---

## Repository map

```
separatrix/              the reusable package (import it; no side effects)
  dynamics.py            Newton map; vectorised basin labelling
  dataset.py             labelled uniform grids (stage 1)
  classifier.py          MLP + MC-dropout entropy (stage 2), torch or sklearn
  refine.py              the entropy-driven refinement loop (stage 3)
  evaluate.py            geometric fractal-band oracle, ECE, capacity accounting
  islands.py             buried-island catalogue and per-island metrics
  peninsula.py           corridor thickness, distance-from-core, strata
  stats.py               seed means, freeze boundary, bootstrap CIs, CSV loaders
  plotting.py            the paper's Fig. 1 reproduction and the metric-trap figure
  paths.py               repository-relative figures/, results/, .cache/

experiments/             the runnable studies (all CLI, all write to figures/results)
  run_demo.py            the 20-second entry point (Tables A and B)
  metric_bias.py         in-sample vs fixed-population, one oracle
  sweep_capacity.py      width/depth sweep + no-oversampling control      (fig3)
  sweep_matched.py       plateau and matched-FLOPs control                (fig4)
  island_study.py        buried islands x 3 sampling arms                 (fig5)
  island_zoom.py         same island at controlled relative scale         (fig6)
  island_phase.py        single-seed freeze phase diagram (historical)    (fig8)
  phase_experiment.py    seed-averaged, resumable phase diagram (island/peninsula)
  paper_protocol_check.py  what the paper's own 16k-point protocol sees
  analyse_island_server.py 5-seed w_c, CIs, collapse test                 (fig12)
  analyse_subsample.py   clean density run + window-content confound      (fig13)
  peninsula_thin.py      corridor thickness on the fractal                (fig10)
  peninsula_geo.py       root-to-tip axis, Euclidean vs geodesic check    (fig11)
  island_landscape.py    constructed well+satellite testbed (scale sweep)
  landscape_scan.py      its figure: a scale-tunable basin and its limits  (fig7)
  peninsula_v2.py        constructed-flow attempt, documented dead end
  screen_hosts.py        do sibling maps (p=4,5,6) widen the island range?
  compare_loops.py       original unbounded loop vs the optimised one
  make_demo_data.py      regenerates the assets of the interactive demo

tests/                   25 unit tests (pytest), ~3 s
docs/                    the interactive demo + the detailed result tables
figures/                 publication figures (committed, regenerable)
results/                 measurement tables, CSV + JSON (committed)
```

---

## Reproducing the numbers

Each command writes into `figures/` and `results/` relative to the **repository
root**, so it can be run from anywhere. Times are from one RTX 4060 laptop GPU;
a CPU-only run of the same scripts is 5–20× slower. `run_demo.py` is seeded, so its
tables are reproducible on a given backend and differ by at most a couple of points
between backends (torch/GPU vs scikit-learn/CPU).

```bash
# entry point: Tables A and B, fig1, fig2, demo_diagnostics.json       (~20 s)
python experiments/run_demo.py

# the metric trap, isolated                                       (~25 s)
python experiments/metric_bias.py

# capacity sweep and its two budget controls              (~3 min / ~12 min)
python experiments/sweep_capacity.py
python experiments/sweep_matched.py

# islands: catalogue, sampling arms, resolution threshold      (~3 min / ~1 min)
python experiments/island_study.py
python experiments/island_zoom.py
python experiments/paper_protocol_check.py                            (~2 min)

# freeze boundary, seed-averaged and resumable (5 seeds)      (~70-90 min total)
python experiments/phase_experiment.py island --seeds 0 1 2 3 4 \
       --windows 100 200 300 --zooms 2 3 5 8 14 25 50 70 100 \
       --caps w32 w128 w512 --out results/island.csv
python experiments/phase_experiment.py subsample --seeds 0 1 2 3 4 \
       --windows 300 --zooms 20 --caps w32 w128 w512 \
       --fractions 1.0 0.5 0.25 0.125 0.0625 0.03125 0.015625 \
       --out results/subsample.csv
python experiments/analyse_island_server.py results/island.csv          (fig12)
python experiments/analyse_subsample.py results/subsample.csv           (fig13)

# peninsulas, on the paper's own fractal                        (~5 min / ~10 min)
python experiments/peninsula_thin.py --sweep --grid 150 300 600 \
       --caps w32 w128 w512 --seeds 0 1 2
python experiments/peninsula_geo.py --validate
python experiments/peninsula_geo.py --sweep --grid 150 300 600 \
       --caps w32 w128 w512 --seeds 0 1 2

# the interactive demo's assets
python experiments/make_demo_data.py                                  (~2 min)
```

`phase_experiment.py` appends one CSV row per (config, seed) **immediately** and
skips rows that already exist, so an interrupted run loses nothing: rerun the same
command. The 3200×3200 label raster is cached in `.cache/` (~10 s to rebuild) and is
shared by every island, peninsula and protocol experiment.

The first row of `results/*.csv` is the original server run; re-running a sweep
overwrites it, which is why the numbers above are also printed by the analysis
scripts and stored in `results/*.json`.

---

## Caveats

* **2-D toy**: `p = 3` Newton map only. No inference is made about the paper's
  3-D Lorenz example or about high-dimensional systems.
* **Island size range is narrow** (3.0–6.3 fine cells at 3200²), so the *shape* of
  `w_c(params)` is only meaningful over the decade the threshold actually moves.
  `screen_hosts.py` shows the sibling maps do not help.
* **The peninsula distance is Euclidean**, validated against a local geodesic
  (median 0.0009 units); it underestimates for winding corridors, and bins narrower
  than 0.02 units are not reliable.
* **No SGM in the loop.** The support-limit statement is an argument, not a
  measurement (see [Measured vs inferred](#measured-vs-inferred)).
* **Capacity was sampled at three widths** (1.3k / 17k / 266k) for the phase
  diagrams; the epoch budget is fixed, mitigated by the plateau control.
* **The bounded-budget schedule** (boundary share 15 % → 60 %, 60k per round) is this
  repository's design choice, not the paper's; freezing rises with the rounds partly
  because of it, and the original unbounded loop never drops uniform points.
* **A single seed in the transition region is noise**: the same configuration can
  give 0.40 and 0.85 recovery. Every number that matters here is 3–5 seeds with a
  spread reported.

---

## Citation

If this helps your work, please cite the paper it audits first, and this repository
second:

```bibtex
@article{crabtree2026separatrices,
  title  = {Generative Learning of Separatrices},
  author = {Crabtree, Ellis R. and Giovanis, Dimitris G. and Georgiou, Anastasia
            and Datseris, George and Kevrekidis, Ioannis G.},
  year   = {2026},
  eprint = {2608.14743},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

`CITATION.cff` in the repository root carries the same metadata for GitHub's
"Cite this repository" button. The code lives at
<https://github.com/Iristtspinnnnng/separatrix-calibration-demo>.

## References

* E. R. Crabtree, D. G. Giovanis, A. Georgiou, G. Datseris, I. G. Kevrekidis,
  *Generative Learning of Separatrices*, arXiv:2608.14743 (2026).
* G. Datseris, *Effortless estimation of basins of attraction*, Chaos **32**, 023104
  (2022), doi:[10.1063/5.0076568](https://doi.org/10.1063/5.0076568) — the
  `basins_of_attraction` example this benchmark is taken from.
* Y. Gal, Z. Ghahramani, *Dropout as a Bayesian Approximation: Representing Model
  Uncertainty in Deep Learning*, ICML (2016), arXiv:1506.02142 — MC dropout.
* M. T. Angulo, J. A. Moreno, R. Barabási, Y.-Y. Liu, *Fundamental limitations of
  network reconstruction from temporal data*, arXiv:1508.03559 (2015) — for readers
  who want the intermingled-basins side of the same problem.

## Use of AI tools

**Applies to version 0.2.0 and later; statement dated 2026-11-09.**

The research question, the choice of the two failure modes, the experimental
design, and every scientific interpretation in this repository are the author's
own work. No result was generated or invented by a tool: every number in
`results/`, every panel in `figures/` and every value in the tables above comes
from executing the code in this repository.

An agentic coding assistant (**DeepSeek Harness**, `deepseek-flash` model,
November 2026) was used for the following, with the author directing and
reviewing each step:

| area | what the assistant did |
|---|---|
| code | split the original flat scripts into the `separatrix` package; docstrings, CLI and repository-relative paths; removed dead code and duplicated helpers |
| tests | drafted the 25 unit tests in `tests/` |
| verification | re-ran every experiment in `experiments/`, re-derived the tabulated numbers from `results/`, and checked the cited literature — this is what surfaced the corrections listed in [`docs/RESULTS.md`](docs/RESULTS.md) |
| documentation | translated the author's Chinese drafts into English; drafted and edited this file, `docs/RESULTS.md` and `docs/README.md` |
| demo | the static page in `docs/` (`index.html`, `style.css`, `fractal.js`, `app.js`) and `experiments/make_demo_data.py` |

Checks applied to AI output before publication: `python -m pytest tests -q`
(25 tests), every script in `experiments/` executed end to end, all figures and
tables regenerated and compared with the committed artefacts, and the demo
exercised in a headless DOM harness. Two known failure modes were targeted
explicitly: references were verified against arXiv/DOI (one inherited citation
error, *Chaos* 31 vs 32, was found and fixed this way), and no numerical claim
was accepted without being recomputed from the committed artefacts — where that
was impossible, the claim is labelled *inferred* in the
[Measured vs inferred](#measured-vs-inferred) table instead of being asserted.

The author reviewed every AI-assisted change and **takes full responsibility for
the entire repository**, including anything contributed by a tool. AI tools are
not authors or contributors. Corrections are welcome as issues or pull requests.

## License

MIT — see [LICENSE](LICENSE). The paper itself is **not** redistributed here; please
cite it as above.
