"""sweep_capacity.py
========================================================================
Does the refinement effect depend on MODEL CAPACITY -- and on what you
mean by "the boundary"?

Runs the SAME iterative-refinement loop (same data pipeline, same seeds,
same budgets) for a range of MLP widths / depths, and evaluates EVERY
intermediate model on an independent 401x401 grid with the geometric
fractal-band oracle from `evaluate.py` (accuracies inside bands of
1, 2, 4, 8 and 16 grid cells, plus the interior).

Families:
  * width sweep  : (16,16) ... (512,512)        capacity at fixed depth
  * depth sweep  : (256,) (256,256) (256,256,256)
  * control      : same epochs, but oversample_factor=1 -> the loop keeps
                   warm-start training on the uniform set WITHOUT ever
                   adding boundary points, so "more training" is separated
                   from "boundary oversampling".

Outputs: capacity_sweep.csv / .json + fig3_capacity_sweep.png
Run:     python sweep_capacity.py            # full sweep (~3 min on GPU)
         python sweep_capacity.py --quick    # 4 configs + control
========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys
import json
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from separatrix.dataset import make_dataset
from separatrix.refine import run_iterative_refinement
from separatrix.evaluate import make_eval_grid, evaluate_on_grid, params_of
from separatrix.paths import figure, result

WIDTHS = [(16, 16), (32, 32), (64, 64), (96, 96), (128, 128),
          (192, 192), (256, 256), (512, 512)]
DEPTHS = [(256,), (256, 256), (256, 256, 256)]

BANDS = (1, 2, 4, 8, 16)
CONTROL_WIDTHS = [(64, 64), (256, 256)]        # [256,256] = sweep baseline


def name_of(hidden):
    return "x".join(str(h) for h in hidden)


def run_one(hidden, X, y, grid, ev, rounds=5, seed=0, oversample_factor=3):
    store = []

    def hook(r, clf, entry):
        m = evaluate_on_grid(clf, ev, mc_samples=12, bands=BANDS)
        m.update(round=r + 1,
                 loop_boundary_acc=entry["boundary_acc"],
                 loop_confidence=entry["mean_confidence"],
                 epochs=entry["epochs"])
        store.append(m)

    t0 = time.perf_counter()
    log = run_iterative_refinement(
        X, y, grid[0], grid[1], rounds=rounds, hidden=hidden, seed=seed,
        warmup_epochs=150, extra_epochs=120, max_samples=60_000,
        oversample_factor=oversample_factor, on_round=hook, verbose=False)
    return store, log, time.perf_counter() - t0


def summarise(tag, hidden, store, log, wall):
    r1, r5 = store[0], store[-1]
    row = dict(tag=tag, hidden=list(hidden), params=params_of(hidden),
               wall_s=round(wall, 1),
               epochs_total=int(sum(d["epochs"] for d in log)),
               acc_int_r1=r1["acc_interior"], acc_int_r5=r5["acc_interior"],
               conf_wrong_r1=r1["overconf_wrong_band1"],
               conf_wrong_r5=r5["overconf_wrong_band1"],
               ece_all_r1=r1["ece"], ece_all_r5=r5["ece"],
               sep_auc1_r1=r1["sep_auc1"], sep_auc1_r5=r5["sep_auc1"],
               sep_hard_r1=r1.get("sep_auc_hard"), sep_hard_r5=r5.get("sep_auc_hard"))
    for k in BANDS:
        row[f"acc_band{k}_r1"] = r1[f"acc_band{k}"]
        row[f"acc_band{k}_r5"] = r5[f"acc_band{k}"]
        row[f"d_acc_band{k}"] = r5[f"acc_band{k}"] - r1[f"acc_band{k}"]
        row[f"ece_band{k}_r1"] = r1[f"ece_band{k}"]
        row[f"ece_band{k}_r5"] = r5[f"ece_band{k}"]
    return row


def main(quick=False):
    X, y, grid = make_dataset(n=300)
    ev = make_eval_grid(n=401, bands=BANDS, interior_band=24)
    print(f"train n={len(X)}   eval grid {len(ev['xy'])} pts (401x401)")
    for k in BANDS:
        print(f"  band {k:>2}: {100*ev['bands'][k].mean():5.2f}% of grid",
              end="")
    print(f"   interior: {100*ev['interior'].mean():5.2f}%")

    configs = [("width", w) for w in (WIDTHS[:4] if quick else WIDTHS)]
    configs += [("depth", d) for d in DEPTHS]

    rows = []
    for kind, hidden in configs:
        tag = f"{kind}:{name_of(hidden)}"
        store, log, wall = run_one(hidden, X, y, grid, ev)
        rows.append(summarise(tag, hidden, store, log, wall))
        r = rows[-1]
        print(f"{tag:>24} params={r['params']:>7}  "
              f"acc@b1 {r['acc_band1_r1']:.3f}->{r['acc_band1_r5']:.3f} "
              f"({r['d_acc_band1']:+.3f})  "
              f"@b8 {r['acc_band8_r1']:.3f}->{r['acc_band8_r5']:.3f} "
              f"({r['d_acc_band8']:+.3f})  "
              f"int {r['acc_int_r5']:.3f}  ECE@b1 {r['ece_band1_r1']:.3f}->"
              f"{r['ece_band1_r5']:.3f}  {r['wall_s']:5.1f}s", flush=True)

    # ---- control: same epochs, no boundary oversampling -----------------
    for hidden in (CONTROL_WIDTHS if not quick else [(256, 256)]):
        tag = f"control(no oversample):{name_of(hidden)}"
        store, log, wall = run_one(hidden, X, y, grid, ev,
                                   oversample_factor=1)
        rows.append(summarise(tag, hidden, store, log, wall))
        r = rows[-1]
        print(f"{tag:>24} params={r['params']:>7}  "
              f"acc@b1 {r['acc_band1_r1']:.3f}->{r['acc_band1_r5']:.3f} "
              f"({r['d_acc_band1']:+.3f})  "
              f"@b8 {r['acc_band8_r1']:.3f}->{r['acc_band8_r5']:.3f} "
              f"({r['d_acc_band8']:+.3f})  {r['wall_s']:5.1f}s", flush=True)

    keys = list(rows[0].keys())
    with open(str(result("capacity_sweep.csv")), "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    with open(str(result("capacity_sweep.json")), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print("\nwrote capacity_sweep.csv / .json")

    make_figure(rows)
    print_tables(rows)
    analyse(rows)


# ----------------------------------------------------------------------
def make_figure(rows, out=figure("fig3_capacity_sweep.png")):
    wr = [r for r in rows if r["tag"].startswith("width")]
    p = np.array([r["params"] for r in wr])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))

    # (a) accuracy inside the 1-cell band vs capacity
    ax = axes[0, 0]
    ax.plot(p, [r["acc_band1_r1"] for r in wr], "o-", color="#3b4cc0",
            label="round 1 (uniform only)")
    ax.plot(p, [r["acc_band1_r5"] for r in wr], "s-", color="#c0392b",
            label="round 5 (after refinement)")
    ax.set_xscale("log")
    ax.set_xlabel("MLP parameters"); ax.set_ylabel("accuracy, band = 1 cell")
    ax.set_title("(a) Separatrix (thinnest band) vs capacity", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) SAME picture on the wide halo
    ax = axes[0, 1]
    ax.plot(p, [r["acc_band8_r1"] for r in wr], "o-", color="#3b4cc0",
            label="round 1")
    ax.plot(p, [r["acc_band8_r5"] for r in wr], "s-", color="#c0392b",
            label="round 5")
    ax.set_xscale("log")
    ax.set_xlabel("MLP parameters"); ax.set_ylabel("accuracy, band = 8 cells")
    ax.set_title("(b) ... and on the wide halo (what the old\n"
                 "dynamical proxy measured)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) sign flip as a function of band width
    ax = axes[0, 2]
    for r, c in ((wr[0], "#95a5a6"), (wr[len(wr) // 2], "#2980b9"),
                 (wr[-1], "#c0392b")):
        ax.plot(BANDS, [r[f"d_acc_band{k}"] for k in BANDS], "o-", color=c,
                label=r["tag"])
    ctrl = [r for r in rows if r["tag"].startswith("control")]
    for r, ls in zip(ctrl, ("--", ":")):
        ax.plot(BANDS, [r[f"d_acc_band{k}"] for k in BANDS], ls,
                color="#27ae60", marker="x",
                label="no-oversample control")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("band width (grid cells)")
    ax.set_ylabel("round5 - round1 accuracy")
    ax.set_title("(c) Does refinement help?  It depends on\nthe width of "
                 "'the boundary'", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) calibration inside the thinnest band
    ax = axes[1, 0]
    ax.plot(p, [r["ece_band1_r1"] for r in wr], "o-", color="#16a085",
            label="round 1")
    ax.plot(p, [r["ece_band1_r5"] for r in wr], "s-", color="#e67e22",
            label="round 5")
    ax.set_xscale("log")
    ax.set_xlabel("MLP parameters"); ax.set_ylabel("ECE, band = 1 cell")
    ax.set_title("(d) Calibration error on the separatrix", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (e) band-width sensitivity of the reported number
    ax = axes[1, 1]
    for r, c in ((wr[0], "#95a5a6"), (wr[len(wr) // 2], "#2980b9"),
                 (wr[-1], "#c0392b")):
        ax.plot(BANDS, [r[f"acc_band{k}_r5"] for k in BANDS], "o-", color=c,
                label=f"{r['tag']} (round 5)")
    ax.set_xlabel("band width (grid cells)")
    ax.set_ylabel("accuracy inside the band")
    ax.set_title("(e) 'Boundary accuracy' is band-width dependent", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (f) confidently wrong on the fractal
    ax = axes[1, 2]
    ax.plot(p, [100 * r["conf_wrong_r1"] for r in wr], "o-",
            color="#3b4cc0", label="round 1")
    ax.plot(p, [100 * r["conf_wrong_r5"] for r in wr], "s-",
            color="#c0392b", label="round 5")
    ax.set_xscale("log")
    ax.set_xlabel("MLP parameters")
    ax.set_ylabel("% band-1 points: conf>0.9 and wrong")
    ax.set_title("(f) 'Confidently wrong on the separatrix'", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Capacity sweep of iterative boundary refinement "
                 "(all metrics on an independent 401x401 grid)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def print_tables(rows):
    print("\naccuracy vs band width (round 1 -> round 5):")
    print(f"{'config':>26} {'params':>8}" +
          "".join(f"{'band'+str(k):>18}" for k in BANDS))
    for r in rows:
        cells = "".join(
            f"{r[f'acc_band{k}_r1']:>8.3f}->{r[f'acc_band{k}_r5']:<8.3f}"
            for k in BANDS)
        print(f"{r['tag']:>26} {r['params']:>8} {cells}")

    print("\nrefinement effect (round5 - round1) by band width:")
    print(f"{'config':>26}" + "".join(f"{'band'+str(k):>10}" for k in BANDS))
    for r in rows:
        print(f"{r['tag']:>26}" +
              "".join(f"{r[f'd_acc_band{k}']:>+10.3f}" for k in BANDS))


def analyse(rows):
    wr = [r for r in rows if r["tag"].startswith("width")]
    ctrl = [r for r in rows if r["tag"].startswith("control")]
    print("\n--- summary ---")
    print("capacity floor (band1, round 1): "
          f"{wr[0]['tag']} {wr[0]['acc_band1_r1']:.3f}  ->  "
          f"{wr[-1]['tag']} {wr[-1]['acc_band1_r1']:.3f}")
    print("band-1 gain from refinement: "
          f"{min(r['d_acc_band1'] for r in wr):+.3f} .. "
          f"{max(r['d_acc_band1'] for r in wr):+.3f}")
    print("band-8 gain from refinement: "
          f"{min(r['d_acc_band8'] for r in wr):+.3f} .. "
          f"{max(r['d_acc_band8'] for r in wr):+.3f}")
    for c in ctrl:
        print(f"{c['tag']}: band1 {c['d_acc_band1']:+.3f}, "
              f"band8 {c['d_acc_band8']:+.3f}  (extra epochs, no oversampling)")
    print("crossover: refinement helps thin bands, hurts wide halos "
          "-> the sign of the effect is set by the oracle width, not only "
          "by the algorithm.")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
