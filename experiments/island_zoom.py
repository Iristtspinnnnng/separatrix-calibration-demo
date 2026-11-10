"""Resolution threshold: which limit is binding, sampling or capacity?

The buried islands of the p=3 Newton map are 0.0028-0.0059 units across, i.e.
far below the 0.01 spacing of the demo's 300x300 training grid.  At those
settings "the model smooths the island away" and "the island was never
sampled" are the same statement; this experiment pulls them apart by zooming
the training window while keeping the island fixed in the plane
(:mod:`separatrix.zoom`).

Sweeping the zoom factor k = 1.5 ... 150 takes the island from "huge" to
"smaller than one sampling cell", with architecture and sample count held
fixed.  Repeating it for three capacities shows where the ceiling comes from.

Run
---
    python experiments/island_zoom.py        # ~1 min on a 4060, labels cached
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from separatrix.islands import cached_fine_labels, find_islands
from separatrix.paths import figure, result
from separatrix.zoom import WIDTHS, XLIM, ZOOMS, CELL, N_FINE, zoom_case


def main():
    labels = cached_fine_labels(N_FINE, XLIM)
    islands = find_islands(labels, min_area=9, ring=2, bury=3,
                           surround_frac=0.6)
    islands.sort(key=lambda d: -d["area"])
    isl = islands[0]
    print(f"[island] basin {isl['label']} inside {isl['surrounding']}, "
          f"scale {isl['scale_cells']:.2f} cells = "
          f"{isl['scale_cells'] * CELL:.4f} units, area {isl['area']} cells")

    rows = []
    for hidden in WIDTHS:
        for k in ZOOMS:
            r = zoom_case(isl, k, hidden)
            r.pop("field", None)
            r.pop("window", None)
            rows.append(r)
            print(f"  width {str(hidden):>12} zoom k={k:>6.1f}  "
                  f"island spans {r['island_cells']:6.1f} cells  "
                  f"recover={r['recover']:.3f} freeze={r['freeze']:.3f}  "
                  f"acc_far={r['acc_far']:.3f}  epochs={r['epochs']:>3}  "
                  f"{r['fit_s']:5.2f}s", flush=True)

    with open(result("island_zoom.json"), "w", encoding="utf-8") as f:
        json.dump(dict(island=dict(label=int(isl["label"]),
                                   surrounding=int(isl["surrounding"]),
                                   scale_cells=float(isl["scale_cells"]),
                                   area=int(isl["area"])),
                       rows=rows), f, indent=1)
    print("wrote", result("island_zoom.json"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for hidden, c in zip(WIDTHS, ("#95a5a6", "#2980b9", "#c0392b")):
        sub = [r for r in rows if tuple(r["hidden"]) == hidden]
        x = [r["island_cells"] for r in sub]
        axes[0].plot(x, [r["recover"] for r in sub], "o-", color=c,
                     label=f"width {hidden}")
        axes[1].plot(x, [r["freeze"] for r in sub], "s-", color=c,
                     label=f"width {hidden}")
    for ax, title, ylab in ((axes[0], "(a) Island recovered?", "recover rate"),
                            (axes[1], "(b) Island frozen into its neighbour?",
                             "freeze rate")):
        ax.set_xscale("log")
        ax.set_xlabel("island width, in training-grid cells")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Zoom experiment: the same island at controlled relative "
                 "scale (fixed architecture, fixed sample count)", fontsize=13)
    fig.tight_layout()
    fig.savefig(figure("fig6_island_zoom.png"), dpi=150, bbox_inches="tight")
    print("wrote", figure("fig6_island_zoom.png"))

    print("\n--- threshold: smallest island width (training cells) that is "
          "still recovered (>= 0.5) ---")
    for hidden in WIDTHS:
        sub = sorted([r for r in rows if tuple(r["hidden"]) == hidden],
                     key=lambda r: r["island_cells"])
        thr = [r["island_cells"] for r in sub if r["recover"] > 0.5]
        if thr:
            print(f"  width {str(hidden):>12}: recovers when the island spans "
                  f">= {min(thr):.1f} cells")
        else:
            print(f"  width {str(hidden):>12}: never recovered")


if __name__ == "__main__":
    main()
