"""Regenerate the assets of the interactive demo in ``docs/``.

Two kinds of output:

* ``docs/data/*.js`` -- the measurements the page displays (island catalogue,
  zoom cases, phase-diagram curves, the metric trap, the paper-protocol
  counts).  All of it is read from ``results/`` or recomputed with the same
  functions the experiments use; nothing is typed in by hand.
* ``docs/fields/*.png`` -- the classifier's prediction over the zoom window for
  each (capacity, zoom) case, palette-indexed: 0/1/2 = basin 1/2/3,
  3 = anything else.

Run
---
    python experiments/make_demo_data.py            # ~2 min on a 4060
    python experiments/make_demo_data.py --no-train # reuse island_zoom.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json

import numpy as np

from separatrix.islands import cached_fine_labels, find_islands
from separatrix.paths import DOCS_DATA, REPO_ROOT, result
from separatrix.stats import (bootstrap_wc, ci, crossing, curve,
                             load_phase_csv)
from separatrix.zoom import (CELL, N_FINE, WIDTHS, XLIM, ZOOMS,
                             island_center, island_width, zoom_case)

FIELDS = REPO_ROOT / "docs" / "fields"
CAPS = ("w32", "w128", "w512")
CAP_HIDDEN = {"w32": (32, 32), "w128": (128, 128), "w512": (512, 512)}

# must match the palette used by docs/app.js
PALETTE = [(59, 76, 192), (224, 176, 0), (192, 57, 43), (51, 51, 51)]


def js(name, payload):
    """Write a ``window.<NAME> = {...}`` file (works from file://)."""
    path = DOCS_DATA / f"{name}.js"
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    path.write_text(f"window.DEMO_{name.upper()} = {body};\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size / 1024:.1f} KB)")


# ----------------------------------------------------------------------
def island_catalogue(labels, islands):
    """One JSON entry per island: geometry in plane units plus its mask."""
    out = []
    for isl in islands:
        cx, cy = island_center(isl)
        y0, x0 = isl["patch_origin"]
        mask = isl["patch_mask"]
        rows = ["".join("1" if v else "0" for v in row) for row in mask]
        out.append(dict(id=int(isl["id"]), label=int(isl["label"]),
                        surrounding=int(isl["surrounding"]),
                        scale_cells=round(float(isl["scale_cells"]), 3),
                        width_units=round(island_width(isl), 6),
                        area=int(isl["area"]),
                        elongation=round(float(isl["elongation"]), 2),
                        buried_cells=round(float(isl["dist_to_main"]), 1),
                        cx=round(cx, 6), cy=round(cy, 6),
                        origin=[int(y0), int(x0)], mask=rows))
    return out


def write_field(field, path):
    """Palette-indexed PNG of a predicted label field (1..3 -> 0..2)."""
    from PIL import Image
    idx = np.full(field.shape, 3, dtype=np.uint8)
    for k in (1, 2, 3):
        idx[field == k] = k - 1
    img = Image.fromarray(idx, mode="P")
    palette = [c for rgb in PALETTE for c in rgb]
    img.putpalette(palette + [0] * (768 - len(palette)))
    img.save(path, optimize=True)


def zoom_cases(island, labels, train=True):
    """Run (or reuse) the zoom experiment and export fields + numbers."""
    FIELDS.mkdir(parents=True, exist_ok=True)
    cache_path = result("island_zoom.json")
    if not train and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        rows = data["rows"]
        by_hidden = {tuple(v): k for k, v in CAP_HIDDEN.items()}
        ok = True
        for r in rows:
            cap = r.get("cap") or by_hidden.get(tuple(r["hidden"]))
            field = r.get("field") or f"{cap}_k{r['zoom']:g}.png"
            r["cap"] = cap
            r["field"] = field
            ok = ok and cap is not None and (FIELDS / field).exists()
        if ok:
            print(f"reusing {cache_path.relative_to(REPO_ROOT)} + {len(rows)} fields")
            return rows
        print("cached rows have no matching field images; retraining")

    rows = []
    for cap in CAPS:
        for k in ZOOMS:
            r = zoom_case(island, k, CAP_HIDDEN[cap], n_field=192)
            name = f"{cap}_k{k:g}.png"
            write_field(r.pop("field"), FIELDS / name)
            r.pop("window", None)
            r.update(cap=cap, field=name)
            rows.append(r)
            print(f"  {cap} k={k:>6.1f}  cells={r['island_cells']:6.1f}  "
                  f"recover={r['recover']:.2f} freeze={r['freeze']:.2f} "
                  f"acc_far={r['acc_far']:.3f}", flush=True)

    cache_path.write_text(json.dumps(
        dict(island=dict(label=int(island["label"]),
                         surrounding=int(island["surrounding"]),
                         scale_cells=float(island["scale_cells"]),
                         area=int(island["area"])),
             rows=rows), indent=1), encoding="utf-8")
    print("wrote", cache_path.relative_to(REPO_ROOT))
    return rows


# ----------------------------------------------------------------------
def phase_diagram():
    """Seed curves and the freeze boundary for the island phase diagram."""
    path = result("island.csv")
    if not path.exists():
        return None
    rows = load_phase_csv(path)
    out = {}
    for cap in CAPS:
        ux, mean, std = curve(rows, "recovery", cap)
        wc = crossing(ux, mean)
        lo_hi = ci(bootstrap_wc(rows, cap, n_boot=400))
        out[cap] = dict(x=[round(float(v), 3) for v in ux],
                        mean=[round(float(v), 3) for v in mean],
                        std=[round(float(v), 3) for v in std],
                        wc=None if np.isnan(wc) else round(float(wc), 2),
                        ci=None if np.isnan(lo_hi[0]) else
                        [round(float(v), 2) for v in lo_hi],
                        params=int([r["params"] for r in rows
                                    if r["cap"] == cap][0]))
    return out


def peninsula(table, key_bins, x_key="t_mean"):
    """Mean/std of recovery per (capacity, bin) from a peninsula CSV."""
    path = result(table)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for cap in CAPS:
        sub = [r for r in rows if r["cap"] == cap]
        if not sub:
            continue
        bins = sorted({int(r[key_bins]) for r in sub})
        mean = [float(np.mean([float(r["recovery"]) for r in sub
                               if int(r[key_bins]) == b])) for b in bins]
        std = [float(np.std([float(r["recovery"]) for r in sub
                             if int(r[key_bins]) == b], ddof=1))
               if len([r for r in sub if int(r[key_bins]) == b]) > 1 else 0.0
               for b in bins]
        xs = [float(np.mean([float(r[x_key]) for r in sub
                             if int(r[key_bins]) == b])) for b in bins]
        out[cap] = dict(bins=bins, mean=[round(v, 3) for v in mean],
                        std=[round(v, 3) for v in std],
                        x=[round(v, 5) for v in xs])
    return out


def protocol():
    path = result("paper_protocol.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_trap():
    out = {}
    path = result("demo_diagnostics.json")
    if path.exists():
        log = json.loads(path.read_text(encoding="utf-8"))
        out["refinement"] = dict(
            rounds=[d["round"] for d in log],
            boundary_fraction=[round(100 * d["boundary_fraction"], 1) for d in log],
            in_sample_boundary_acc=[round(d["boundary_acc"], 3) for d in log],
            in_sample_interior_acc=[round(d["interior_acc"], 3) for d in log],
            band1=[round(d["fair_acc_band1"], 3) for d in log],
            band8=[round(d["fair_acc_band8"], 3) for d in log],
            ece_band1=[round(d["fair_ece_band1"], 3) for d in log],
            n_train=[d["n_samples"] for d in log])
    path = result("metric_bias.json")
    if path.exists():
        out["bias"] = json.loads(path.read_text(encoding="utf-8"))
    return out or None


# ----------------------------------------------------------------------
def main(train=True):
    labels = cached_fine_labels(N_FINE, XLIM)
    islands = find_islands(labels, min_area=9, ring=2, bury=3,
                           surround_frac=0.6)
    islands.sort(key=lambda d: -d["area"])
    buried = [d for d in islands if d["dist_to_main"] >= 10]
    area_pct = 100 * sum(d["area"] for d in islands) / labels.size

    js("islands", dict(
        n_fine=N_FINE, lim=list(XLIM), cell=CELL,
        catalogue=island_catalogue(labels, islands),
        summary=dict(total=len(islands), buried=len(buried),
                     width_min=round(min(d["scale_cells"] for d in islands) * CELL, 5),
                     width_max=round(max(d["scale_cells"] for d in islands) * CELL, 5),
                     area_pct=round(area_pct, 4),
                     burial_median=float(np.median([d["dist_to_main"] for d in islands])))))

    main_island = islands[0]
    rows = zoom_cases(main_island, labels, train=train)
    js("cases", dict(
        island=dict(id=int(main_island["id"]), label=int(main_island["label"]),
                    surrounding=int(main_island["surrounding"]),
                    scale_cells=round(float(main_island["scale_cells"]), 2),
                    width_units=round(island_width(main_island), 6),
                    cx=round(island_center(main_island)[0], 6),
                    cy=round(island_center(main_island)[1], 6),
                    origin=[int(main_island["patch_origin"][0]),
                            int(main_island["patch_origin"][1])],
                    mask=["".join("1" if v else "0" for v in row)
                          for row in main_island["patch_mask"]]),
        zooms=list(ZOOMS), widths=[list(w) for w in WIDTHS],
        cases=[dict(cap=r["cap"], zoom=r["zoom"], cells=round(r["island_cells"], 2),
                    recover=round(r["recover"], 3), freeze=round(r["freeze"], 3),
                    acc_far=None if np.isnan(r["acc_far"]) else round(r["acc_far"], 3),
                    w_units=round(r["island_units"], 6),
                    window_units=round(r["window_units"], 6),
                    n_train=r["n_train"], field=r["field"])
               for r in rows]))

    js("stats", dict(
        metric_trap=metric_trap(),
        island_phase=phase_diagram(),
        peninsula_thin=peninsula("thin.csv", "bin"),
        peninsula_geo=peninsula("geo.csv", "d_bin", "d_mean"),
        protocol=protocol(),
        summary=dict(total=len(islands), buried=len(buried),
                     area_pct=round(area_pct, 4))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-train", action="store_true",
                    help="reuse results/island_zoom.json instead of retraining")
    main(train=not ap.parse_args().no_train)
