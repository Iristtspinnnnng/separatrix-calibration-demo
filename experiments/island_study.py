"""island_study.py
========================================================================
Do the failure modes of active iterative sampling show up on the p=3 Newton
fractal?  Buried islands make this testable: they are small patches of one
basin enclosed by another, they exist at every scale, and their position is
known exactly.

Three sampling arms, same classifier, same 60k training budget, same seeds:

  jitter  the demo's scheme: pick the top-15% *training points* by entropy
          and add Gaussian jitter around them.  Support = the training set.
  pool    proper entropy-guided active learning over the PLANE: keep a
          500k-point uniform candidate pool, take the top-15% by entropy
          and add jittered copies.  This can reach regions with no training
          point at all.
  oracle  upper bound: put the new samples *inside the islands* themselves.
          If the model still cannot separate them, the limit is capacity /
          smoothness, not sampling.

Per round, for every island: recover_rate (interior labelled correctly),
freeze_rate (interior labelled as the surrounding basin), max entropy
percentile w.r.t. the sampler's own reference, and whether the island was
sampled at all.

Run:  python island_study.py            (~2-4 min on GPU, labels cached)
========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from separatrix.classifier import MCClassifier
from separatrix.dynamics import newton_root
from separatrix.islands import (fine_labels, find_islands, island_metrics,
                                island_points)
from separatrix.paths import figure, result, cache
from separatrix.refine import run_iterative_refinement

N_FINE = 3200
XLIM = (-1.5, 1.5)
CACHE = str(cache(f"island_labels_{N_FINE}.npy"))
DENSITIES = (150, 300, 600)
SEED = 0
MAX_SAMPLES = 60_000


# ----------------------------------------------------------------------
def get_fine_labels():
    try:
        return np.load(CACHE)
    except Exception:
        t0 = time.perf_counter()
        lab = fine_labels(n=N_FINE, xlim=XLIM)
        np.save(CACHE, lab)
        print(f"[islands] labelled {N_FINE}x{N_FINE} in "
              f"{time.perf_counter()-t0:.1f}s", flush=True)
        return lab


def grid_dataset(n):
    g = np.linspace(*XLIM, n)
    Xp, Yp = np.meshgrid(g, g, indexing="xy")
    xy = np.column_stack([Xp.ravel(), Yp.ravel()])
    lab = fine_labels(n=n, xlim=XLIM).ravel()
    keep = lab > 0
    return xy[keep], lab[keep].astype(np.int8), (g, g)


def plane_coords(n, xlim=XLIM):
    g = np.linspace(*xlim, n)
    Xp, Yp = np.meshgrid(g, g, indexing="xy")
    return np.column_stack([Xp.ravel(), Yp.ravel()])


# ----------------------------------------------------------------------
def assemble(r, rng, X_base, y_base, X_bnd, y_bnd, max_samples=MAX_SAMPLES):
    frac_b = min(0.60, 0.15 + 0.10 * r)
    n_total = int(min(max_samples, len(X_base) + len(X_bnd)))
    n_b = int(min(len(X_bnd), round(frac_b * n_total)))
    n_u = max(1, n_total - n_b)
    iu = rng.choice(len(X_base), n_u, replace=False) if n_u < len(X_base) \
        else np.arange(len(X_base))
    ib = rng.choice(len(X_bnd), n_b, replace=False) if n_b else np.array([], int)
    X = np.vstack([X_base[iu], X_bnd[ib]])
    y = np.concatenate([y_base[iu], y_bnd[ib]])
    return X, y


def label_points(xy):
    y = np.asarray([newton_root(x + 1j * yc, p=3) for x, yc in xy], dtype=np.int8)
    keep = y > 0
    return xy[keep], y[keep]


def run_jitter_arm(X, y, grid, isls, n_fine, store):
    """The demo's sampler, plus a hook that evaluates islands each round."""
    def hook(r, clf, entry):
        X_train = entry.pop("_X_train")
        _, H_train = clf.mc_predict(X_train, M=6)
        entry["H_train"] = H_train
        thr = np.percentile(H_train, 85)
        m = island_metrics(clf, isls, n_fine, H_train=H_train, thr_H=thr)
        store.append({d["id"]: d for d in m})
        print(f"    [jitter] round {r+1}: recovered "
              f"{sum(1 for d in m if d['recover_rate'] > 0.5)}/{len(m)}",
              flush=True)

    run_iterative_refinement(X, y, grid[0], grid[1], rounds=5,
                             hidden=(256, 256), seed=SEED,
                             max_samples=MAX_SAMPLES, warmup_epochs=150,
                             extra_epochs=120, on_round=hook, verbose=False,
                             keep_train_set=True)


def run_pool_or_oracle(X, y, grid, isls, n_fine, store, mode="pool",
                       pool_xy=None, rounds=5, pct=85, factor=3,
                       hidden=(256, 256)):
    """Entropy-guided sampling over the plane ('pool') or inside the islands
    ('oracle').  Same bounded budget assembly as the demo loop."""
    rng = np.random.default_rng(SEED)
    clf = MCClassifier(hidden=hidden, random_state=SEED)
    X_base, y_base = X.copy(), y.copy()
    X_bnd = np.empty((0, 2)); y_bnd = np.empty((0,), np.int8)
    if mode == "oracle":
        oracle_xy = np.vstack([island_points(i, n_fine, xlim=XLIM)
                               for i in isls])

    for r in range(rounds):
        Xt, yt = assemble(r, rng, X_base, y_base, X_bnd, y_bnd)
        clf.fit(Xt, yt, warm_start=r > 0, max_iter=120 if r else 150)
        _, H_train = clf.mc_predict(Xt, M=6)
        thr_train = np.percentile(H_train, 85)
        m = island_metrics(clf, isls, n_fine, H_train=H_train, thr_H=thr_train)
        store.append({d["id"]: d for d in m})
        print(f"    [{mode}] round {r+1}: recovered "
              f"{sum(1 for d in m if d['recover_rate'] > 0.5)}/{len(m)}",
              flush=True)

        if mode == "pool":
            _, H_pool = clf.mc_predict(pool_xy, M=6)
            thr = np.percentile(H_pool, pct)
            sel = pool_xy[H_pool >= thr]
            if len(sel) == 0:
                continue
            n_new = len(sel) * (factor - 1)
            new_xy = sel.repeat(factor - 1, axis=0) + \
                rng.normal(scale=0.01, size=(n_new, 2))
        else:
            new_xy = oracle_xy + rng.normal(scale=0.002,
                                            size=oracle_xy.shape)
        new_xy = np.clip(new_xy, XLIM[0], XLIM[1])
        new_xy, new_y = label_points(new_xy)
        if len(new_xy):
            X_bnd = np.vstack([X_bnd, new_xy])
            y_bnd = np.concatenate([y_bnd, new_y])
            if len(X_bnd) > 240_000:
                s = rng.choice(len(X_bnd), 240_000, replace=False)
                X_bnd, y_bnd = X_bnd[s], y_bnd[s]
    return clf


# ----------------------------------------------------------------------
def summarise(store, isls, arm, n_fine):
    last, first = store[-1], store[0]
    rows = []
    for isl in isls:
        i = isl["id"]
        if i not in last:
            continue
        rows.append(dict(arm=arm, id=i, scale=isl["scale_cells"],
                         elong=isl["elongation"], area=isl["area"],
                         label=isl["label"], surrounding=isl["surrounding"],
                         r1_recover=first[i]["recover_rate"],
                         r5_recover=last[i]["recover_rate"],
                         r1_freeze=first[i]["freeze_rate"],
                         r5_freeze=last[i]["freeze_rate"],
                         ent_pct=first[i]["max_entropy_pct"],
                         selected=first[i]["selected"]))
    return rows


def main():
    labels = get_fine_labels()
    isls = find_islands(labels, min_area=9, ring=2, bury=3, surround_frac=0.6)
    isls.sort(key=lambda d: -d["area"])
    scales = np.array([d["scale_cells"] for d in isls])
    print(f"[islands] {len(isls)} buried islands; scale "
          f"{scales.min():.2f}..{scales.max():.2f} fine cells = "
          f"{3*scales.min()/N_FINE:.4f}..{3*scales.max()/N_FINE:.4f} units; "
          f"total area {100*sum(d['area'] for d in isls)/labels.size:.4f}% "
          f"of the plane")

    all_rows = []
    stores = {}

    # ---- jitter arm at three initial-sampling densities ----------------
    for nd in DENSITIES:
        X, y, grid = grid_dataset(nd)
        print(f"\n=== jitter arm, initial grid {nd}x{nd} "
              f"(spacing {3.0/(nd-1):.4f}, {len(X)} pts) ===", flush=True)
        store = []
        run_jitter_arm(X, y, grid, isls, N_FINE, store)
        stores[("jitter", nd)] = store
        all_rows += summarise(store, isls, f"jitter@{nd}", N_FINE)

    # ---- pool / oracle arms at the demo density ------------------------
    nd = 300
    X, y, grid = grid_dataset(nd)
    pool = plane_coords(1600)
    pool = pool[np.random.default_rng(0).choice(len(pool), 500_000,
                                                replace=False)]
    for mode in ("pool", "oracle"):
        print(f"\n=== {mode} arm, initial grid {nd}x{nd} ===", flush=True)
        store = []
        run_pool_or_oracle(X, y, grid, isls, N_FINE, store, mode=mode,
                           pool_xy=pool)
        stores[(mode, nd)] = store
        all_rows += summarise(store, isls, mode, N_FINE)

    with open(str(result("island_study.json")), "w", encoding="utf-8") as f:
        json.dump(dict(islands=[dict(id=d["id"], scale=float(d["scale_cells"]),
                                    area=int(d["area"]),
                                    label=int(d["label"]),
                                    surrounding=int(d["surrounding"]),
                                    centroid=list(map(float, d["centroid"])),
                                    elongation=float(d["elongation"]))
                             for d in isls],
                       rows=all_rows), f, indent=1)
    print("\nwrote island_study.json")
    report(all_rows, stores, isls)
    make_figure(labels, isls, all_rows, stores)


# ----------------------------------------------------------------------
def report(rows, stores, isls):
    print("\n=== freeze / recovery by arm (mean over islands) ===")
    print(f"{'arm':>12} {'freeze r1':>10} {'freeze r5':>10} "
          f"{'recovered r1':>13} {'recovered r5':>13} {'selected r1':>12}")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        sub = [r for r in rows if r["arm"] == arm]
        f1 = np.mean([r["r1_freeze"] for r in sub])
        f5 = np.mean([r["r5_freeze"] for r in sub])
        c1 = np.mean([r["r1_recover"] > 0.5 for r in sub])
        c5 = np.mean([r["r5_recover"] > 0.5 for r in sub])
        se = np.mean([r["selected"] for r in sub])
        print(f"{arm:>12} {f1:>10.3f} {f5:>10.3f} {c1:>13.2f} {c5:>13.2f} "
              f"{se:>12.2f}")

    bury = {d["id"]: d["dist_to_main"] for d in isls}
    cell = 3.0 / (N_FINE - 1)
    print("\n=== genuinely buried islands only "
          f"(burial >= 10 fine cells = {10*cell:.4f} units) ===")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        sub = [r for r in rows if r["arm"] == arm and bury[r["id"]] >= 10]
        if not sub:
            continue
        print(f"{arm:>12}  n={len(sub):>2}  freeze "
              f"{np.mean([r['r1_freeze'] for r in sub]):.3f} -> "
              f"{np.mean([r['r5_freeze'] for r in sub]):.3f}   "
              f"recovered {np.mean([r['r1_recover'] > 0.5 for r in sub]):.2f} -> "
              f"{np.mean([r['r5_recover'] > 0.5 for r in sub]):.2f}   "
              f"selected {np.mean([r['selected'] for r in sub]):.2f}")

    from scipy.stats import spearmanr
    print("\n=== correlations with round-1 freeze (jitter@300) ===")
    sub = [r for r in rows if r["arm"] == "jitter@300"]
    if len(sub) > 5:
        for name, vals in (("island scale", [r["scale"] for r in sub]),
                           ("entropy percentile", [r["ent_pct"] for r in sub]),
                           ("elongation", [r["elong"] for r in sub]),
                           ("burial distance", [bury[r["id"]] for r in sub])):
            rho, p = spearmanr(vals, [r["r1_freeze"] for r in sub])
            print(f"  Spearman(freeze, {name:<18}) = {rho:+.3f} (p={p:.3f})")


def make_figure(labels, isls, rows, stores, out=figure("fig5_islands.png")):
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(labels, origin="lower",
              cmap=ListedColormap(["#333333", "#3b4cc0", "#e0b000", "#c0392b"]),
              extent=(-1.5, 1.5, -1.5, 1.5), aspect="equal",
              interpolation="nearest")
    cx = [-1.5 + 3.0 * d["centroid"][0] / (N_FINE - 1) for d in isls]
    cy = [-1.5 + 3.0 * d["centroid"][1] / (N_FINE - 1) for d in isls]
    ax.scatter(cx, cy, s=[10 + 5 * d["scale_cells"] for d in isls],
               c="#ffffff", edgecolor="k", linewidth=0.5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"(a) {len(isls)} buried islands (white)", fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    for arm, c in (("jitter@150", "#95a5a6"), ("jitter@300", "#2980b9"),
                   ("jitter@600", "#8e44ad"), ("pool", "#e67e22"),
                   ("oracle", "#c0392b")):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        ax.scatter([r["scale"] for r in sub], [r["r5_freeze"] for r in sub],
                   s=26, color=c, alpha=0.8, label=arm)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("island scale (fine cells)")
    ax.set_ylabel("freeze rate after 5 rounds")
    ax.set_title("(b) Freezing vs island scale, per sampler", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    sub = [r for r in rows if r["arm"] == "jitter@300"]
    ax.scatter([r["ent_pct"] for r in sub], [r["r1_freeze"] for r in sub],
               s=32, c=["#c0392b" if r["selected"] else "#7f8c8d" for r in sub],
               edgecolor="k", linewidth=0.3)
    ax.axvline(85, color="#c0392b", lw=1.2, ls="--",
               label="sampler threshold (85th pct)")
    ax.set_xlabel("island max entropy, percentile of training set")
    ax.set_ylabel("freeze rate, round 1")
    ax.set_title("(c) High entropy is not enough:\nselected islands stay frozen",
                 fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    for arm, c in (("jitter@300", "#2980b9"), ("pool", "#e67e22"),
                   ("oracle", "#c0392b")):
        sub = sorted([r for r in rows if r["arm"] == arm],
                     key=lambda r: r["scale"])
        if not sub:
            continue
        bins = np.array([r["scale"] for r in sub])
        rec = np.array([r["r5_recover"] for r in sub])
        ax.scatter(bins, rec, s=26, color=c, alpha=0.8, label=arm)
    ax.set_xlabel("island scale (fine cells)")
    ax.set_ylabel("recovery rate after 5 rounds")
    ax.set_title("(d) Can the sampling ever recover them?", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for arm, c in (("jitter@300", "#2980b9"), ("pool", "#e67e22"),
                   ("oracle", "#c0392b")):
        st = stores.get((arm.split("@")[0], 300))
        if not st:
            continue
        rr = list(range(1, len(st) + 1))
        fr = [np.mean([d["freeze_rate"] for d in s.values()]) for s in st]
        ax.plot(rr, fr, "o-", color=c, label=arm)
    ax.set_xlabel("refinement round")
    ax.set_ylabel("mean freeze rate")
    ax.set_title("(e) Freezing over rounds", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    # show a *genuinely buried* island (farthest from the main Julia band)
    big = max(isls, key=lambda d: d["dist_to_main"])
    y0, y1, x0, x1 = big["bbox"]
    pad = 25
    sl = (slice(max(0, y0 - pad), min(N_FINE, y1 + pad)),
          slice(max(0, x0 - pad), min(N_FINE, x1 + pad)))
    ax.imshow(labels[sl], origin="lower",
              cmap=ListedColormap(["#333333", "#3b4cc0", "#e0b000", "#c0392b"]),
              interpolation="nearest")
    py0, px0 = big["patch_origin"]
    mask = big["patch_mask"]
    ax.contour(np.arange(mask.shape[1]) + px0 - sl[1].start,
               np.arange(mask.shape[0]) + py0 - sl[0].start,
               mask.astype(float), levels=[0.5], colors="white", linewidths=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(f) island of basin {big['label']} buried {big['dist_to_main']:.0f}"
                 f" cells inside basin {big['surrounding']}\n"
                 f"({big['scale_cells']:.1f} cells = "
                 f"{3*big['scale_cells']/N_FINE:.4f} units across)", fontsize=10)

    fig.suptitle("Buried islands: what entropy-driven iterative sampling "
                 "cannot reach", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
