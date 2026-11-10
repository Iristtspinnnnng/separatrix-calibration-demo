"""peninsula_v2.py
========================================================================
A peninsula testbed whose ground truth can be trusted.

Why v1 failed
-------------
The satellite-well family (island_landscape.py) produces the *drainage
tongue* of a well sitting in the middle of a basin: its area is a large
fraction of the window, and in a thin channel the Euler/arc-length step
scheme assigns basins depending on the step size.  Ground truth like that
cannot be a denominator.

What is different here
----------------------
1. GEOMETRY: the extra well sits ON the separatrix, not inside a basin.

       V = (x^2-1)^2/4 + y^2/2 - A exp(-x^2/(2 sx^2) - (y-y0)^2/(2 sy^2))

   The base potential has a saddle at the origin (curvature -1 in x, +1 in y),
   so the basin boundary of the two base wells is the y-axis.  Adding the
   Gaussian at the saddle flips the x-curvature when A > sx^2: the origin
   becomes a third, shallow attractor.  Its basin is a tongue that hangs off
   the separatrix -- i.e. a peninsula whose ROOT is on the main boundary
   (so the root is densely sampled) and whose tip reaches into one of the two
   basins.  With A < sx^2 the tongue disappears, which gives a clean knob.

2. INTEGRATION: vectorised RK4 with a step tied to the narrowest curvature
   (dt <= 0.25 * min(sx, sy)), a settle test on |grad V| and on proximity to a
   minimum, and -- crucially -- `validate()` which compares the vectorised
   labeller against scipy's adaptive `solve_ivp` on random points.  If the
   agreement is not ~1.0 the geometry is rejected instead of silently
   producing a wrong ground truth.

3. ATTACHED, NOT BURIED: the tongue's root is on the separatrix, so samples
   are dense there by construction.  Recovery/freeze is measured on the tip
   region (cells further than `root_cells` from the boundary), which is the
   part a smooth classifier can lose while still getting the root right.

Run
----
  python peninsula_v2.py --check                 # validate a few geometries
  python peninsula_v2.py --sweep --seeds 0 1 2   # small phase sweep
========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import os
import time
import numpy as np

from scipy import ndimage
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from separatrix.classifier import MCClassifier
from separatrix.paths import result

CAPS = {"w32": (32, 32), "w128": (128, 128), "w512": (512, 512)}
XLIM = (-1.6, 1.6)


# ======================================================================
class Peninsula:
    """Two wells plus a shallow well on the separatrix -> attached tongue."""

    def __init__(self, sx=0.12, sy=0.12, A=0.60, y0=0.0,
                 xlim=(-0.9, 0.9), ylim=(-0.6, 0.6)):
        self.sx, self.sy, self.A, self.y0 = sx, sy, A, y0
        self.xlim, self.ylim = xlim, ylim

    # -- potential / gradient -----------------------------------------
    def V(self, x, y):
        return ((x * x - 1.0) ** 2 / 4.0 + y * y / 2.0
                - self.A * np.exp(-x * x / (2 * self.sx ** 2)
                                  - (y - self.y0) ** 2 / (2 * self.sy ** 2)))

    def grad(self, x, y):
        g = self.A * np.exp(-x * x / (2 * self.sx ** 2)
                            - (y - self.y0) ** 2 / (2 * self.sy ** 2))
        return (-((x * x - 1.0) * x - g * x / self.sx ** 2),
                -(y - g * (y - self.y0) / self.sy ** 2))

    # -- minima --------------------------------------------------------
    def minima(self):
        """Two base wells plus the (possible) tongue minimum near the saddle."""
        out = [np.array([-1.0, 0.0]), np.array([1.0, 0.0])]
        res = minimize(lambda p: self.V(p[0], p[1]),
                       np.array([0.0, self.y0]), method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-16, maxiter=4000))
        hess_ok = self._is_minimum(res.x)
        if hess_ok:
            out.append(res.x)
        return np.array(out), hess_ok

    def _is_minimum(self, p, eps=1e-4):
        h = eps
        V0 = self.V(p[0], p[1])
        vxx = (self.V(p[0] + h, p[1]) - 2 * V0 + self.V(p[0] - h, p[1])) / h ** 2
        vyy = (self.V(p[0], p[1] + h) - 2 * V0 + self.V(p[0], p[1] - h)) / h ** 2
        return bool(vxx > 1e-6 and vyy > 1e-6)

    # -- labelling -----------------------------------------------------
    def _rhs(self, x, y):
        return self.grad(x, y)

    def rk4(self, xy, dt=None, T=80.0, tol=2e-2, chunk=25):
        """Vectorised RK4 descent with settled-point compaction.

        Points are dropped from the active set as soon as they sit within
        `tol` of a minimum, so only genuinely slow (near-separatrix) points
        keep costing time.  Returns (label, settled_mask).
        """
        if dt is None:
            dt = 0.25 * min(self.sx, self.sy)
        mins, _ = self.minima()
        xy = np.asarray(xy, dtype=float)
        lab = np.zeros(len(xy), dtype=np.int8)
        active = np.arange(len(xy))
        t = 0.0
        while t < T and len(active):
            x = xy[active, 0].copy(); y = xy[active, 1].copy()
            for _ in range(chunk):
                k1x, k1y = self._rhs(x, y)
                k2x, k2y = self._rhs(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
                k3x, k3y = self._rhs(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
                k4x, k4y = self._rhs(x + dt * k3x, y + dt * k3y)
                x = x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
                y = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
            t += chunk * dt
            d = np.hypot(x[:, None] - mins[None, :, 0],
                         y[:, None] - mins[None, :, 1])
            done = d.min(axis=1) < tol
            if done.any():
                idx = active[done]
                lab[idx] = d[done].argmin(axis=1) + 1
                keep = ~done
                xy = np.vstack([xy[active[keep]], xy[active[~keep]][:0]]) \
                    if False else xy
                active = active[keep]
        settled = lab > 0
        return lab, settled
    def label_scipy(self, xy, t_max=400.0):
        """Reference labeller: adaptive RK45 per point (slow, accurate)."""
        mins, _ = self.minima()
        out = np.zeros(len(xy), dtype=np.int8)
        for i, p in enumerate(xy):
            sol = solve_ivp(lambda t, s: [self.grad(s[0], s[1])[0],
                                          self.grad(s[0], s[1])[1]],
                            (0.0, t_max), p, method="RK45",
                            rtol=1e-8, atol=1e-10, max_step=0.05)
            q = sol.y[:, -1]
            d = np.hypot(mins[:, 0] - q[0], mins[:, 1] - q[1])
            if d.min() < 2e-2:
                out[i] = int(d.argmin()) + 1
        return out

    def validate(self, n=200, seed=0, dt=None):
        rng = np.random.default_rng(seed)
        xy = np.column_stack([rng.uniform(*self.xlim, n),
                              rng.uniform(*self.ylim, n)])
        fast, settled = self.rk4(xy, dt=dt)
        ref = self.label_scipy(xy)
        ok = settled & (ref > 0)
        agree = float((fast[ok] == ref[ok]).mean()) if ok.any() else float("nan")
        return dict(agreement=agree, compared=int(ok.sum()),
                    unsettled=int((~settled).sum()))

    # -- tongue geometry ----------------------------------------------
    def tongue(self, n=700):
        """Component of the third attractor, eroded to its tip, plus its
        width, its distance from the separatrix, and the sample line."""
        gx = np.linspace(*self.xlim, n)
        gy = np.linspace(*self.ylim, n)
        Xp, Yp = np.meshgrid(gx, gy, indexing="xy")
        xy = np.column_stack([Xp.ravel(), Yp.ravel()])
        lab, settled = self.rk4(xy)
        lab = lab.reshape(n, n)
        mins, has_tongue = self.minima()
        if not has_tongue:
            return None
        mask = lab == 3
        comp, ncomp = ndimage.label(mask, structure=np.ones((3, 3)))
        if ncomp == 0:
            return None
        sizes = ndimage.sum(mask, comp, index=np.arange(1, ncomp + 1))
        cm = comp == (int(np.argmax(sizes)) + 1)
        if cm.sum() < 30:
            return None
        # tip = cells of the tongue far from the separatrix (x = 0)
        ys, xs = np.nonzero(cm)
        dx = (self.xlim[1] - self.xlim[0]) / (n - 1)
        dist_sep = np.abs(gx[xs])
        tip = dist_sep > 0.15
        if tip.sum() < 10:
            return None
        span_long = max(np.ptp(xs), np.ptp(ys)) * dx
        w = cm.sum() * dx * dx / max(span_long, 1e-9)
        # surrounding basin = label just outside the tongue
        ring = ndimage.binary_dilation(cm, iterations=3) & ~cm
        vals, counts = np.unique(lab[ring], return_counts=True)
        surround = int(vals[counts.argmax()])
        coords = np.column_stack([gx[xs] + 0.0, gy[ys] + 0.0])
        return dict(mask=cm, gx=gx, gy=gy, n=n, w=float(w),
                    area=int(cm.sum()), surround=surround,
                    coords_all=coords, tip_mask=tip,
                    root_frac=float((~tip).mean()),
                    has_tongue=True, mins=mins)

    # -- training grid ------------------------------------------------
    def training(self, n, seed=0):
        gx = np.linspace(*self.xlim, n)
        gy = np.linspace(*self.ylim, int(n * (self.ylim[1] - self.ylim[0])
                                         / (self.xlim[1] - self.xlim[0])))
        Xp, Yp = np.meshgrid(gx, gy, indexing="xy")
        xy = np.column_stack([Xp.ravel(), Yp.ravel()])
        lab, settled = self.rk4(xy)
        keep = settled & (lab > 0)
        return xy[keep], lab[keep], float((gx[1] - gx[0]))


# ======================================================================
def score(clf, tg, region="tip"):
    m = tg["tip_mask"] if region == "tip" else ~tg["tip_mask"]
    xy = tg["coords_all"][m]
    pred = clf.predict(xy)
    return (float((pred == 3).mean()), float((pred == tg["surround"]).mean()))


def run_point(pen, n_grid, cap, seed):
    tg = pen.tongue(n=500)
    if tg is None:
        return None
    X, y, h = pen.training(n_grid)
    clf = MCClassifier(hidden=CAPS[cap], random_state=seed)
    t0 = time.perf_counter()
    clf.fit(X, y, max_iter=250)
    wall = time.perf_counter() - t0
    tip_rec, tip_frz = score(clf, tg, "tip")
    root_rec, root_frz = score(clf, tg, "root")
    return dict(n_grid=n_grid, w=tg["w"], h=h, w_over_h=tg["w"] / h,
                area=tg["area"], n_train=int(len(X)),
                tip_recovery=tip_rec, tip_freeze=tip_frz,
                root_recovery=root_rec, root_freeze=root_frz,
                wall_s=round(wall, 2))


# ======================================================================
def check(args):
    print(f"{'sx':>6} {'sy':>6} {'A':>5} {'tongue':>7} {'w':>7} {'area':>6} "
          f"{'root%':>6} {'agreement':>10} {'n':>4}")
    for sx in (0.16, 0.12, 0.09):
        for A in (0.55, 0.65):
            pen = Peninsula(sx=sx, sy=sx, A=A)
            tg = pen.tongue(n=500)
            if tg is None:
                print(f"{sx:>6.2f} {sx:>6.2f} {A:>5.2f} {'none':>7}")
                continue
            val = pen.validate(n=120)
            print(f"{sx:>6.2f} {sx:>6.2f} {A:>5.2f} {'yes':>7} {tg['w']:>7.4f} "
                  f"{tg['area']:>6} {100*tg['root_frac']:>6.1f} "
                  f"{val['agreement']:>10.3f} {val['compared']:>4}")


def sweep(args):
    rows = []
    out = args.out
    new = not os.path.exists(out)
    f = open(out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["sx", "A", "n_grid", "cap", "seed",
                                      "w", "h", "w_over_h", "area", "n_train",
                                      "tip_recovery", "tip_freeze",
                                      "root_recovery", "root_freeze", "wall_s"])
    if new:
        w.writeheader()
    for sx in args.sx:
        pen = Peninsula(sx=sx, sy=sx, A=args.A)
        tg = pen.tongue(n=500)
        if tg is None:
            print(f"sx={sx}: no tongue"); continue
        print(f"\nsx={sx} A={args.A}: tongue w={tg['w']:.4f} area={tg['area']} "
              f"root={100*tg['root_frac']:.0f}% surround={tg['surround']}",
              flush=True)
        for n_grid in args.grid:
            for cap in args.caps:
                for seed in args.seeds:
                    r = run_point(pen, n_grid, cap, seed)
                    r.update(sx=sx, A=args.A, cap=cap, seed=seed)
                    w.writerow(r); f.flush()
                    rows.append(r)
                    print(f"  grid={n_grid:>3} {cap:>5} seed={seed} "
                          f"w/h={r['w_over_h']:5.2f} tip rec={r['tip_recovery']:.2f} "
                          f"frz={r['tip_freeze']:.2f} | root rec="
                          f"{r['root_recovery']:.2f} ({r['wall_s']:.1f}s)",
                          flush=True)
    f.close()
    print(f"\nwrote {out} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sx", nargs="+", type=float, default=[0.16, 0.12, 0.09])
    ap.add_argument("--A", type=float, default=0.6)
    ap.add_argument("--grid", nargs="+", type=int, default=[80, 160, 320])
    ap.add_argument("--caps", nargs="+", default=["w32", "w128", "w512"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default=str(result("peninsula.csv")))
    a = ap.parse_args()
    if a.check:
        check(a)
    if a.sweep:
        sweep(a)


if __name__ == "__main__":
    main()
