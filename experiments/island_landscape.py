"""island_landscape.py
========================================================================
A *tunable* basin-with-island testbed: gradient flow on

    V(x, y) = (x^2 - 1)^2 / 4 + y^2 / 2  -  A exp(-r_m^2 / (2 sigma^2))

i.e. the standard 2-well potential plus one shallow satellite well at m.
The satellite well is a genuine third attractor whose basin is a closed
island inside the right-hand basin; its radius is set by sigma.

Swap the satellite for a *ridge* (anisotropic sigma_x >> sigma_y) and the
same construction gives a peninsula: a narrow finger of the satellite basin
sticking into the big basin.

Nothing here is a toy label: the labels are "which minimum does the flow
reach", exactly the basin-of-attraction setting of the paper, and the
island boundary is a real closed basin boundary (not a fractal, which is the
point: it lets us isolate the *scale* variable from the fractal-resolution
variable).
========================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

__all__ = ["gradV", "flow_labels", "label_grid", "island_component"]


def gradV(x, y, m=(0.7, 0.3), A=1.0, sx=0.2, sy=0.2):
    """-grad V for the 2-well potential plus satellite Gaussian well."""
    r2 = ((x - m[0]) ** 2 + (y - m[1]) ** 2)
    g = A * np.exp(-r2 / (2 * sx * sy))
    # d/dx [ -A exp(-r2/(2 sx sy)) ] =  A*(x-mx)/(sx*sy) * exp(...)
    dVx = (x ** 2 - 1.0) * x + g * (x - m[0]) / (sx * sy)
    dVy = y + g * (y - m[1]) / (sx * sy)
    return -dVx, -dVy


def flow_labels(xy, m=(0.7, 0.3), A=1.0, sx=0.2, sy=0.2, dt=0.004,
                n_steps=4000, tol=1e-3):
    """
    Integrate dx/dt = -grad V and label by the minimum reached.

    The step is normalised to arc length (x += dt * -gradV/|gradV|), which has
    the SAME orbits as the gradient flow (positive time reparametrisation) but
    is stable for narrow wells, where explicit Euler explodes.

    1 -> left well (-1, 0), 2 -> right well (+1, 0), 3 -> satellite well,
    0 -> not settled on any of them.
    """
    x = xy[:, 0].copy()
    y = xy[:, 1].copy()
    for _ in range(n_steps):
        vx, vy = gradV(x, y, m=m, A=A, sx=sx, sy=sy)
        norm = np.hypot(vx, vy) + 1e-12
        x = x + dt * vx / norm
        y = y + dt * vy / norm
    # minima: the two base wells and the (slightly displaced) satellite min
    vx, vy = gradV(np.array([m[0]]), np.array([m[1]]), m=m, A=A, sx=sx, sy=sy)
    sx_ = min(0.3, 0.5 * float(np.hypot(sx, sy)))
    cx, cy = np.meshgrid(np.linspace(-sx_, sx_, 41), np.linspace(-sx_, sx_, 41))
    from_min = np.stack([cx.ravel() + m[0], cy.ravel() + m[1]], axis=1)
    Vg = np.array([pot(px, py, m=m, A=A, sx=sx, sy=sy) for px, py in from_min])
    best = from_min[Vg.argmin()]
    mins = np.array([[-1.0, 0.0], [1.0, 0.0], [best[0], best[1]]])
    d = np.hypot(x[:, None] - mins[None, :, 0], y[:, None] - mins[None, :, 1])
    lab = d.argmin(axis=1) + 1
    lab[d.min(axis=1) > 0.03] = 0
    return lab.astype(np.int8)


def pot(x, y, m=(0.7, 0.3), A=1.0, sx=0.2, sy=0.2):
    r2 = (x - m[0]) ** 2 + (y - m[1]) ** 2
    return (x ** 2 - 1.0) ** 2 / 4.0 + y ** 2 / 2.0 - A * np.exp(-r2 / (2 * sx * sy))

def label_grid(n, xlim=(-1.6, 1.6), ylim=(-1.2, 1.2), **kw):
    gx = np.linspace(*xlim, n)
    gy = np.linspace(*ylim, n)
    Xp, Yp = np.meshgrid(gx, gy, indexing="xy")
    xy = np.column_stack([Xp.ravel(), Yp.ravel()])
    lab = flow_labels(xy, **kw).reshape(Xp.shape)
    return lab, gx, gy


def island_component(lab, island_label=3, surround_label=2, min_area=9):
    """Largest connected island of `island_label` enclosed by `surround_label`."""
    from scipy import ndimage
    mask = lab == island_label
    comp, ncomp = ndimage.label(mask)
    if ncomp == 0:
        return None
    sizes = ndimage.sum(mask, comp, index=np.arange(1, ncomp + 1))
    struct = ndimage.generate_binary_structure(2, 2)
    best = None
    for k in range(1, ncomp + 1):
        if sizes[k - 1] < min_area:
            continue
        cm = comp == k
        ring = ndimage.binary_dilation(cm, structure=struct, iterations=2) & ~cm
        vals, counts = np.unique(lab[ring], return_counts=True)
        top = vals[counts.argmax()]
        if top == island_label or top != surround_label:
            continue
        ys, xs = np.nonzero(cm)
        rec = dict(area=int(sizes[k - 1]), mask=cm,
                   centroid=(float(xs.mean()), float(ys.mean())),
                   bbox=(int(ys.min()), int(ys.max()),
                         int(xs.min()), int(xs.max())))
        if best is None or rec["area"] > best["area"]:
            best = rec
    return best


if __name__ == "__main__":
    import time
    print("satellite well at (0.7, 0.3), A=1 -> island radius vs sigma")
    print(f"{'sigma':>7} {'island area':>12} {'r_eq(cells)':>12} "
          f"{'r_eq(units)':>12} {'cells @200':>11} {'cells @400':>11}")
    n = 600
    cellx = 3.2 / (n - 1)
    for sigma in (0.30, 0.20, 0.12, 0.07, 0.04, 0.02, 0.01):
        t0 = time.perf_counter()
        lab, gx, gy = label_grid(n, sx=sigma, sy=sigma)
        st = island_component(lab)
        if st is None:
            print(f"{sigma:>7.3f}   no enclosed island "
                  f"({time.perf_counter()-t0:.1f}s)")
            continue
        r_units = np.sqrt(st["area"] * cellx * (2.4 / (n - 1)) / np.pi)
        print(f"{sigma:>7.3f} {st['area']:>12} {np.sqrt(st['area']):>12.2f} "
              f"{r_units:>12.5f} {r_units/(3.2/199):>11.3f} "
              f"{r_units/(3.2/399):>11.3f}")
