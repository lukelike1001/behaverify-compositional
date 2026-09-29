"""
Affine bounds for the displacement (v*cos(th)*dt, v*sin(th)*dt).

Once the network contracts became affine, the INTERVAL displacement bounds
became the binding constraint (dy width 0.0732 per step at nvd=32 compounds
past NAV's 0.2445 obstacle margin within ~4 steps).  The same trick applies:
linearise about the cell centre and add a sound second-order remainder.

    f(v,th) = v*cos(th)*dt
    f = f0 + cos(th0)*dt*(v-v0) - v0*sin(th0)*dt*(th-th0) + R

The remainder is bounded from the Hessian over the cell:
    d2f/dv2 = 0,  d2f/dvdth = -sin(th)*dt,  d2f/dth2 = -v*cos(th)*dt
    |R| <= dv*dth*max|sin|*dt + 0.5*dth^2*max|v*cos|*dt
with dv, dth the cell half-widths.  Both maxima are taken over the whole cell,
so the bound is sound rather than a local approximation.
"""
import math

import numpy as np


def affine_displacement(vlo, vhi, tlo, thi, dt=0.2, use_sin=False):
    """Sound affine bounds for v*cos(th)*dt (or v*sin(th)*dt) over one cell.

    Returns (av, ath, c, rad): the true value lies within
        av*v + ath*th + c  +/-  rad
    for every (v, th) in the cell.
    """
    v0, t0 = (vlo + vhi) / 2, (tlo + thi) / 2
    dv, dth = (vhi - vlo) / 2, (thi - tlo) / 2
    ts = np.linspace(tlo, thi, 200)

    if use_sin:                      # f = v*sin(th)*dt
        f0 = v0 * math.sin(t0) * dt
        av, ath = math.sin(t0) * dt, v0 * math.cos(t0) * dt
        max_dg = float(np.max(np.abs(np.cos(ts))))   # |d/dth of sin|
        max_g = float(np.max(np.abs(np.sin(ts))))
    else:                            # f = v*cos(th)*dt
        f0 = v0 * math.cos(t0) * dt
        av, ath = math.cos(t0) * dt, -v0 * math.sin(t0) * dt
        max_dg = float(np.max(np.abs(np.sin(ts))))   # |d/dth of cos|
        max_g = float(np.max(np.abs(np.cos(ts))))

    # |R| <= [ max|g'| * dv * dth  +  0.5 * max|v*g| * dth^2 ] * dt
    max_vg = max(abs(vlo), abs(vhi)) * max_g
    rad = (max_dg * dv * dth + 0.5 * max_vg * dth ** 2) * dt

    c = f0 - av * v0 - ath * t0
    return av, ath, c, rad


def _interval_width(vlo, vhi, tlo, thi, dt=0.2, use_sin=False, samples=400):
    """Width of the concretized interval bound, for comparison."""
    ts = np.linspace(tlo, thi, samples)
    tr = np.sin(ts) if use_sin else np.cos(ts)
    vals = np.concatenate([vlo * tr, vhi * tr]) * dt
    pad = abs(vhi - vlo) * (thi - tlo) * dt
    return float(vals.max() - vals.min()) + 2 * pad


def main():
    print(f"{'nvd':>5} {'contracts':>10} {'interval w':>11} {'affine w':>9} {'gain':>6}")
    for nvd in (16, 32, 64, 128):
        vd = np.linspace(-2, 2, nvd + 1)
        td = np.linspace(-math.pi, math.pi, nvd + 1)
        a = min(max(int((-1.554 + 2) / 4 * nvd), 0), nvd - 1)
        b = min(max(int((0.0 + math.pi) / (2 * math.pi) * nvd), 0), nvd - 1)
        iw = _interval_width(vd[a], vd[a + 1], td[b], td[b + 1])
        _, _, _, rad = affine_displacement(vd[a], vd[a + 1], td[b], td[b + 1])
        print(f"{nvd:>5} {nvd**2:>10} {iw:>11.4f} {2*rad:>9.4f} "
              f"{iw / max(2*rad, 1e-12):>5.1f}x")

    # soundness spot check
    rng = np.random.default_rng(0)
    worst = -math.inf
    for _ in range(4000):
        vlo = rng.uniform(-2, 1.8); vhi = vlo + rng.uniform(0.01, 0.2)
        tlo = rng.uniform(-math.pi, math.pi - 0.3); thi = tlo + rng.uniform(0.01, 0.3)
        for use_sin in (False, True):
            av, ath, c, rad = affine_displacement(vlo, vhi, tlo, thi, use_sin=use_sin)
            vs = rng.uniform(vlo, vhi, 30); ths = rng.uniform(tlo, thi, 30)
            true = vs * (np.sin(ths) if use_sin else np.cos(ths)) * 0.2
            approx = av * vs + ath * ths + c
            worst = max(worst, float(np.max(np.abs(true - approx) - rad)))
    print(f"\nsoundness spot check, 4000 random cells x 60 points: "
          f"worst excess {worst:+.3e}")
    print("SOUND" if worst <= 0 else "*** UNSOUND ***")


if __name__ == "__main__":
    main()
