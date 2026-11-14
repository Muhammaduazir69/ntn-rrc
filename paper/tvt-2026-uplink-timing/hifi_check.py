#!/usr/bin/env python3
"""Kepler-J2 propagator against a numerical truth with zonal terms J2..J6 and drag (Section VI-A).

SGP4 is an analytic theory, so a propagator's residual against SGP4 mixes physics with SGP4's own
approximations. Here the truth is a numerical integration (RK4, 0.1 s) of the gravity field with the
zonal harmonics J2..J6 of EGM2008 and an exponential-atmosphere drag, started from the SGP4 state of a
real satellite. The UE's Kepler-J2 propagator starts from the same state. Writes out/hifi.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_ta  # noqa: E402
import ta_core as T  # noqa: E402

# unnormalized zonal coefficients J_n (EGM2008)
JN = {2: 1.08262668e-3, 3: -2.53265649e-6, 4: -1.61962159e-6, 5: -2.27296083e-7, 6: 5.40681239e-7}
OM = T.OMEGA_E


def legendre(n, x):
    p = [np.ones_like(x), x]
    for k in range(2, n + 1):
        p.append(((2 * k - 1) * x * p[k - 1] - (k - 1) * p[k - 2]) / k)
    return p


def potential(xyz, nmax):
    r = np.linalg.norm(xyz)
    s = xyz[2] / r
    P = legendre(nmax, s)
    u = T.MU / r
    for n in range(2, nmax + 1):
        u -= T.MU / r * JN[n] * (T.RE / r) ** n * P[n]
    return u


def accel(xyz, vel_i, nmax, drag):
    h = 0.5
    g = np.array([(potential(xyz + h * e, nmax) - potential(xyz - h * e, nmax)) / (2 * h) for e in np.eye(3)])
    if drag:
        alt = np.linalg.norm(xyz) - T.RE
        rho = 3.6e-12 * np.exp(-(alt - 400e3) / 58e3)        # kg/m^3, moderate solar activity
        v_rel = vel_i - np.cross([0, 0, OM], xyz)
        g = g - 0.5 * rho * 0.02 * np.linalg.norm(v_rel) * v_rel   # ballistic coeff. C_D A / m = 0.02 m^2/kg
    return g


def integrate(x0, v0, T_end, dt, nmax, drag):
    y = np.r_[x0, v0]
    out = [y[:3].copy()]
    f = lambda y: np.r_[y[3:], accel(y[:3], y[3:], nmax, drag)]
    for _ in range(int(round(T_end / dt))):
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y[:3].copy())
    return np.array(out)


def main(n=12, T_end=240.0):
    rng = np.random.default_rng(3)
    res = {}
    for shell in ["starlink-vleo", "starlink-480", "oneweb-1200"]:
        sats = mc_ta.shell_sats(shell)
        errs = []
        for _ in range(n):
            s = sats[rng.integers(len(sats))]
            t0 = rng.random() * 86400
            p0, v0 = T.sgp4_ecef(s, 2461315.5, 0.0, t0)
            p0, v0 = p0[0], v0[0]
            vi = v0 + np.cross([0, 0, OM], p0)            # inertial frame aligned with ECEF at t0
            truth_i = integrate(p0, vi, T_end, 0.1, 6, True)[::10]          # 1 s samples
            kj2_i = integrate(p0, vi, T_end, 1.0, 2, False)
            ue = mc_ta.random_ground_under(p0, np.radians(25), rng)
            taus = np.arange(len(kj2_i)) * 1.0
            th = OM * taus
            c, sn = np.cos(th), np.sin(th)
            rot = lambda X: np.stack([c * X[:, 0] + sn * X[:, 1], -sn * X[:, 0] + c * X[:, 1], X[:, 2]], -1)
            tr, kj = rot(truth_i), rot(kj2_i)
            e = 2 * np.abs(np.linalg.norm(kj - ue, axis=1) - np.linalg.norm(tr - ue, axis=1)) / T.C
            errs.append(e)
        E = np.array(errs)
        res[shell] = {str(t): float(np.max(E[:, t]) * 1e9) for t in (30, 60, 120, 240)}
        print(shell, res[shell], flush=True)
    json.dump(res, open(os.path.join(HERE, "out", "hifi.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
