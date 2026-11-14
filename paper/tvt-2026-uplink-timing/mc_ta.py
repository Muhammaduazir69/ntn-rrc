#!/usr/bin/env python3
"""Exact-geometry Monte Carlo of the uplink timing error on real constellations (Section V-A).

Truth: SGP4 propagation of the 2 Oct 2026 TLE sets, rotated to ECEF (Earth rotation, J2-J4,
drag as in SGP4).  At a random epoch t0 the network fills SIB19 from the truth: PV ephemeris
(quantized) and the common-delay polynomial (T_C, T_D, T_V), either Taylor at the epoch (as a
straightforward implementation) or fitted over the validity window (proposed).  The UE predicts
the satellite with one of four propagators and adds the polynomial; the residual against the
true two-way delay is recorded for ages tau = 0..TMAX while UE and reference both stay above the
elevation mask.  Writes out/mc_<shell>.npz.

    python3 mc_ta.py --shell starlink-low --n 3000
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ta_core as T  # noqa: E402

SHELLS = {  # name: (file, altitude band km, inclination band deg)
    "starlink-vleo": ("starlink_20261002.tle", (330, 400), (0, 180)),
    "starlink-480": ("starlink_20261002.tle", (455, 505), (0, 180)),
    "starlink-550": ("starlink_20261002.tle", (530, 580), (0, 180)),
    "oneweb-1200": ("oneweb_20261002.tle", (1150, 1250), (0, 180)),
}
MODELS = ["linear", "gravity-ecef", "kepler", "kepler-j2"]
TMAX, DT = 600.0, 1.0
Q_POS, Q_VEL = 1.3, 0.06        # PV ephemeris steps, TS 38.331 PositionVelocity-r17 (m, m/s)


def shell_sats(name):
    f, (a0, a1), (i0, i1) = SHELLS[name]
    sats = T.load_tles(os.path.join(HERE, "data", f))
    keep = []
    for s in sats:
        n = s.no_kozai / 60
        a = (T.MU / n ** 2) ** (1 / 3) - T.RE
        if a0 * 1e3 <= a <= a1 * 1e3 and i0 <= np.degrees(s.inclo) <= i1 and s.ecco < 0.005:
            keep.append(s)
    return keep


def elev(p_ground, p_sat):
    up = p_ground / np.linalg.norm(p_ground, axis=-1, keepdims=True)
    los = p_sat - p_ground
    return np.arcsin(np.sum(up * los, -1) / np.linalg.norm(los, axis=-1))


def random_ground_under(p_sat, mask, rng):
    """Uniform-by-area point inside the coverage cap of a satellite at ECEF p_sat."""
    r = np.linalg.norm(p_sat)
    psim = np.arccos(T.RE / r * np.cos(mask)) - mask
    cp = 1 - rng.random() * (1 - np.cos(psim))
    psi = np.arccos(cp)
    az = rng.random() * 2 * np.pi
    z = p_sat / r
    e = np.cross([0, 0, 1.0], z)
    e /= np.linalg.norm(e)
    n = np.cross(z, e)
    u = np.cos(psi) * z + np.sin(psi) * (np.cos(az) * e + np.sin(az) * n)
    lat = np.arcsin(u[2])
    lon = np.arctan2(u[1], u[0])
    return T.geodetic_to_ecef(lat, lon)


def economize(tc, td, tv, d3, V):
    """Chebyshev economization of the cubic Taylor term over [0, V] (Proposition 2):
    tau^3 ~ (3/2) V tau^2 - (9/16) V^2 tau + V^3/32, maximum error V^3/32.
    T_V must stay inside the unsigned field; if the economized value is negative the
    curvature is fixed at zero and (T_C, T_D) are the minimax line of the remaining
    polynomial over [0, V] (Chebyshev nodes, Remark 3)."""
    a = d3 / 6.0
    tv2 = tv + 1.5 * a * V
    if tv2 >= 0:
        return tc + a * V ** 3 / 32, td - 9 / 16 * a * V ** 2, tv2, False
    nodes = 0.5 * V * (1 - np.cos((2 * np.arange(1, 9) - 1) * np.pi / 16))
    y = tc + td * nodes + tv * nodes ** 2 + a * nodes ** 3
    A = np.stack([np.ones_like(nodes), nodes], 1)
    c0, c1 = np.linalg.lstsq(A, y, rcond=None)[0]
    return c0, c1, 0.0, True


def window_fit(delay_at, V, nn=16):
    """Fit of the network's own orbit prediction over [0, V]: least squares at Chebyshev nodes,
    with the orbit evaluated exactly at the nodes and the curvature constrained to the unsigned
    field (T_V >= 0)."""
    nodes = 0.5 * V * (1 - np.cos((2 * np.arange(1, nn + 1) - 1) * np.pi / (2 * nn)))
    yn = delay_at(nodes)
    A = np.stack([np.ones_like(nodes), nodes, nodes ** 2], 1)
    c = np.linalg.lstsq(A, yn, rcond=None)[0]
    if c[2] < 0:
        c = np.r_[np.linalg.lstsq(A[:, :2], yn, rcond=None)[0], 0.0]
    return c


def run(shell, n, seed, mask_deg=25.0, V_fit=tuple(float(v) for v in range(5, 61, 5))):
    """UE-side terms need only the UE to see the satellite; the common term needs only the gateway.
    Each is recorded over its own visible span, ages 0..TMAX s on a 1 s grid, and the Taylor common
    term and the linear propagator also on a 0.25 s grid up to 30 s."""
    rng = np.random.default_rng(seed)
    sats = shell_sats(shell)
    mask = np.radians(mask_deg)
    taus = np.arange(0, TMAX + DT / 2, DT)
    tausf = np.arange(0, 30.0 + 1e-9, 0.25)
    jd0, fr0 = 2461315.5, 0.0  # 2026-10-02 00:00 UTC (TLE epoch day)
    res = {m: np.full((n, len(taus)), np.nan) for m in MODELS}
    com = {"taylor": np.full((n, len(taus)), np.nan)}
    fine = {"taylor": np.full((n, len(tausf)), np.nan), "linear": np.full((n, len(tausf)), np.nan)}
    for V in V_fit:
        com[f"fit{int(V)}"] = np.full((n, len(taus)), np.nan)
        com[f"win{int(V)}"] = np.full((n, len(taus)), np.nan)
    meta = np.zeros((n, 11))
    clipped = {f"fit{int(V)}": np.zeros(n, bool) for V in V_fit}
    k = 0
    t_start = time.time()
    while k < n:
        s = sats[rng.integers(len(sats))]
        t0 = rng.random() * 86400.0
        p0, v0 = T.sgp4_ecef(s, jd0, fr0, t0)
        p0, v0 = p0[0], v0[0]
        ue = random_ground_under(p0, mask, rng)
        ref = random_ground_under(p0, mask, rng)
        P, _ = T.sgp4_ecef(s, jd0, fr0, t0 + taus)
        ok_ue = elev(ue, P) >= mask
        ok_gw = elev(ref, P) >= mask
        last_ue = np.argmin(ok_ue) if not ok_ue.all() else len(taus)
        last_gw = np.argmin(ok_gw) if not ok_gw.all() else len(taus)
        if last_ue < 6 or last_gw < 6:
            continue
        d_ue = np.linalg.norm(P - ue, axis=1)
        ta_c = 2 * np.linalg.norm(P - ref, axis=1) / T.C
        # Taylor coefficients at the epoch from the truth (five-point stencils, 0.5 s)
        h = 0.5
        Pm, _ = T.sgp4_ecef(s, jd0, fr0, t0 + h * np.arange(-2, 3))
        f = 2 * np.linalg.norm(Pm - ref, axis=1) / T.C
        TC0 = f[2]
        TD0 = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h)
        TV0 = 0.5 * (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h * h)
        D30 = (-f[0] + 2 * f[1] - 2 * f[3] + f[4]) / (2 * h ** 3)
        q = lambda a, b, c_: (T.quantize(a, T.Q_TC, T.MAX_TC, 0), T.quantize(b, T.Q_TD, T.MAX_TD),
                              T.quantize(c_, T.Q_TV, T.MAX_TV, 0))
        a, b, c_ = q(TC0, TD0, TV0)
        com["taylor"][k, :last_gw] = (a + b * taus + c_ * taus ** 2 - ta_c)[:last_gw]
        lf = int(min(len(tausf), (last_gw - 1) / 0.25 + 1))
        Pf, _ = T.sgp4_ecef(s, jd0, fr0, t0 + tausf[:lf])
        fine["taylor"][k, :lf] = a + b * tausf[:lf] + c_ * tausf[:lf] ** 2 - 2 * np.linalg.norm(Pf - ref, axis=1) / T.C
        delay_at = lambda x: 2 * np.linalg.norm(T.sgp4_ecef(s, jd0, fr0, t0 + x)[0] - ref, axis=1) / T.C
        for V in V_fit:
            if last_gw * DT >= V:
                cw = window_fit(delay_at, V)
                a, b, c_ = q(*cw)
                com[f"win{int(V)}"][k, :last_gw] = (a + b * taus + c_ * taus ** 2 - ta_c)[:last_gw]
                cc = economize(TC0, TD0, TV0, D30, V)
                clipped[f"fit{int(V)}"][k] = cc[3]
                a, b, c_ = q(*cc[:3])
                com[f"fit{int(V)}"][k, :last_gw] = (a + b * taus + c_ * taus ** 2 - ta_c)[:last_gw]
        pq = T.quantize(p0, Q_POS, 1e12)
        vq = T.quantize(v0, Q_VEL, 1e12)
        for m in MODELS:
            Pp = T.propagate_pv(pq, vq, taus[:last_ue], m, step=DT)
            res[m][k, :last_ue] = 2 * (np.linalg.norm(Pp - ue, axis=1) - d_ue[:last_ue]) / T.C
        lfu = int(min(len(tausf), (last_ue - 1) / 0.25 + 1))
        Pfu, _ = T.sgp4_ecef(s, jd0, fr0, t0 + tausf[:lfu])
        Pl = pq[None] + vq[None] * tausf[:lfu, None]
        fine["linear"][k, :lfu] = 2 * (np.linalg.norm(Pl - ue, axis=1) - np.linalg.norm(Pfu - ue, axis=1)) / T.C
        r = np.linalg.norm(p0)
        a_sm = (T.MU / (s.no_kozai / 60) ** 2) ** (1 / 3)
        meta[k] = [r - T.RE, np.degrees(elev(ue, p0[None])[0]), np.degrees(elev(ref, p0[None])[0]),
                   np.degrees(s.inclo), last_gw * DT, TV0, D30, TD0, last_ue * DT,
                   a_sm * (1 - s.ecco) - T.RE, s.ecco]
        k += 1
        if k % 250 == 0:
            print(f"{shell}: {k}/{n}  {time.time() - t_start:.0f} s", flush=True)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "out", f"mc_{shell}.npz"), taus=taus, tausf=tausf, meta=meta,
                        **{f"eph_{m}": v for m, v in res.items()}, **{f"com_{m}": v for m, v in com.items()},
                        **{f"fine_{m}": v for m, v in fine.items()},
                        **{f"clip_{m}": v for m, v in clipped.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", default="starlink-480")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--windows", default=",".join(str(v) for v in range(5, 61, 5)))
    a = ap.parse_args()
    run(a.shell, a.n, a.seed, V_fit=tuple(float(x) for x in a.windows.split(",")))
