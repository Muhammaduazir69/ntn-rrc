#!/usr/bin/env python3
"""Uplink timing pre-compensation in NR NTN: geometry, closed forms and UE-side propagators.

Section III (system model) and Section IV (analysis) of the TVT manuscript evaluate the functions
in this file; the equation numbers are listed in Revision_Plan/math/equation_map.md.

Conventions: SI units (m, s) inside; TA is the two-way delay 2 d / c (TS 38.213 cl. 4.2,
TS 38.211 cl. 4.3.1); the SIB19 common-delay polynomial is
    TA_common(t) = T_C + T_D tau + T_V tau^2,  tau = t - t_epoch,
so T_V is HALF the second derivative of the two-way common delay.
"""
import numpy as np
from sgp4.api import Satrec, jday

MU = 3.986004418e14          # m^3/s^2 (WGS-84)
RE = 6378137.0               # m, equatorial radius (WGS-84)
F_WGS = 1 / 298.257223563
J2 = 1.08262668e-3
J3 = -2.5327e-6
OMEGA_E = 7.2921150e-5       # rad/s
C = 299792458.0              # m/s
TC = 1.0 / (480e3 * 4096)    # NR basic time unit, TS 38.211 cl. 4.1

# SIB19 field granularity and range, TS 38.331 (NTN-Config-r17)
Q_TC, MAX_TC = 4.072e-9, 66485757
Q_TD, MAX_TD = 0.2e-9, 257303          # s/s per step
Q_TV, MAX_TV = 0.2e-10, 28949          # s/s^2 per step
# UE transmit timing error limit for satellite access Te_NTN, TS 38.133 V19.6.0 cl. 7.1C.2,
# Table 7.1C.2-1 (FR1-NTN, SSB/UL SCS 15/15, 30/30) and Table 7.1C.2-3 (FR2-NTN VSAT, UL 120 kHz)
TE = {15: 29 * 64 * TC, 30: 22 * 64 * TC, 120: 7.5 * 64 * TC}
# Implementation allowance: the terrestrial Te of Table 7.1.2-1 (SSB/UL 30/30 kHz), which covers downlink
# timing and RF error with no pre-compensation in play. A proxy; TS 38.133 does not split Te_NTN.
A_IMPL = 8 * 64 * TC


# ---------------------------------------------------------------- closed forms (non-rotating)
def range_derivs(h, psi0, theta):
    """Slant range and its first three time derivatives for a circular orbit of altitude h seen by
    a ground point whose minimum central angle to the ground track is psi0, at orbital phase theta
    from closest approach.  cos(phi) = cos(psi0) cos(theta) (spherical right triangle), hence
    d^2 = R^2 + r^2 - 2 s cos(theta) with s = R r cos(psi0).  Returns d, d', d'', d''' (m, m/s, ...)."""
    R, r = RE, RE + h
    w = np.sqrt(MU / r ** 3)
    s = R * r * np.cos(psi0)
    d = np.sqrt(R * R + r * r - 2 * s * np.cos(theta))
    d1 = s * w * np.sin(theta) / d
    d2 = (s * w * w * np.cos(theta) - d1 ** 2) / d
    d3 = (-s * w ** 3 * np.sin(theta) - 3 * d1 * d2) / d
    return d, d1, d2, d3


def range_d4(h, psi0, theta):
    # Fourth derivative of the slant range: -(s w^4 cos + 3 d2^2 + 4 d1 d3) / d.
    R, r = RE, RE + h
    w = np.sqrt(MU / r ** 3)
    s = R * r * np.cos(psi0)
    d, d1, d2, d3 = range_derivs(h, psi0, theta)
    return (-s * w ** 4 * np.cos(theta) - 3 * d2 ** 2 - 4 * d1 * d3) / d


def central_angle_max(h, elev_min):
    r = RE + h
    return np.arccos(RE / r * np.cos(elev_min)) - elev_min


def elevation_of(h, phi):
    r = RE + h
    return np.arctan2(np.cos(phi) - RE / r, np.sin(phi))


def ta2_zenith(h):
    """Peak second derivative of the two-way delay, overhead pass:  2 mu R / (c h r^2)."""
    r = RE + h
    return 2 * MU * RE / (C * h * r ** 2)


def h_sat_variant(tv_max=MAX_TV * Q_TV):
    """Altitude below which T_V = TA''/2 at an overhead pass exceeds the largest encodable value."""
    from scipy.optimize import brentq
    return brentq(lambda h: ta2_zenith(h) / 2 - tv_max, 100e3, 2000e3)


def ta1_max(h, elev_min):
    """Largest two-way drift 2|d'|/c over a pass, attained at the elevation mask of an overhead pass."""
    psim = central_angle_max(h, elev_min)
    _, d1, _, _ = range_derivs(h, 0.0, psim)
    return 2 * abs(d1) / C


def ta3_max(h, elev_min, psi0=0.0, n=20001):
    psim = central_angle_max(h, elev_min)
    if psi0 >= psim:
        return 0.0
    thm = np.arccos(np.cos(psim) / np.cos(psi0))
    th = np.linspace(-thm, thm, n)
    return 2 * np.max(np.abs(range_derivs(h, psi0, th)[3])) / C


def ta4_max(h, elev_min, psi0=0.0, n=20001):
    psim = central_angle_max(h, elev_min)
    if psi0 >= psim:
        return 0.0
    thm = np.arccos(np.cos(psim) / np.cos(psi0))
    th = np.linspace(-thm, thm, n)
    return 2 * np.max(np.abs(range_d4(h, psi0, th))) / C


def econ_bound(h, elev_min, V):
    # Proposition 2: worst common-term error of economized coefficients over [0, V].
    return ta3_max(h, elev_min) * V ** 3 / 192 + ta4_max(h, elev_min) * V ** 4 / 24


# residual line-of-sight acceleration left by each UE propagation model (Section IV-B)
def residual_acc(h, model, lat=0.0):
    r = RE + h
    v = np.sqrt(MU / r)
    if model == "linear":           # straight line in ECEF: misses gravity (with J2) and the frame terms
        return MU / r ** 2 * (1 + 3 * J2 * (RE / r) ** 2) + 2 * OMEGA_E * v + OMEGA_E ** 2 * r
    if model == "gravity-ecef":     # two-body gravity in the rotating frame: misses Coriolis, centrifugal, J2
        return 2 * OMEGA_E * v + OMEGA_E ** 2 * r + 1.5 * J2 * MU * RE ** 2 / r ** 4 * 2.0
    if model == "kepler":           # two-body in an inertial frame: misses J2
        s2 = np.sin(lat) ** 2       # magnitude of the J2 acceleration: radial and latitudinal parts
        return 1.5 * J2 * MU * RE ** 2 / r ** 4 * np.sqrt(1 - 2 * s2 + 5 * s2 ** 2)
    raise ValueError(model)


def tau_star(a_res, budget, cos_eta=1.0):
    """Largest information age with (2/c) * 0.5 * a_res * cos_eta * tau^2 <= budget."""
    return np.sqrt(budget * C / (a_res * cos_eta))


# ---------------------------------------------------------------- truth: SGP4 in ECEF
def gmst(jd_ut1, fr=0.0):
    # jd and its fraction are kept apart: a single double near 2.46e6 days resolves only ~40 us,
    # which jitters the Earth-fixed frame by millimeters and spoils third-derivative stencils
    t = ((jd_ut1 - 2451545.0) + fr) / 36525.0
    g = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t ** 3
    return np.radians((g % 86400) / 240.0)


def teme_to_ecef(r, v, jd, fr=0.0):
    th = gmst(jd, fr)
    c, s = np.cos(th), np.sin(th)
    x = c * r[..., 0] + s * r[..., 1]
    y = -s * r[..., 0] + c * r[..., 1]
    pos = np.stack([x, y, r[..., 2]], -1)
    vx = c * v[..., 0] + s * v[..., 1] + OMEGA_E * y
    vy = -s * v[..., 0] + c * v[..., 1] - OMEGA_E * x
    vel = np.stack([vx, vy, v[..., 2]], -1)
    return pos, vel


def load_tles(path):
    L = [x.rstrip() for x in open(path) if x.strip()]
    return [Satrec.twoline2rv(L[i + 1], L[i + 2]) for i in range(0, len(L) - 2, 3)]


def sgp4_ecef(sat, jd0, fr0, t):
    """ECEF position and velocity (m, m/s) at times t (s) after (jd0, fr0)."""
    t = np.atleast_1d(t)
    fr = fr0 + t / 86400.0
    jd = np.full_like(fr, jd0)
    e, r, v = sat.sgp4_array(jd, fr)
    r = np.asarray(r) * 1e3
    v = np.asarray(v) * 1e3
    return teme_to_ecef(r, v, jd, fr)


def geodetic_to_ecef(lat, lon, alt=0.0):
    e2 = F_WGS * (2 - F_WGS)
    N = RE / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.stack([(N + alt) * np.cos(lat) * np.cos(lon), (N + alt) * np.cos(lat) * np.sin(lon),
                     (N * (1 - e2) + alt) * np.sin(lat)], -1)


# ---------------------------------------------------------------- UE-side propagators from SIB19 PV
def _acc_inertial(x, j2=True):
    r = np.linalg.norm(x)
    a = -MU * x / r ** 3
    if j2:
        z2 = (x[2] / r) ** 2
        k = 1.5 * J2 * MU * RE ** 2 / r ** 5
        a = a + k * np.array([x[0] * (5 * z2 - 1), x[1] * (5 * z2 - 1), x[2] * (5 * z2 - 3)])
    return a


def _rk4(f, y, dt, n):
    out = [y]
    for _ in range(n):
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y)
    return np.array(out)


def propagate_pv(p0, v0, taus, model, step=1.0):
    """Satellite ECEF position predicted by a UE at ages taus from the broadcast ECEF state (p0, v0).
    Models: linear, gravity-ecef (two-body in the rotating frame, no frame terms), kepler (two-body in
    the inertial frame), kepler-j2 (with J2)."""
    taus = np.asarray(taus, float)
    if model == "linear":
        return p0[None] + v0[None] * taus[:, None]
    T = float(taus.max())
    n = max(1, int(np.ceil(T / step)))
    dt = T / n if T > 0 else step
    grid = np.arange(n + 1) * dt
    if model == "gravity-ecef":
        f = lambda y: np.concatenate([y[3:], _acc_inertial(y[:3], j2=False)])
        Y = _rk4(f, np.concatenate([p0, v0]), dt, n)
        P = Y[:, :3]
    else:
        # ECEF -> inertial frame aligned with ECEF at the epoch
        vi = v0 + np.cross([0, 0, OMEGA_E], p0)
        f = lambda y: np.concatenate([y[3:], _acc_inertial(y[:3], j2=(model == "kepler-j2"))])
        Y = _rk4(f, np.concatenate([p0, vi]), dt, n)
        th = OMEGA_E * grid
        c, s = np.cos(th), np.sin(th)
        X = Y[:, :3]
        P = np.stack([c * X[:, 0] + s * X[:, 1], -s * X[:, 0] + c * X[:, 1], X[:, 2]], -1)
    idx = np.round(taus / dt).astype(int)
    assert np.allclose(idx * dt, taus, atol=1e-9), "ages must lie on the integration grid"
    return P[idx]


def quantize(x, q, xmax, lo=None):
    lo = -xmax if lo is None else lo
    return np.clip(np.round(x / q), lo, xmax) * q
