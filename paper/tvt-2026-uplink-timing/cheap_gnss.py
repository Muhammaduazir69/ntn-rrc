#!/usr/bin/env python3
"""Cheap screen before the GNSS rerun (round 3, M1): which results can the vertical GNSS model change?
(a) closed-form rule: largest V meeting (vs) under sigma_v = sigma_g / 2 (old) and 2 sigma_g (cited ratio);
(b) Monte Carlo of the line-of-sight GNSS deviation over the served elevations of the campaign's UE drops;
(c) per configuration, largest error of out_r3 plus a 3-sigma LOS change, against T_e'."""
import json, os, sys
import numpy as np
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
C = 299792458.0; RE = 6378137.0
TE, A = 716.15, 260.42
ENUM = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120, 180, 240, 900]
ACC = {"kepler-j2": 1.1e-3}
def au(V, r, sig=3.0, L=0.64):
    return 2 * norm.isf(0.0042 / (2 * np.ceil(V + L))) * sig * max(1, r) / C * 1e9
def es(V, a, qp=0.5 * 1.3 * 3 ** .5, qv=0.5 * 0.06 * 3 ** .5):
    return (a * V * V + 2 * (qp + qv * V)) / C * 1e9
def res_acc(m, h):
    r = RE + h; mu = 3.986004418e14; J = 1.08262668e-3; w = 7.292115e-5
    if m == "kepler-j2": return 1.1e-3
    j = 1.5 * J * mu * RE * RE / r ** 4 * 2
    if m == "kepler": return j
    if m == "gravity-ecef": return 2 * w * (mu / r) ** .5 + w * w * r + j
    return mu / r ** 2 * (1 + 3 * J * (RE / r) ** 2) + 2 * w * (mu / r) ** .5 + w * w * r
print("(a) rule, largest V meeting the UE-side budget, h = 470 km")
for lab, lim, share in (("sat, 456 ns", 455.73, 1.0), ("sat, FR2 130 ns", 130.17, 1.0), ("gw UE share, 456 ns", 455.73, .5), ("sat, 15 kHz 553 ns", 553.37, 1.0)):
    for m in ("linear", "gravity-ecef", "kepler", "kepler-j2"):
        v = [[V for V in ENUM if es(V, res_acc(m, 470e3)) <= share * lim - au(V, r)] for r in (.5, 2)]
        print(f"  {lab:22s} {m:13s} old {max(v[0] or [0]):4d}  new {max(v[1] or [0]):4d}")
print("  A_u(5), A_u(240) old", round(au(5, .5)), round(au(240, .5)), "new", round(au(5, 2)), round(au(240, 2)))
rng = np.random.default_rng(5)
eps = np.radians(25 + 65 * (1 - rng.random(200000)) ** 2)  # served elevations, weighted toward the mask
for r in (.5, 2):
    los = 3 * np.sqrt(np.cos(eps) ** 2 + (r * np.sin(eps)) ** 2)
    print(f"(b) sigma_v/sigma_h {r}: LOS sd m min {los.min():.2f} mean {los.mean():.2f} max {los.max():.2f}, two-way ns {2 * los.mean() / C * 1e9:.1f}")
R = [json.loads(x) for x in open(os.path.join(HERE, "out_r3", "ns3_runs.jsonl"))]
cf = {}
for x in R:
    c = cf.setdefault(x["config"], dict(mx=0, lim=x["limit_ns"], v=0)); c["mx"] = max(c["mx"], x["max_ns"]); c["v"] += x["viol_frac"] > 0
d = 2 * (6 * 3.5) / C * 1e9  # 3.5 sigma at zenith under the corrected model
print(f"(c) crossing T_e' if {d:.0f} ns more LOS error adds to the largest error, zero-violation configurations:")
print("   ", sorted(k for k, c in cf.items() if c["v"] == 0 and c["mx"] + d > c["lim"]))
print("    zero-violation, still clear:", sum(c["v"] == 0 and c["mx"] + d <= c["lim"] for c in cf.values()), "of", len(cf))
