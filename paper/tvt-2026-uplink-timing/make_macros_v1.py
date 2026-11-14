#!/usr/bin/env python3
"""Write ../macros_p5.tex: every number the TVT manuscript cites, from the closed forms
(ta_core), the Monte Carlo (out/mc_summary.json) and the ns-3 campaign (out/ns3_runs.csv)."""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_ta  # noqa: E402
import ta_core as T  # noqa: E402

TE = T.TE[30]
S = json.load(open(os.path.join(HERE, "out", "mc_summary.json")))
R = [r for r in csv.DictReader(open(os.path.join(HERE, "out", "ns3_runs.csv"))) if not r.get("error")]
SH = ["starlink-vleo", "starlink-480", "starlink-550", "oneweb-1200"]
M = {}


def f(x, n=1):
    return f"{x:.{n}f}"


# ---- closed forms
M["HVkm"] = f(T.h_sat_variant() / 1e3, 0)
hv = S["starlink-vleo"]["h_km"] * 1e3
M["VleoKm"] = f(S["starlink-vleo"]["h_km"], 0)
M["VleoTvPct"] = f(100 * S["starlink-vleo"]["tv_max_obs"] / (T.MAX_TV * T.Q_TV), 0)
M["DriftMaxVleo"] = f(T.ta1_max(hv, np.radians(10)) * 1e6, 1)
h480 = S["starlink-480"]["h_km"] * 1e3
M["AccLin"] = f(T.residual_acc(h480, "linear"), 1)
M["AccGrav"] = f(T.residual_acc(h480, "gravity-ecef"), 2)
M["AccKep"] = f(T.residual_acc(h480, "kepler", lat=np.pi / 2), 3)
M["AccKepJ"] = f(S["starlink-480"]["kepler-j2"]["a_res_emp"] * 1e3, 1) + r"\times10^{-3}"
M["AccKepJ"] = "$" + M["AccKepJ"] + "$"
M["TauLinSec"] = f(T.tau_star(T.residual_acc(h480, "linear"), TE), 1)
m3 = T.ta3_max(h480, np.radians(25))
M["MthreeFourEighty"] = f(m3 * 1e9, 1)
M["TauTaySec"] = f((6 * TE / m3) ** (1 / 3), 1)
M["TauTayHalf"] = f((3 * TE / m3) ** (1 / 3), 1)
M["RhoRanFour"] = f(60 / (2.56 * np.floor(5 / 2.56)), 1)
for k, v in (("Car", 33.3), ("Hst", 138.9), ("Air", 250.0)):
    M[f"Tg{k}"] = f(T.C * 0.5 * TE / (2 * v * np.cos(np.radians(25))), 2)
M["NMC"] = f"{S['starlink-480']['n']:,}".replace(",", "{,}")
counts = {sh: len(mc_ta.shell_sats(sh)) for sh in SH}
M["NVleo"], M["NFourEighty"], M["NFiveFifty"], M["NOneweb"] = (f"{counts[s]:,}".replace(",", "{,}") for s in SH)

# ---- Monte Carlo: bound gaps and validity per fit
gaps = []
for sh in SH:
    r = S[sh]
    for m in ("linear", "gravity-ecef", "kepler"):
        gaps.append(100 * (r[m]["tau99"] - r[m]["tau_star"]) / r[m]["tau99"])
    gaps.append(100 * (r["taylor"]["tau99"] - r["taylor"]["tau_star"]) / r["taylor"]["tau99"])
M["MaxBoundGapPct"] = f(max(gaps), 0)
M["MinBoundGapPct"] = f(min(gaps), 0)


def vmax(sh, kind):
    w = S[sh]["win"]
    ok = [int(v) for v, x in w.items() if x[kind]["pass"] >= 0.99]
    return max(ok) if ok else 0


gain = []
for sh, tag in zip(SH, ["Vleo", "FourEighty", "FiveFifty", "Oneweb"]):
    for kind, kt in (("taylor", "Tay"), ("fit", "Econ"), ("win", "Win")):
        M[f"V{kt}{tag}"] = str(vmax(sh, kind))
    gain.append(vmax(sh, "win") / max(vmax(sh, "taylor"), 1))
    for m, mt in (("linear", "Lin"), ("gravity-ecef", "Grav"), ("kepler", "Kep"), ("taylor", "TayC")):
        M[f"TauNN{mt}{tag}"] = f(S[sh][m]["tau99"], 0)
        M[f"TauStar{mt}{tag}"] = f(S[sh][m]["tau_star"], 1)
M["WinGainLo"] = f(min(gain), 1)
M["WinGainHi"] = f(max(gain), 0)
M["JTwoAtSixty"] = f(S["starlink-480"]["kepler-j2"]["p99_ns"]["60"], 0)
M["JTwoAtOneTwenty"] = f(S["starlink-480"]["kepler-j2"]["p99_ns"]["120"], 0)
M["KepAtSixty"] = f(S["starlink-480"]["kepler"]["p99_ns"]["60"], 0)


# ---- ns-3 campaign
def rows(cfg, sh=None):
    return [r for r in R if r["config"] == cfg and (sh is None or r["shell"] == sh)]


def pooled_viol(cfg, sh=None):
    rr = rows(cfg, sh)
    n = sum(float(r["samples"]) for r in rr)
    k = sum(float(r["viol_frac"]) * float(r["samples"]) for r in rr)
    return k, n


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def rho(cfg, sh=None):
    return np.array([float(r["reads_per_min"]) for r in rows(cfg, sh)])


M["NPasses"] = str(len({(r["shell"], r["pass"]) for r in R}))
cfgs = sorted({r["config"] for r in R})
for cfg in cfgs:
    words = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
    tag = "".join(words.get(ch, ch) for ch in cfg.title() if ch.isalnum())
    x = rho(cfg)
    if len(x):
        m = x.mean()
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        M[f"Rho{tag}"] = f(m, 1 if m >= 1 else 2)
        M[f"Rho{tag}Ci"] = f(h, 1 if m >= 1 else 2)
    k, n = pooled_viol(cfg)
    p, lo, hi = wilson(k, n)
    M[f"Viol{tag}"] = f(100 * p, 2 if p < 0.1 else 1)
    M[f"Viol{tag}Hi"] = f(100 * hi, 2)
    pr = [float(r["p99_ns"]) for r in rows(cfg)]
    M[f"Pnn{tag}"] = f(np.median(pr), 0)
    M[f"Val{tag}"] = f(np.mean([float(r["mean_validity"]) for r in rows(cfg)]), 1)


def paired_ratio(a, b):
    """Mean over passes of rho(a)/rho(b), paired by (shell, pass), with a t interval."""
    A = {(r["shell"], r["pass"]): float(r["reads_per_min"]) for r in rows(a)}
    B = {(r["shell"], r["pass"]): float(r["reads_per_min"]) for r in rows(b)}
    ks = [k for k in A if k in B and B[k] > 0]
    q = np.array([A[k] / B[k] for k in ks])
    h = stats.t.ppf(0.975, len(q) - 1) * q.std(ddof=1) / np.sqrt(len(q))
    return q.mean(), h


g, h = paired_ratio("B0-ran4", "P-gw-window")
M["RhoGainWin"], M["RhoGainWinCi"] = f(g, 1), f(h, 1)
g, h = paired_ratio("B0-ran4", "P-sat")
M["RhoGainSat"], M["RhoGainSatCi"] = f(g, 0), f(h, 0)
M["SlotViolPct"] = M["ViolLSlot"]

with open(os.path.join(HERE, "..", "macros_p5.tex"), "w") as fo:
    fo.write("% written by sim/make_macros.py; do not edit\n")
    for k in sorted(M):
        fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(json.dumps(M, indent=0)[:4000])
