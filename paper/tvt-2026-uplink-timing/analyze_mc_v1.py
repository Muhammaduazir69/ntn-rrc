#!/usr/bin/env python3
"""Analysis vs exact-geometry Monte Carlo (Section V-A). Reads out/mc_<shell>.npz, writes
out/mc_summary.json and ../mc_macros.tex.

For each shell and each error term it reports the empirical 99th-percentile error curve, the
closed-form bound of Section IV, the share of samples above the bound, and the information age at
which the 99th percentile first exceeds the uplink timing error limit Te (30 kHz), next to the
closed-form validity.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ta_core as T  # noqa: E402

SHELLS = ["starlink-vleo", "starlink-480", "starlink-550", "oneweb-1200"]
TE = T.TE[30]
A_RES = {"linear": "linear", "gravity-ecef": "gravity-ecef", "kepler": "kepler"}


def crossing(taus, curve, level):
    above = np.where(np.isfinite(curve) & (curve > level))[0]
    return float(taus[above[0]]) if len(above) else float("inf")


def pct(e, q=99, nmin=50):
    out = np.full(e.shape[1], np.nan)
    for i in range(e.shape[1]):
        col = e[:, i]
        col = col[np.isfinite(col)]
        if len(col) >= nmin:
            out[i] = np.percentile(col, q)
    return out


def main():
    S = {}
    for sh in SHELLS:
        f = os.path.join(HERE, "out", f"mc_{sh}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        taus = d["taus"]
        meta = d["meta"]
        h = float(np.median(meta[:, 0]))
        hmin = float(meta[:, 0].min())
        r = T.RE + h
        res = {"h_km": h / 1e3, "n": int(len(meta)), "tv_max_obs": float(meta[:, 5].max()),
               "tv_sat_frac": float(np.mean(meta[:, 5] > T.MAX_TV * T.Q_TV))}
        # ephemeris terms
        for m in ["linear", "gravity-ecef", "kepler", "kepler-j2"]:
            e = np.abs(d[f"eph_{m}"])
            p99 = pct(e)
            rec = {"p99_ns": {str(int(t)): float(p99[i] * 1e9) for i, t in enumerate(taus)
                              if int(t) in (5, 10, 30, 60, 120, 240) and np.isfinite(p99[i])},
                   "tau99": crossing(taus, p99, TE)}
            if m in A_RES:
                a = T.residual_acc(hmin, A_RES[m], lat=np.pi / 2 if m == "kepler" else 0.0)
                bound = a * taus ** 2 / T.C + 2 * (0.65 + 0.03 * taus) / T.C
                ok = np.isfinite(e)
                rec["bound_exceed_frac"] = float(np.mean(e[ok] > np.broadcast_to(bound, e.shape)[ok]))
                rec["tau_star"] = float(T.tau_star(a, TE))
                rec["a_res"] = float(a)
            else:
                # empirical residual acceleration of the J2 propagator: fit p99 = a tau^2 / c
                k = (taus >= 30) & np.isfinite(p99)
                rec["a_res_emp"] = float(np.median(p99[k] * T.C / taus[k] ** 2))
            res[m] = rec
        # common-delay polynomial
        m3 = T.ta3_max(hmin, np.radians(25))
        e = np.abs(d["com_taylor"])
        p99 = pct(e)
        bound = m3 * taus ** 3 / 6 + 0.5 * (T.Q_TC + T.Q_TD * taus + T.Q_TV * taus ** 2)
        ok = np.isfinite(e)
        res["taylor"] = {"tau99": crossing(taus, p99, TE), "tau_star": float((6 * TE / m3) ** (1 / 3)),
                         "m3": m3, "bound_exceed_frac": float(np.mean(e[ok] > np.broadcast_to(bound, e.shape)[ok])),
                         "p99_ns": {str(int(t)): float(p99[i] * 1e9) for i, t in enumerate(taus)
                                    if int(t) in (2, 5, 10, 20, 30) and np.isfinite(p99[i])}}
        for V in sorted(int(k[7:]) for k in d.files if k.startswith("com_win")):
            for kind in ("taylor", "fit", "win"):
                k = "com_taylor" if kind == "taylor" else f"com_{kind}{V}"
                if k not in d.files:
                    continue
                e = np.abs(d[k])[:, :V + 1]
                full = np.isfinite(e).all(axis=1) & np.isfinite(np.abs(d[f"com_win{V}"])[:, :V + 1]).all(axis=1)
                mx = e[full].max(axis=1)
                res.setdefault("win", {}).setdefault(str(V), {})[kind] = {
                    "p99_max_ns": float(np.percentile(mx, 99) * 1e9), "pass": float(np.mean(mx <= TE)),
                    "n": int(full.sum())}
            if f"clip_fit{V}" in d.files:
                res["win"][str(V)]["econ_clip_frac"] = float(d[f"clip_fit{V}"].mean())
            res["win"][str(V)]["econ_bound_ns"] = float((m3 * V ** 3 / 6 / 32) * 1e9)
        res["tau_star_econ"] = float((6 * 32 * TE / m3) ** (1 / 3))
        S[sh] = res
    json.dump(S, open(os.path.join(HERE, "out", "mc_summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1)[:6000])


if __name__ == "__main__":
    main()
