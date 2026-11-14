#!/usr/bin/env python3
"""Analysis vs exact-geometry Monte Carlo (Section VI-A). Reads out/mc_<shell>.npz, writes
out/mc_summary.json.

Per shell and error term: the empirical 99th-percentile curve against age with the number of samples
per age, the closed-form bound of Section V, the share of samples above it, and the age at which the
99th percentile first exceeds T_e (30 kHz) next to the closed-form usable age. For the common term:
the largest enumerated window in which 99 % of cases stay within T_e for Taylor, economized and
window-fit coefficients, with the closed-form bounds (economized including the quartic term).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_ta  # noqa: E402
import ta_core as T  # noqa: E402

SHELLS = ["starlink-vleo", "starlink-480", "starlink-550", "oneweb-1200"]
TE = T.TE[30]
LIM = TE - T.A_IMPL   # what the extrapolation may use of Te
QP, QV = 0.5 * 1.3 * np.sqrt(3), 0.5 * 0.06 * np.sqrt(3)
MASK = np.radians(25)


def crossing(taus, curve, level):
    above = np.where(np.isfinite(curve) & (curve > level))[0]
    return float(taus[above[0]]) if len(above) else float("inf")


def pct(e, q=99, nmin=50):
    out = np.full(e.shape[1], np.nan)
    n = np.isfinite(e).sum(axis=0)
    for i in range(e.shape[1]):
        if n[i] >= nmin:
            out[i] = np.nanpercentile(e[:, i], q)
    return out, n


def main():
    S = {}
    for sh in SHELLS:
        f = os.path.join(HERE, "out", f"mc_{sh}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        taus, tausf, meta = d["taus"], d["tausf"], d["meta"]
        h = float(np.median(meta[:, 0]))
        hmin = float(meta[:, 0].min())
        sats = mc_ta.shell_sats(sh)
        a_sm = np.array([(T.MU / (s.no_kozai / 60) ** 2) ** (1 / 3) for s in sats])
        perigee = np.array([a * (1 - s.ecco) - T.RE for a, s in zip(a_sm, sats)])
        res = {"h_km": h / 1e3, "h_min_km": hmin / 1e3, "n": int(len(meta)),
               "tv_max_obs": float(meta[:, 5].max()), "td_max_obs": float(np.abs(meta[:, 7]).max()),
               "n_below_hv": int(np.sum(perigee < T.h_sat_variant())),
               "ue_span_med": float(np.median(meta[:, 8])), "gw_span_med": float(np.median(meta[:, 4]))}
        # UE-side terms
        for m in ["linear", "gravity-ecef", "kepler", "kepler-j2"]:
            e = np.abs(d[f"eph_{m}"])
            p99, n = pct(e)
            rec = {"p99_ns": {str(int(t)): float(p99[i] * 1e9) for i, t in enumerate(taus)
                              if int(t) in (5, 10, 30, 60, 120, 240, 360) and np.isfinite(p99[i])},
                   "n_at": {str(int(t)): int(n[i]) for i, t in enumerate(taus) if int(t) in (60, 120, 240, 360)},
                   "tau99": crossing(taus, p99, TE)}
            if m == "linear":
                ef = np.abs(d["fine_linear"])
                p99f, _ = pct(ef)
                rec["tau99"] = crossing(tausf, p99f, TE)
            if m != "kepler-j2":
                a = T.residual_acc(hmin, m, lat=np.pi / 2 if m == "kepler" else 0.0)
                bound = a * taus ** 2 / T.C + 2 * (QP + QV * taus) / T.C
                ok = np.isfinite(e)
                rec["bound_exceed_frac"] = float(np.mean(e[ok] > np.broadcast_to(bound, e.shape)[ok]))
                rec["tau_star"] = float(T.tau_star(a, TE))
                rec["a_res"] = float(a)
            else:
                # effective residual acceleration, quantization included: p99 c / tau^2 at long ages
                k = (taus >= 120) & np.isfinite(p99) & (n >= 200)
                rec["a_res_emp"] = float(np.max(p99[k] * T.C / taus[k] ** 2)) if k.any() else float("nan")
            res[m] = rec
        # common term, Taylor, on the fine grid
        m3 = T.ta3_max(hmin, MASK)
        ef = np.abs(d["fine_taylor"])
        p99f, _ = pct(ef)
        e = np.abs(d["com_taylor"])
        bound = m3 * taus ** 3 / 6 + 0.5 * (T.Q_TC + T.Q_TD * taus + T.Q_TV * taus ** 2)
        ok = np.isfinite(e)
        res["taylor"] = {"tau99": crossing(tausf, p99f, TE), "tau_star": float((6 * TE / m3) ** (1 / 3)),
                         "m3": m3, "m4": T.ta4_max(hmin, MASK),
                         "bound_exceed_frac": float(np.mean(e[ok] > np.broadcast_to(bound, e.shape)[ok]))}
        # windows: largest enumerated V passing at 99 %
        win = {}
        for V in sorted(int(k[7:]) for k in d.files if k.startswith("com_win")):
            rec = {}
            for kind in ("taylor", "fit", "win"):
                k = "com_taylor" if kind == "taylor" else f"com_{kind}{V}"
                e = np.abs(d[k])[:, :V + 1]
                full = np.isfinite(e).all(axis=1)
                mx = e[full].max(axis=1)
                rec[kind] = {"p99_max_ns": float(np.percentile(mx, 99) * 1e9), "pass": float(np.mean(mx <= TE)),
                             "pass_lim": float(np.mean(mx <= LIM)), "max_ns": float(mx.max() * 1e9),
                             "n": int(full.sum())}
            rec["econ_clip_frac"] = float(d[f"clip_fit{V}"].mean())
            rec["taylor_bound_ns"] = float(m3 * V ** 3 / 6 * 1e9)
            rec["econ_bound_ns"] = float(T.econ_bound(hmin, MASK, V) * 1e9)
            win[str(V)] = rec
        res["win"] = win

        def vmax(kind):
            ok = [int(v) for v, x in win.items() if x[kind]["pass"] >= 0.99 and x[kind]["n"] >= 200]
            return max(ok) if ok else 0

        res["vmax"] = {k: vmax(k) for k in ("taylor", "fit", "win")}

        def vmax_lim(kind):
            ok = [int(v) for v, x in win.items() if x[kind]["pass_lim"] >= 0.99 and x[kind]["n"] >= 200]
            return max(ok) if ok else 0

        res["vmax_lim"] = {k: vmax_lim(k) for k in ("taylor", "fit", "win")}
        res["taylor"]["tau99_lim"] = crossing(tausf, p99f, LIM)
        res["vbound_lim"] = {
            "taylor": max([v for v in range(5, 61, 5) if m3 * v ** 3 / 6 <= LIM] or [0]),
            "fit": max([v for v in range(5, 61, 5) if T.econ_bound(hmin, MASK, v) <= LIM] or [0])}
        res["vbound"] = {
            "taylor": max([v for v in range(5, 61, 5) if m3 * v ** 3 / 6 <= TE] or [0]),
            "fit": max([v for v in range(5, 61, 5) if T.econ_bound(hmin, MASK, v) <= TE] or [0])}
        res["v_cross"] = float(m3 / (8 * T.ta4_max(hmin, MASK)))
        S[sh] = res
    json.dump(S, open(os.path.join(HERE, "out", "mc_summary.json"), "w"), indent=1)
    for sh, r in S.items():
        print(sh, "h", round(r["h_km"]), "vmax", r["vmax"], "vbound", r["vbound"],
              {m: (r[m]["tau99"], round(r[m].get("tau_star", 0), 1)) for m in ("linear", "gravity-ecef", "kepler")},
              "taylor", (r["taylor"]["tau99"], round(r["taylor"]["tau_star"], 1)),
              "J2 a_emp %.2e" % r["kepler-j2"]["a_res_emp"], "n@240", r["kepler-j2"]["n_at"].get("240"),
              "nbelowHV", r["n_below_hv"], "tdmax %.1f" % (r["td_max_obs"] * 1e6))


if __name__ == "__main__":
    main()
