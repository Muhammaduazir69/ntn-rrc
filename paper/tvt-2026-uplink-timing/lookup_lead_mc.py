#!/usr/bin/env python3
"""Effect of announcing each epoch L seconds ahead on the common term (M1, Nov 2026). Replays the Monte Carlo draws
of mc_ta.py (same seeds, so L = 0 must reproduce out/lookup.json), and for each lead L and validity V records the
largest error over [-L, V] of: the window fit on [-L, V] ("wide"), the window fit on [0, V] ("naive") and Taylor
coefficients. Tables are frozen from this file before the ns-3 campaign. Writes out/lookup_lead.json.
Run with a shell name per process (at most two), then merge the out/lookup_lead_*.json into lookup_lead.json."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import mc_ta, ta_core as T  # noqa: E402

SH = ["starlink-vleo", "starlink-480", "starlink-550", "oneweb-1200"]
SEED = 7  # run_mc.sh
LS = [0.0, 0.32, 0.64, 1.28, 2.56, 5.0]
VS = list(range(5, 46, 5))
NN = 16
N = 4000
jd0, fr0 = 2461315.5, 0.0
mask = np.radians(25.0)
tausg = np.arange(0, 601.0)


def fit(y, x):
    A = np.stack([np.ones_like(x), x, x ** 2], 1)
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    if c[2] < 0:
        c = np.r_[np.linalg.lstsq(A[:, :2], y, rcond=None)[0], 0.0]
    return c


def q(c):
    return np.array([T.quantize(c[0], T.Q_TC, T.MAX_TC, 0), T.quantize(c[1], T.Q_TD, T.MAX_TD), T.quantize(c[2], T.Q_TV, T.MAX_TV, 0)])


cheb = 0.5 * (1 - np.cos((2 * np.arange(1, NN + 1) - 1) * np.pi / (2 * NN)))
out = {"Ls": LS, "Vs": VS, "worst": {}, "p99": {}, "n": {}}
for sh in sys.argv[1:] or SH:
    rng = np.random.default_rng(SEED)
    sats = mc_ta.shell_sats(sh)
    rec = {(k, L, V): [] for k in ("wide", "naive", "taylor") for L in LS for V in VS}
    n = 0
    while n < N:
        s = sats[rng.integers(len(sats))]
        t0 = rng.random() * 86400.0
        p0 = T.sgp4_ecef(s, jd0, fr0, t0)[0][0]
        ue = mc_ta.random_ground_under(p0, mask, rng)
        ref = mc_ta.random_ground_under(p0, mask, rng)
        P = T.sgp4_ecef(s, jd0, fr0, t0 + tausg)[0]
        ok_ue, ok_gw = mc_ta.elev(ue, P) >= mask, mc_ta.elev(ref, P) >= mask
        last_ue = int(np.argmin(ok_ue)) if not ok_ue.all() else len(tausg)
        last_gw = int(np.argmin(ok_gw)) if not ok_gw.all() else len(tausg)
        if last_ue < 6 or last_gw < 6:  # the rejection of mc_ta.py
            continue
        # every instant this sample needs, in one SGP4 call
        h = 0.5
        pts = {"h": h * np.arange(-2, 3), "g": tausg[:min(last_gw, VS[-1] + 1)], "back": -np.array(LS)}
        for L in LS:
            if L > 0:
                pts[("pre", L)] = -np.linspace(L, 0, 9)[:-1]
            for V in VS:
                if last_gw >= V + 1:
                    pts[("nod", L, V)] = -L + (V + L) * cheb
        keys = list(pts)
        allt = np.concatenate([pts[k] for k in keys])
        D = 2 * np.linalg.norm(T.sgp4_ecef(s, jd0, fr0, t0 + allt)[0] - ref, axis=1) / T.C
        B = T.sgp4_ecef(s, jd0, fr0, t0 + pts["back"])[0]
        vis = dict(zip(LS, mc_ta.elev(ref, B) >= mask))
        Y, i = {}, 0
        for k in keys:
            Y[k] = D[i:i + len(pts[k])]
            i += len(pts[k])
        f = Y["h"]
        tay = q([f[2], (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h), 0.5 * (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h * h)])
        for L in LS:
            if L > 0 and not vis[L]:
                continue
            pre, epre = (pts[("pre", L)], Y[("pre", L)]) if L > 0 else (np.zeros(0), np.zeros(0))
            for V in VS:
                if last_gw < V + 1:
                    continue
                x = np.r_[pre, tausg[:V + 1]]
                y = np.r_[epre, Y["g"][:V + 1]]
                ev = lambda c: float(np.abs(c[0] + c[1] * x + c[2] * x * x - y).max() * 1e9)
                rec["wide", L, V].append(ev(q(fit(Y[("nod", L, V)], pts[("nod", L, V)]))))
                if L > 0:
                    rec["naive", L, V].append(ev(q(fit(Y[("nod", 0.0, V)], pts[("nod", 0.0, V)]))))
                rec["taylor", L, V].append(ev(tay))
        n += 1
        if n % 500 == 0:
            print(sh, n, flush=True)
    out["worst"][sh] = {k: {str(L): {str(V): (max(rec[k, L, V]) if rec[k, L, V] else None) for V in VS} for L in LS} for k in ("wide", "naive", "taylor")}
    out["p99"][sh] = {k: {str(L): {str(V): (float(np.percentile(rec[k, L, V], 99)) if rec[k, L, V] else None) for V in VS} for L in LS} for k in ("wide", "naive", "taylor")}
    out["n"][sh] = {str(L): {str(V): len(rec["wide", L, V]) for V in VS} for L in LS}
    print(sh, "done", flush=True)
tag = "_".join(sys.argv[1:]) or "all"
json.dump(out, open(os.path.join(HERE, "out", f"lookup_lead_{tag}.json"), "w"), indent=0)
