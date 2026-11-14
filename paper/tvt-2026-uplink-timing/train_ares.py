#!/usr/bin/env python3
"""Round 5, M1: the residual of a Kepler-J2 UE from the Monte Carlo training draws only (seed 7, as mc_ta.py), from an
exact state, so with no quantization and no GNSS. Writes the shape test and the constant to out/ares_train_<shell>.json.
One shell per process, at most two processes; merge with --merge. No V is computed here."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import mc_ta, ta_core as T  # noqa: E402
SH = ["starlink-vleo", "starlink-480", "starlink-550", "oneweb-1200"]
jd0, fr0 = 2461315.5, 0.0
mask = np.radians(25.0)
taus = np.arange(0, 601.0)
if sys.argv[1] == "--merge":
    out = {}
    for sh in SH:
        out[sh] = json.load(open(os.path.join(HERE, "out", f"ares_train_{sh}.json")))
    e = {sh: out[sh]["p99"] for sh in SH}
    w = max(out[sh]["w_max"] for sh in SH)
    slopes = {sh: out[sh]["slope"] for sh in SH}
    json.dump(dict(shells=out, w_max=w, slopes=slopes), open(os.path.join(HERE, "out", "ares_train.json"), "w"), indent=0)
    print("slopes", slopes, "w_max m/s", w, {sh: out[sh]["w_max"] for sh in SH})
    sys.exit()
for sh in sys.argv[1:]:
    rng = np.random.default_rng(7)
    sats = mc_ta.shell_sats(sh)
    k = 0; rows = []
    while k < 4000:
        s = sats[rng.integers(len(sats))]
        t0 = rng.random() * 86400.0
        p0, v0 = T.sgp4_ecef(s, jd0, fr0, t0)
        p0, v0 = p0[0], v0[0]
        ue = mc_ta.random_ground_under(p0, mask, rng)
        ref = mc_ta.random_ground_under(p0, mask, rng)
        P = T.sgp4_ecef(s, jd0, fr0, t0 + taus)[0]
        ok_ue, ok_gw = mc_ta.elev(ue, P) >= mask, mc_ta.elev(ref, P) >= mask
        lu = int(np.argmin(ok_ue)) if not ok_ue.all() else len(taus)
        lg = int(np.argmin(ok_gw)) if not ok_gw.all() else len(taus)
        if lu < 6 or lg < 6:
            continue
        Pp = T.propagate_pv(p0, v0, taus[:lu], "kepler-j2", step=1.0)
        e = np.full(len(taus), np.nan)
        e[:lu] = 2 * np.abs(np.linalg.norm(Pp - ue, axis=1) - np.linalg.norm(P[:lu] - ue, axis=1)) / T.C
        rows.append(e)
        k += 1
    E = np.array(rows)
    ok = taus >= 1
    w = np.nanmax(E[:, ok] * T.C / (2 * taus[ok]))
    p99 = np.nanpercentile(E, 99, axis=0)
    n = np.isfinite(E).sum(axis=0)
    a = np.where((taus >= 30) & (taus <= 200) & (n >= 50))[0]
    slope = float(np.polyfit(np.log(taus[a]), np.log(p99[a]), 1)[0])
    a2 = E[:, ok] * T.C / taus[ok] ** 2
    json.dump(dict(w_max=float(w), slope=slope, a_tau2=float(np.nanmax(a2[:, taus[ok] >= 30])), p99={str(int(t)): (float(p99[int(t)]) * 1e9 if n[int(t)] else None) for t in (30, 60, 120, 180, 240, 300, 400, 500)},
                   n={str(int(t)): int(n[int(t)]) for t in (30, 60, 120, 180, 240, 300, 400, 500)}, argmax_age=float(taus[ok][np.nanargmax(np.nanmax(E[:, ok] * T.C / (2 * taus[ok]), axis=0))]), maxns=float(np.nanmax(E) * 1e9)),
              open(os.path.join(HERE, "out", f"ares_train_{sh}.json"), "w"), indent=0)
    print(sh, "done", flush=True)
