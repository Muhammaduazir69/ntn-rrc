#!/usr/bin/env python3
"""Round 5, m3: the Kepler-J2 propagator against the numerical orbit of hifi_check.py (zonal J2 to J6 and drag), out to the
longest age a shell serves. Writes out/hifi_pass.json. Tesseral terms are not modeled."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import mc_ta, ta_core as T, hifi_check as F  # noqa: E402
OM = T.OMEGA_E
rng = np.random.default_rng(3)
res = {}
for shell, Tend in (("starlink-480", 250), ("oneweb-1200", 568)):
    sats = mc_ta.shell_sats(shell); out = []
    for _ in range(12):
        s = sats[rng.integers(len(sats))]; t0 = rng.random() * 86400
        p0, v0 = T.sgp4_ecef(s, 2461315.5, 0.0, t0); p0, v0 = p0[0], v0[0]
        vi = v0 + np.cross([0, 0, OM], p0)
        kj = F.integrate(p0, vi, Tend, 1.0, 2, False)
        tr = F.integrate(p0, vi, Tend, 1.0, 6, True)
        ue = mc_ta.random_ground_under(p0, np.radians(25), rng)
        tau = np.arange(len(kj)) * 1.0; th = OM * tau; c, sn = np.cos(th), np.sin(th)
        rot = lambda X: np.stack([c * X[:, 0] + sn * X[:, 1], -sn * X[:, 0] + c * X[:, 1], X[:, 2]], -1)
        out.append(float((2 * np.abs(np.linalg.norm(rot(kj) - ue, axis=1) - np.linalg.norm(rot(tr) - ue, axis=1)) / T.C * 1e9).max()))
    res[shell] = dict(age=Tend, max_ns=max(out))
    print(shell, res[shell], flush=True)
json.dump(res, open(os.path.join(HERE, "out", "hifi_pass.json"), "w"), indent=0)
