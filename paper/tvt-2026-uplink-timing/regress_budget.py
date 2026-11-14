#!/usr/bin/env python3
"""Gate on the round 5 binary: every configuration in out_r4 is reproduced exactly on two passes, those the budget does
not touch as they are, and the rule configurations with --budgetLegacy=1. Four jobs, nice 19."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import run_ns3_campaign as R
rng = np.random.default_rng(2026)
P = {sh: R.passes_for(sh, 100, rng) for sh in R.SHELLS}
old = {(r["shell"], r["pass"], r["config"]): r for r in map(json.loads, open(os.path.join(R.OLD, "ns3_runs.jsonl")))}
os.makedirs(R.TMP, exist_ok=True)
jobs = []
for name, args in R.ALL_CONFIGS.items():
    a = args + (" --budgetLegacy=1" if "--validity=budget" in args else "")
    for sh, i in (("starlink-480", 3), ("oneweb-1200", 7)):
        jobs.append((name, a, sh, i))
def work(j):
    name, a, sh, i = j
    r = R.run_one(P[sh][i], a, 1000 * (R.SHELLS.index(sh) + 1) + i, f"reg_{name}_{sh}_{i}.csv", False)
    o = old[(sh, i, name)]
    return name, sh, i, all(abs(r[k] - o[k]) < 1e-9 for k in ("reads", "samples", "max_ns", "p99_ns", "viol_frac"))
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(work, jobs))
bad = [x for x in res if not x[3]]
json.dump(dict(runs=len(res), configs=len(R.ALL_CONFIGS), mismatches=len(bad), bad=[x[:3] for x in bad]), open(os.path.join(HERE, "out", "regress_budget.json"), "w"))
print(len(res), "runs", len(R.ALL_CONFIGS), "configs; mismatches", len(bad), bad[:8])
