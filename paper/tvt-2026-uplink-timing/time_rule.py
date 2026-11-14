#!/usr/bin/env python3
"""Run time of Algorithm 1: CPU of the rule against a fixed validity with the same fit, per epoch, on a few
campaign passes (one job at a time, nice 19). Writes out_r5/runtime.json."""
import json, os, resource, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_ns3_campaign as R

rng = np.random.default_rng(2026)
P = R.passes_for("starlink-480", 10, rng)
tmp = os.path.join(R.TMP, "time_rule.csv")


def cpu(args, p):
    a = resource.getrusage(resource.RUSAGE_CHILDREN)
    subprocess.run(["nice", "-n", "19", R.EXE, "--tleLine1=" + p["l1"].replace(" ", "_"), "--tleLine2=" + p["l2"].replace(" ", "_"),
                    f"--crossTrackKm={p['ue_x']:.3f}", f"--gwCrossTrackKm={p['gw_x']:.3f}", f"--gwAlongTrackKm={p['gw_a']:.3f}",
                    f"--startS={p['start']:.3f}", "--readPeriod=1000", "--seed=1", f"--out={tmp}"] + args.split(),
                   cwd=R.NS3, check=True, capture_output=True)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (b.ru_utime + b.ru_stime) - (a.ru_utime + a.ru_stime)


rule = [cpu(R.ALL_CONFIGS["P-gw-window"], p) for p in P]
fixed = [cpu(R.ALL_CONFIGS["W-cell"], p) for p in P]
os.unlink(tmp)
epochs = 900 / 5
ms = 1e3 * (np.median(rule) - np.median(fixed)) / epochs
json.dump({"ms_per_epoch": float(ms), "rule_s": rule, "fixed_s": fixed}, open(os.path.join(R.OUT, "runtime.json"), "w"), indent=1)
print("ms per epoch", ms)
