#!/usr/bin/env python3
"""ns-3 campaign of the TVT study: the paper's own binary (bin/p5-ul-timing, ns3_patch/) over LEO passes.

Each pass: a satellite drawn from the 2 Oct 2026 element set of a shell, closest approach at a time drawn
uniformly over one day after the element set's epoch (so all orbital phases are covered), the 900 s window
centered on it, UE cross-track offset U(0, 0.8 psi_max), gateway at a random offset inside the footprint.
Every configuration runs on the same passes (paired design); the noise seed depends on shell and pass.
Per run: SIB19 decodings, share of samples above the limit Te - A_impl and above Te, 99th percentile and
largest |error|, closest-approach latitude, binary hash. Resumable: finished runs stay in
out_r4/ns3_runs.jsonl. Writes out_r4/ns3_runs.csv and out_r4/ns3_raw.npz.
Every scheme on the 5 s grid announces its epochs LEAD seconds ahead, and its UE decodes at a random point of
the last LEAD seconds of its validity and applies the content at once, so tau covers [-LEAD, V].

    python3 run_ns3_campaign.py --passes 100 --jobs 4   (the runs themselves go through nice -n 19)
"""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_ta  # noqa: E402
import ta_core as T  # noqa: E402

NS3 = "/home/uzair/6g_ntn_ns3/ns-3-dev"
EXE = os.path.join(HERE, "bin", "p5-ul-timing")
OUT = os.path.join(HERE, "out_r5")   # rule configurations on the round 5 binary; all others stay in out_r4
OLD = os.path.join(HERE, "out_r4")
TMP = os.path.join(HERE, "tmp")
TE = 716.15
A_DEF = 260.42                 # 8*64*Tc: terrestrial Te at SSB/UL 30/30 kHz, TS 38.133 Table 7.1.2-1
SHELLS = ["starlink-vleo", "starlink-480", "starlink-550", "oneweb-1200"]
LEAD = 0.64                    # si-Periodicity rf64: every UE meets one SI window before the epoch
LEADS = (0.0, 0.32, 0.64, 1.28, 2.56, 5.0)
A_OF = {0: 0.0, 130: 130.0, 260: A_DEF, 390: 390.0}
# static tables, frozen from out/lookup_lead.json (Monte Carlo replay, lookup_lead_mc.py) before this campaign: the
# largest V with no MC case above beta*(Te - A) ("b") or (Te - A) ("f") for the window fit on [-L, V], or above
# (Te - A) ("t") for Taylor coefficients, all over [-L, V]. Written @b260@0.64 in a configuration.
_LL = json.load(open(os.path.join(HERE, "out", "lookup_lead.json")))


def table(kind, a, lead, lim_te=TE):
    lim = lim_te - A_OF.get(a, a)
    src, x = {"b": ("wide", 0.5), "f": ("wide", 1.0), "t": ("taylor", 1.0)}[kind]
    out = []
    for sh in SHELLS:
        w = _LL["worst"][sh][src][str(float(lead))]
        out.append(max([int(v) for v, e in w.items() if e is not None and e <= x * lim] or [5]))
    return out


def build_configs():
    S = "--gnssSigma=3"
    I = f"--implNs={A_DEF}"
    c = {}
    # ---- satellite reference point
    for name, ts in (("B0", 2.56), ("B1", 1), ("B5", 5)):
        c[name] = f"--ref=satellite --validity=fixed --fitWindow=5 --sibPeriod={ts} {S} {I}"
        c[name + "-lin"] = c[name] + " --model=linear"
    # UE that takes every update instead of waiting for expiry (the later --readPeriod wins)
    c["B1-r"] = c["B1"] + " --readPeriod=1"
    c["B1-r-lin"] = c["B1-lin"] + " --readPeriod=1"
    c["F60"] = f"--ref=satellite --validity=fixed --fitWindow=60 --sibPeriod=5 {S} {I}"
    for m in ("linear", "gravity-ecef", "kepler", "kepler-j2"):
        c["S-" + m] = f"--ref=satellite --validity=fixed --fitWindow=240 --sibPeriod=5 {S} {I} --model={m}"
    P = f"--ref=satellite --validity=budget --sibPeriod=5 {S}"
    c["P-sat"] = f"{P} {I}"
    for m, t in (("linear", "linear"), ("gravity-ecef", "gravity"), ("kepler", "kepler")):
        c["P-sat-" + t] = f"{P} {I} --model={m}"
    c["B2-ekf"] = f"{P} {I} --model=kepler --ueFilter=ekf"
    # ---- gateway reference point
    G = f"--ref=gateway --sibPeriod=5 {S}"
    c["T5-grid"] = f"--ref=gateway --sibPeriod=2.56 {S} {I} --fit=taylor --validity=fixed --fitWindow=5"
    c["T5"] = f"{G} {I} --fit=taylor --validity=fixed --fitWindow=5"
    c["T60"] = f"{G} {I} --fit=taylor --validity=fixed --fitWindow=60"
    # the same static table idea with Taylor coefficients, so that only the coefficients differ from W-shell
    c["T-shell"] = f"{G} {I} --fit=taylor --validity=fixed --fitWindow=@t260@{LEAD}"
    c["T-shell-grid"] = f"{G} {I} --fit=taylor --validity=fixed --fitWindow=@t260@0.0".replace("--sibPeriod=5", "--sibPeriod=2.56")
    c["L-slot"] = f"{G} {I} --fit=taylor --validity=slot"
    c["W-cell"] = f"{G} {I} --fit=window --validity=fixed --fitWindow={min(table('b', 260, LEAD))}"
    c["W-cell-full"] = f"{G} {I} --fit=window --validity=fixed --fitWindow={min(table('f', 260, LEAD))}"
    W = f"{G} {I} --fit=window --validity=fixed"
    c["W-shell"] = W + f" --fitWindow=@b260@{LEAD}"
    c["W-shell-full"] = W + f" --fitWindow=@f260@{LEAD}"
    c["W-shell-grid"] = (W + " --fitWindow=@b260@0.0").replace("--sibPeriod=5", "--sibPeriod=2.56")
    B = f"{G} --validity=budget"
    c["P-gw-taylor"] = f"{B} {I} --fit=taylor"
    c["P-gw-econ"] = f"{B} {I} --fit=econ"
    c["P-gw-window"] = f"{B} {I} --fit=window"
    c["P-gw-window-grid"] = c["P-gw-window"].replace("--sibPeriod=5", "--sibPeriod=2.56")
    # ---- UE weaker than the network assumes
    for m, t in (("linear", "linear"), ("gravity-ecef", "gravity"), ("kepler", "kepler")):
        c["M-gw-" + t] = c["P-gw-window"] + f" --model={m} --assumedModel=kepler-j2"
    for m, t in (("linear", "linear"), ("kepler", "kepler")):
        c["M-shell-" + t] = c["W-shell"] + f" --model={m}"
    # ---- network orbit error, budgeted or ignored
    for od in (10, 30):
        e = f"--odPos={od} --odVel=0.01"
        c[f"P-sat-od{od}"] = f"{c['P-sat']} {e}"
        c[f"P-sat-od{od}-ign"] = f"{c['P-sat']} {e} --odBudget=0"
        c[f"P-satf-od{od}"] = f"{c['P-sat']} {e} --gnbFeeder=orbit"
        c[f"P-satf-od{od}-ign"] = f"{c['P-sat']} {e} --gnbFeeder=orbit --odBudget=0"
        c[f"P-gw-od{od}"] = f"{c['P-gw-window']} {e}"
        c[f"P-gw-od{od}-ign"] = f"{c['P-gw-window']} {e} --odBudget=0"
        c[f"W-shell-od{od}"] = f"{c['W-shell']} {e}"
    # ---- other timing limits, each with the terrestrial Te of the same SCS (TS 38.133 Table 7.1.2-1):
    # 15/15 kHz 12*64*Tc; FR2 120/120 kHz 3.5*64*Tc
    for lab, te, a in (("15k", 944.0, 390.63), ("vsat", 244.1, 113.93)):
        c[f"P-gw-{lab}"] = f"{B} --fit=window --teNs={te} --implNs={a}"
        c[f"P-sat-{lab}"] = f"{P} --teNs={te} --implNs={a}"
    c["W-shell-vsat"] = W.replace(I, "--implNs=113.93") + f" --fitWindow=@b260@{LEAD} --teNs=244.1"
    # ---- implementation allowance sweep
    for a in (0, 130, 390):
        i = f"--implNs={A_OF[a]}"
        c[f"P-gw-a{a}"] = f"{B} --fit=window {i}"
        c[f"P-sat-a{a}"] = f"{P} {i}"
        c[f"W-shell-a{a}"] = f"{G} {i} --fit=window --validity=fixed --fitWindow=@b{a}@{LEAD}"
        c[f"T-shell-a{a}"] = f"{G} {i} --fit=taylor --validity=fixed --fitWindow=@t{a}@{LEAD}"
    # ---- common-term share
    for b, t in ((0.25, "b25"), (0.75, "b75")):
        c[f"P-gw-{t}"] = c["P-gw-window"] + f" --commonShare={b}"
    # ---- orbital format of EphemerisInfo-r17
    o = " --ephFormat=orbital"
    c["P-sat-orb"] = c["P-sat"] + o
    c["P-sat-orb-kepler"] = c["P-sat-kepler"] + o
    c["P-sat-orb-vsat"] = c["P-sat-vsat"] + o
    c["S-orb"] = c["S-kepler-j2"] + o
    c["P-gw-window-orb"] = c["P-gw-window"] + o
    # ---- vehicles: name, speed m/s, fix period s, projection
    for v, (sp, fix, pr) in {"car": (33.3, 1.0, 0), "hst": (138.9, 1.0, 0), "aircraft": (250.0, 1.0, 0),
                             "hst-fast": (138.9, 0.2, 0), "aircraft-fast": (250.0, 0.2, 0),
                             "hst-proj": (138.9, 1.0, 1), "aircraft-proj": (250.0, 1.0, 1)}.items():
        c[f"V-{v}"] = f"{P} {I} --ueAllowanceNs=93 --ueSpeed={sp} --gnssPeriod={fix} --ueProject={pr}"
    # ---- announcing each epoch ahead: the protected recipe (fit and rule on [-L, V], tables re-frozen),
    # the unprotected one (fit and rule on [0, V], table of L = 0), and the reference without a lead
    for L in LEADS:
        t = f"{L:g}".replace(".", "p")
        if L != LEAD:
            c[f"T-shell-L{t}"] = f"{G} {I} --fit=taylor --validity=fixed --fitWindow=@t260@{L} --leadS={L}"
            c[f"W-shell-L{t}"] = W + f" --fitWindow=@b260@{L} --leadS={L}"
            c[f"P-gw-L{t}"] = c["P-gw-window"] + f" --leadS={L}"
        if L > 0:
            c[f"W-naive-L{t}"] = W + f" --fitWindow=@b260@0.0 --leadS={L} --leadAware=0"
            c[f"P-naive-L{t}"] = c["P-gw-window"] + f" --leadS={L} --leadAware=0"
    # ---- the GNSS level: core schemes at other sigma_g; the UE that labels with its transmit instant (A1)
    for n in ("T-shell", "W-shell", "W-shell-full", "P-gw-window", "P-sat", "S-kepler-j2"):
        for g in (1, 5, 10):
            c[f"{n}-g{g}"] = c[n].replace("--gnssSigma=3", f"--gnssSigma={g}")
    for n in ("T5", "T-shell", "W-shell", "W-shell-full", "P-gw-window", "P-sat"):
        c[f"{n}-tx"] = c[n] + " --ueTxLabel=1"
    # ---- round 4: no expiry inside a pass at the satellite reference (V = 900 s, the top of the enumerated set),
    # and the common-term share at the larger GNSS levels
    c["S-900"] = f"--ref=satellite --validity=fixed --fitWindow=900 --sibPeriod=5 {S} {I} --model=kepler-j2"
    for g in (5, 10):
        c[f"S-900-g{g}"] = c["S-900"].replace("--gnssSigma=3", f"--gnssSigma={g}")
    for m, t in (("linear", "linear"), ("gravity-ecef", "gravity"), ("kepler", "kepler")):
        c["S-900-" + t] = c["S-900"].replace("--model=kepler-j2", f"--model={m}")
    c["S-900-tx"] = c["S-900"] + " --ueTxLabel=1"
    for od in (10, 30):
        e = f"--odPos={od} --odVel=0.01 --odBudget=0"
        c[f"S-900-od{od}"] = c["S-900"] + " " + e
        c[f"S-900-satf-od{od}"] = c["S-900"] + " " + e + " --gnbFeeder=orbit"
    for b, t in ((0.25, "b25"), (0.40, "b40"), (0.45, "b45"), (0.48, "b48"), (0.49, "b49")):
        c[f"P-gw-g5-{t}"] = c["P-gw-window-g5"] + f" --commonShare={b}"
    c["P-gw-g10-b25"] = c["P-gw-window-g10"] + " --commonShare=0.25"
    # ---- round 5: the velocity error of the network orbit, at 10 m (position sigma), unbudgeted
    for v, t in ((0.03, "v03"), (0.1, "v10")):
        e = f"--odPos=10 --odVel={v} --odBudget=0"
        c[f"S-900-od10-{t}"] = c["S-900"] + " " + e
        c[f"S-900-satf-od10-{t}"] = c["S-900"] + " " + e + " --gnbFeeder=orbit"
        c[f"W-shell-od10-{t}"] = c["W-shell"] + f" --odPos=10 --odVel={v}"
    # every other scheme on the 5 s grid uses the bound; the conformance grids need no lead
    for k in c:
        if "--sibPeriod=5" in c[k] and "--leadS" not in c[k]:
            c[k] += f" --leadS={LEAD}"
    return c


UNREPORTED = ("F60", "P-gw-window-grid", "M-shell-kepler", "P-sat-15k", "P-sat-orb", "P-sat-orb-kepler", "P-sat-orb-vsat",
              "P-gw-window-orb")  # no number of the manuscript uses them since the cuts of round 2
ALL_CONFIGS = {k: v for k, v in build_configs().items() if k not in UNREPORTED}
# what the round 5 budget touches: every configuration in which the rule chooses the validity
CONFIGS = {k: v for k, v in ALL_CONFIGS.items() if "--validity=budget" in v}
RAW = {"S-linear", "S-gravity-ecef", "S-kepler", "S-kepler-j2"}
# Core comparisons run on all passes (100 per shell, so the rule of three reaches 0.75 % of passes);
# the stress configurations on the first 40 per shell, the same passes (the machine is shared).
CORE = {"B0", "B1", "B5", "P-sat", "S-kepler-j2", "T5", "T5-grid", "T-shell", "W-cell", "W-cell-full", "W-shell", "W-shell-full",
        "P-gw-taylor", "P-gw-econ", "P-gw-window", "T-shell-L0", "W-shell-L0", "S-900", "W-shell-g5", "P-gw-window-g5",
        "P-gw-g5-b25", "P-gw-g5-b40", "P-gw-g5-b45", "P-gw-g5-b48", "P-gw-g5-b49"}
N_STRESS = 40
HOLD = {"T-shell", "W-shell", "W-shell-full", "T5", "P-gw-window", "P-sat", "S-900", "S-kepler-j2"}
HOLD_DATE = "20261104"


def opt(args, key, default):
    for tok in args.split():
        if tok.startswith(f"--{key}="):
            return float(tok.split("=")[1])
    return default


def passes_for(shell, n, rng):
    sats = mc_ta.shell_sats(shell)
    f = mc_ta.SHELLS[shell][0]
    lines = [x.rstrip() for x in open(os.path.join(HERE, "data", f)) if x.strip()]
    by_num = {}
    for i in range(0, len(lines) - 2, 3):
        by_num[lines[i + 1][2:7].strip()] = (lines[i + 1], lines[i + 2])
    out = []
    for _ in range(n):
        s = sats[rng.integers(len(sats))]
        l1, l2 = by_num[str(s.satnum_str).strip()]
        r = (T.MU / (s.no_kozai / 60) ** 2) ** (1 / 3)
        psim = T.central_angle_max(r - T.RE, np.radians(25)) * T.RE / 1e3
        out.append(dict(shell=shell, l1=l1, l2=l2, start=float(rng.uniform(0, 86400)),
                        ue_x=float(rng.uniform(0, 0.8 * psim)),
                        gw_x=float(rng.uniform(-0.8 * psim, 0.8 * psim)),
                        gw_a=float(rng.uniform(-0.5 * psim, 0.5 * psim))))
    return out


def run_one(p, args, seed, tmpname, keep):
    shell_idx = SHELLS.index(p["shell"])
    toks = []
    for tok in args.split():
        if "=@" in tok:  # per-shell lookup value
            k, v = tok.split("=@")
            ak, L = v.split("@")
            tok = f"{k}={table(ak[0], int(ak[1:]), float(L))[shell_idx]}"
        toks.append(tok)
    path = os.path.join(TMP, tmpname)
    cmd = ["nice", "-n", "19", EXE, "--tleLine1=" + p["l1"].replace(" ", "_"), "--tleLine2=" + p["l2"].replace(" ", "_"),
           f"--crossTrackKm={p['ue_x']:.3f}", f"--gwCrossTrackKm={p['gw_x']:.3f}", f"--gwAlongTrackKm={p['gw_a']:.3f}",
           f"--startS={p['start']:.3f}", "--readPeriod=1000", f"--seed={seed}", f"--out={path}"] + toks
    r = subprocess.run(cmd, cwd=NS3, capture_output=True, text=True, timeout=1800)
    try:
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-300:])
        summ = dict(kv.split("=", 1) for kv in r.stdout.strip().split() if "=" in kv)
        rows = list(csv.DictReader(open(path)))
    finally:
        if os.path.exists(path):
            os.unlink(path)
    if not rows:
        return None
    e = np.abs(np.array([float(x["err_ns"]) for x in rows]))
    age = np.array([float(x["age_s"]) for x in rows])
    val = np.array([float(x["validity_s"]) for x in rows])
    te = opt(args, "teNs", TE)
    lim = te - opt(args, "implNs", 0.0)
    out = dict(samples=len(rows), reads=int(summ["reads"]), infeasible=int(summ.get("infeasible", 0)),
               viol_frac=float(np.mean(e > lim)), viol_te=float(np.mean(e > te)),
               p99_ns=float(np.percentile(e, 99)), max_ns=float(e.max()), mean_validity=float(val.mean()),
               limit_ns=lim, ca_lat=float(summ["ca_lat_deg"]), start=p["start"])
    if keep:
        out["raw"] = (age.astype(np.float32), e.astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--holdout", action="store_true", help="core schemes on a fresh element set and new seed")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    assert a.jobs <= 4
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "campaign_holdout.pid" if a.holdout else "campaign.pid"), "w") as fo:
        fo.write(str(os.getpid()))
    os.makedirs(TMP, exist_ok=True)
    raw_dir = os.path.join(OUT, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    binhash = hashlib.sha256(open(EXE, "rb").read()).hexdigest()
    if a.holdout:  # element sets fetched after every table and rule was frozen, and a new seed
        a.seed = 777
        for sh in SHELLS:
            f0, *rest = mc_ta.SHELLS[sh]
            mc_ta.SHELLS[sh] = (f0.replace("20261002", HOLD_DATE), *rest)
    rng = np.random.default_rng(a.seed)
    P = {sh: passes_for(sh, a.passes, rng) for sh in SHELLS}
    # stress runs first per pass index, so a partial campaign is still balanced
    jobs = [(sh, i, n, P[sh][i], args) for i in range(a.passes) for sh in SHELLS for n, args in CONFIGS.items()
            if (n in HOLD if a.holdout else (n in CORE or i < N_STRESS))]
    done_path = os.path.join(OUT, "holdout_runs.jsonl" if a.holdout else "ns3_runs.jsonl")
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            rec = json.loads(line)
            assert rec.get("bin") == binhash, "campaign mixes binaries"
            if "error" not in rec:
                done.add((rec["shell"], rec["pass"], rec["config"]))
    lock = threading.Lock()
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} runs, {len(CONFIGS)} configs, {len(done)} done, {len(todo)} to run, bin {binhash[:12]}", flush=True)

    def work(j):
        sh, i, name, p, args = j
        if shutil.disk_usage("/").free < 2e9:
            return None  # stop under 2 GB free; finished runs are kept and the campaign resumes
        keep = name in RAW
        try:
            r = run_one(p, args, 1000 * (SHELLS.index(sh) + 1) + i, f"{sh}_{i}_{name}.csv", keep)
        except Exception as ex:
            r = {"error": str(ex)[:200]}
        if r is None:
            r = {"error": "no samples"}
        if "raw" in r:
            age, err = r.pop("raw")
            np.savez_compressed(os.path.join(raw_dir, f"{sh}__{name}__{i}.npz"), age=age, err=err)
        rec = dict(shell=sh, **{"pass": i}, config=name, bin=binhash, **r)
        with lock:
            with open(done_path, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
        return rec

    if a.holdout:
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(work, todo))
        print("holdout done", flush=True)
        return
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, _ in enumerate(ex.map(work, todo)):
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{len(todo)}", flush=True)
    res = {}
    for line in open(done_path):
        rec = json.loads(line)
        res[(rec["shell"], rec["pass"], rec["config"])] = rec
    fields = ["shell", "pass", "config", "samples", "reads", "infeasible", "viol_frac", "viol_te", "p99_ns", "max_ns",
              "mean_validity", "limit_ns", "ca_lat", "start", "bin", "error"]
    with open(os.path.join(OUT, "ns3_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in res.values():
            w.writerow({k: r.get(k, "") for k in fields})
    # (the per-run raw files are removed once merged; a rerun of a finished campaign must not overwrite the merge)
    if not os.listdir(raw_dir):  # no new raw profile: keep the merged file
        print("done", sum("error" in r for r in res.values()), "errors", flush=True)
        return
    ages = {}
    for f in os.listdir(raw_dir):
        sh, name, i = f[:-4].split("__")
        z = np.load(os.path.join(raw_dir, f))
        ages.setdefault(sh, {}).setdefault(name, []).append((z["age"], z["err"]))
    np.savez_compressed(os.path.join(OUT, "ns3_raw.npz"),
                        **{f"{sh}|{n}|age": np.concatenate([x[0] for x in v]) for sh, d in ages.items() for n, v in d.items()},
                        **{f"{sh}|{n}|err": np.concatenate([x[1] for x in v]) for sh, d in ages.items() for n, v in d.items()})
    print("done", sum("error" in r for r in res.values()), "errors", flush=True)


if __name__ == "__main__":
    main()
