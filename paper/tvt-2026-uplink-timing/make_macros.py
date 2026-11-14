#!/usr/bin/env python3
"""Write ../macros_p5.tex: every number the TVT manuscript cites, from the closed forms (ta_core), the
Monte Carlo (out/mc_summary.json, out/lookup.json), the high-fidelity check (out/hifi.json) and the ns-3
campaign (out_r4/ns3_runs.jsonl, one binary). Campaign statistics are clustered by pass: bootstrap over the
passes for means, ratios of means and violation shares; a rule-of-three bound where no pass violated."""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_ta  # noqa: E402
import ta_core as T  # noqa: E402

TE = T.TE[30]
S = json.load(open(os.path.join(HERE, "out", "mc_summary.json")))
HF = json.load(open(os.path.join(HERE, "out", "hifi.json")))
# two binaries, by configuration: rule rows from out_r5 (round 5 budget), all others from out_r4; regress_budget.py
# shows the round 5 binary reproduces the others
R4 = [json.loads(x) for x in open(os.path.join(HERE, "out_r4", "ns3_runs.jsonl"))]
R5 = [json.loads(x) for x in open(os.path.join(HERE, "out_r5", "ns3_runs.jsonl"))]
assert len({r["bin"] for r in R4}) == 1 and len({r["bin"] for r in R5}) == 1, "one binary per record file"
_new = {r["config"] for r in R5}
assert all("P-" == c[:2] or c[:2] in ("B2", "V-", "M-") for c in _new), sorted(_new)[:5]
R = [r for r in R4 if r["config"] not in _new] + R5
BINHASH = R5[0]["bin"]
BINOLD = R4[0]["bin"]
NERR = sum(1 for r in R if r.get("error"))
R = [r for r in R if not r.get("error")]
LOOKUP = json.load(open(os.path.join(HERE, "out", "lookup.json")))
RAWZ = os.path.join(HERE, "out_r4", "ns3_raw.npz")
SH = ["starlink-vleo", "starlink-480", "starlink-550", "oneweb-1200"]
TAG = {"starlink-vleo": "Vleo", "starlink-480": "FourSeventy", "starlink-550": "FiveSeventy", "oneweb-1200": "Oneweb"}
WORDS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
M = {}
rng = np.random.default_rng(2026)
NB = 4000


def f(x, n=1):
    return f"{x:.{n}f}"


def tag(cfg):
    return "".join(WORDS.get(ch, ch) for ch in cfg.title() if ch.isalnum())




# ---- closed forms and Monte Carlo
M["HVkm"] = f(T.h_sat_variant() / 1e3, 0)
M["VleoKm"] = f(S["starlink-vleo"]["h_km"], 0)
_all = T.load_tles(os.path.join(HERE, "data", "starlink_20261002.tle"))
_a = np.array([(T.MU / (x.no_kozai / 60) ** 2) ** (1 / 3) for x in _all])
_e = np.array([x.ecco for x in _all])
M["NBelowHV"] = str(int(np.sum(_a * (1 - _e) - T.RE < T.h_sat_variant())))
M["NStarlinkAll"] = f"{len(_all):,}".replace(",", "{,}")
M["VleoTvPct"] = f(100 * S["starlink-vleo"]["tv_max_obs"] / (T.MAX_TV * T.Q_TV), 0)
M["DriftObsMax"] = f(max(S[s]["td_max_obs"] for s in SH) * 1e6, 1)
h470 = S["starlink-480"]["h_km"] * 1e3
M["AccLin"] = f(T.residual_acc(h470, "linear"), 1)
M["AccGrav"] = f(T.residual_acc(h470, "gravity-ecef"), 2)
M["AccKep"] = f(T.residual_acc(h470, "kepler", lat=np.pi / 2), 3)
aj = S["starlink-480"]["kepler-j2"]["a_res_emp"]
M["AccKepJ"] = "$" + f(aj * 1e3, 2) + r"\times10^{-3}$"
M["AccKepJMax"] = "$" + f(max(S[s]["kepler-j2"]["a_res_emp"] for s in SH) * 1e3, 2) + r"\times10^{-3}$"
M["TauLinSec"] = f(T.tau_star(T.residual_acc(h470, "linear"), TE), 1)
m3 = T.ta3_max(S["starlink-480"]["h_min_km"] * 1e3, np.radians(25))
M["MthreeFourEighty"] = f(m3 * 1e9, 1)
M["HMinFourEighty"] = f(S["starlink-480"]["h_min_km"], 0)
M["TauTaySec"] = f((6 * TE / m3) ** (1 / 3), 1)
M["GridFactorFive"], M["GridFactorFifteen"] = f(5 / 2.56, 2), f(15 / 12.8, 2)
M["GridAgeLo"] = f(5 % 2.56, 2)
M["RhoRanFour"] = f(60 / (2.56 * np.floor(5 / 2.56)), 1)
# M2: the union over the fixes used in a validity, and the orbit event with its velocity draw
from scipy.stats import norm  # noqa: E402
DELTA_G, A_V = 0.0042, 93e-9


def z_g(v, tg=1.0, dg=DELTA_G):
    return norm.isf(dg / (2 * np.ceil((v + 0.64) / tg)))  # N of Proposition 3 at the paper's lead


def a_u(v, tg=1.0, sig=3.0):
    return 2 * z_g(v, tg) * 2 * sig / T.C  # sigma_v = 2 sigma_g bounds the LOS deviation


_rng = np.random.default_rng(7)
_x = _rng.normal(size=(4000000, 3)) * np.array([1, 1 / 3, 1 / 3])
PE = float(np.mean(np.linalg.norm(_x, axis=1) > 3))
PV = float(2 * norm.sf(3))
M["DeltaOPos"], M["DeltaOVel"] = f(PE, 4), f(PV, 4)
M["DeltaO"], M["DeltaG"] = "0.0058", "0.0042"
assert PE + PV <= 0.0058 and 0.0058 + DELTA_G <= 0.01
M["ZgFive"], M["ZgMax"] = f(z_g(5), 2), f(z_g(240), 2)
M["AuFive"], M["AuSixty"], M["AuMax"] = f(a_u(5) * 1e9, 0), f(a_u(60) * 1e9, 0), f(a_u(240) * 1e9, 0)
_z1 = norm.isf(DELTA_G / 2)
M["ZgOne"] = f(_z1, 2)
M["PSupThirty"], M["PSupSixty"], M["PSupMax"] = (f(1 - (1 - 2 * norm.sf(_z1)) ** n, 2) for n in (30, 60, 240))
M["EqSixtyNs"] = f(0.5 * (T.Q_TC + T.Q_TD * 60 + T.Q_TV * 60 ** 2) * 1e9, 0)
M["AvNs"] = f(A_V * 1e9, 0)
for k, v in (("Car", 33.3), ("Hst", 138.9), ("Air", 250.0)):
    M[f"Tg{k}"] = f(T.C * A_V / (2 * v * np.cos(np.radians(25))), 2)
# M3: the part of Te left to the extrapolation, and the velocity term at the longest PV validity
LIM = TE - T.A_IMPL
M["AImplNs"], M["TePrimeNs"], M["TeNs"] = f(T.A_IMPL * 1e9, 0), f(LIM * 1e9, 0), f(TE * 1e9, 0)
M["BetaTePrimeNs"] = f(LIM * 0.5 * 1e9, 0)
M["AImplFifteenNs"] = f(12 * 64 * T.TC * 1e9, 0)
M["AImplVsatNs"] = f(3.5 * 64 * T.TC * 1e9, 0)
M["TePrimeVsatNs"] = f((T.TE[120] - 3.5 * 64 * T.TC) * 1e9, 0)
M["VelTermNs"] = f(2 * 0.5 * 0.06 * np.sqrt(3) * 240 / T.C * 1e9, 0)
M["OrbQuantVelTermNs"] = f(2 * 3e-4 * 240 / T.C * 1e9, 1)
# m3: a UE that labels its prediction with its own transmit instant
_h = 470e3
_psim = T.central_angle_max(_h, np.radians(25))
_d, _d1, _, _ = T.range_derivs(_h, 0.0, _psim)
M["LightBiasNs"] = f(2 * abs(_d1) * _d / T.C ** 2 * 1e9, 0)
M["LightRangeKm"] = f(_d / 1e3, 0)
M["NMC"] = f"{S['starlink-480']['n']:,}".replace(",", "{,}")
counts = {sh: len(mc_ta.shell_sats(sh)) for sh in SH}
for sh in SH:
    M[f"N{TAG[sh]}"] = f"{counts[sh]:,}".replace(",", "{,}")
    r = S[sh]
    for k, kt in (("taylor", "Tay"), ("fit", "Econ"), ("win", "Win")):
        M[f"V{kt}{TAG[sh]}"] = str(r["vmax"][k])
    M[f"VBoundTay{TAG[sh]}"] = str(r["vbound"]["taylor"])
    M[f"VBoundEcon{TAG[sh]}"] = str(r["vbound"]["fit"])
    for m, mt in (("linear", "Lin"), ("gravity-ecef", "Grav"), ("kepler", "Kep")):
        M[f"TauNN{mt}{TAG[sh]}"] = f(r[m]["tau99"], 1 if m == "linear" else 0)
        M[f"TauStar{mt}{TAG[sh]}"] = f(r[m]["tau_star"], 1)
    M[f"TauNNTayC{TAG[sh]}"] = f(r["taylor"]["tau99"], 1)
    M[f"TauStarTayC{TAG[sh]}"] = f(r["taylor"]["tau_star"], 1)
    M[f"UeSpan{TAG[sh]}"] = f(r["ue_span_med"], 0)
# m4: window validity against Taylor's continuous usable age at the same limit (the enumerated set
# rounds Taylor down, so the ratio of enumerated values overstates the gain)
for sh in SH:
    M[f"VTayLim{TAG[sh]}"] = str(S[sh]["vmax_lim"]["taylor"])
    M[f"VEconLim{TAG[sh]}"] = str(S[sh]["vmax_lim"]["fit"])
    M[f"VWinLim{TAG[sh]}"] = str(S[sh]["vmax_lim"]["win"])
    M[f"TauTayLim{TAG[sh]}"] = f(S[sh]["taylor"]["tau99_lim"], 1)
    M[f"VBoundTayLim{TAG[sh]}"] = str(S[sh]["vbound_lim"]["taylor"])
    M[f"VBoundEconLim{TAG[sh]}"] = str(S[sh]["vbound_lim"]["fit"])
gain = [S[s]["vmax_lim"]["win"] / S[s]["taylor"]["tau99_lim"] for s in SH]
M["WinGainLo"], M["WinGainHi"] = f(min(gain), 1), f(max(gain), 1)
gain_enum = [S[s]["vmax_lim"]["win"] / max(S[s]["vmax_lim"]["taylor"], 1) for s in SH]
M["WinEnumLo"], M["WinEnumHi"] = f(min(gain_enum), 1), f(max(gain_enum), 1)
# tables frozen for the lead the paper's recipe uses (run_ns3_campaign.py, LEAD), and the replay gate at no lead
LL = json.load(open(os.path.join(HERE, "out", "lookup_lead.json")))
LEAD = "0.64"
M["LeadS"] = LEAD


def tab(kind, lim, lead=LEAD, key="worst"):
    return [max([int(v) for v, x in LL[key][sh][kind][lead].items() if x is not None and x <= lim] or [0]) for sh in SH]


for sh in SH:
    for v in ("5", "10", "15", "30"):
        assert abs(LL["worst"][sh]["wide"]["0.0"][v] - LOOKUP["worst_ns"][sh][v]) < 1e-6, (sh, v)
_lb, _lf = tab("wide", LIM * 0.5e9), tab("wide", LIM * 1e9)
for i, sh in enumerate(SH):
    M[f"LookB{TAG[sh]}"], M[f"LookF{TAG[sh]}"] = str(_lb[i]), str(_lf[i])
M["LookCellB"], M["LookCellF"] = str(min(_lb)), str(min(_lf))
_tq = [max(x, 5) for x in tab("taylor", LIM * 1e9)]
for i, sh in enumerate(SH):
    M[f"LookT{TAG[sh]}"] = str(_tq[i])
M["NLookLo"] = str(min(LOOKUP["n"][sh]["60"] for sh in SH))
M["NLookHi"] = str(max(LOOKUP["n"][sh]["5"] for sh in SH))
# M1: what the lead does. Window validity at the bound (99 % of cases over [-L, V], as Fig. 3) against Taylor's usable age
_vp = tab("wide", LIM * 1e9, key="p99")
_g = [v / S[s]["taylor"]["tau99_lim"] for v, s in zip(_vp, SH)]
M["WinGainLeadLo"], M["WinGainLeadHi"] = f(min(_g), 1), f(max(_g), 1)
for i, sh in enumerate(SH):
    M[f"VWinLead{TAG[sh]}"] = str(_vp[i])
# growth of a fit on [0, V] before its epoch, against the closed form (Corollary 3), 470 km V = 15 s and 357 km V = 10 s
M["NaiveHfive"] = f(LL["worst"]["starlink-480"]["naive"]["5.0"]["15"] / 1e3, 1)
M["NaiveVleoHfive"] = f(LL["worst"]["starlink-vleo"]["naive"]["5.0"]["10"] / 1e3, 1)
for L, t in (("0.32", "A"), ("0.64", "B"), ("1.28", "C"), ("2.56", "D"), ("5.0", "E")):
    x = 1 + 2 * float(L) / 15
    _m3 = T.ta3_max(S["starlink-480"]["h_min_km"] * 1e3, np.radians(25))
    M[f"NaiveMc{t}"] = f(LL["worst"]["starlink-480"]["naive"][L]["15"], 0)
    M[f"NaivePn{t}"] = f(LL["p99"]["starlink-480"]["naive"][L]["15"], 0)
    M[f"TaylorMc{t}"] = f(LL["worst"]["starlink-480"]["taylor"][L]["15"], 0)
    M[f"NaiveCf{t}"] = f(_m3 / 6 * 7.5 ** 3 * (4 * x ** 3 - 3 * x) / 4 * 1e9, 0)
    M[f"TThree{t}"] = f(4 * x ** 3 - 3 * x, 1)
    M[f"WideMc{t}"] = f(LL["worst"]["starlink-480"]["wide"][L]["15"], 0)
    for name, key in (("Table", "wide"),):
        _t = tab(key, LIM * 0.5e9, L)
        M[f"LeadTab{t}"] = ", ".join(str(max(v, 5)) for v in _t)
M["LeadTabZero"] = ", ".join(str(max(v, 5)) for v in tab("wide", LIM * 0.5e9, "0.0"))
M["WideMcZero"] = f(LL["worst"]["starlink-480"]["wide"]["0.0"]["15"], 0)
assert round(LIM / (64 * T.TC)) == 14
vc = [S[s]["v_cross"] for s in SH]
M["VCrossLo"], M["VCrossHi"] = f(min(vc), 0), f(max(vc), 0)
J = S["starlink-480"]["kepler-j2"]
M["JTwoAtSixty"], M["JTwoAtOneTwenty"] = f(J["p99_ns"]["60"], 0), f(J["p99_ns"]["120"], 0)
M["JTwoNAtTwoForty"] = str(J["n_at"].get("240", 0))
for sh in SH:
    M[f"JTwoN{TAG[sh]}"] = str(S[sh]["kepler-j2"]["n_at"].get("240", 0))
M["HifiOneTwenty"] = f(max(HF[s]["120"] for s in HF), 1)
M["HifiTwoForty"] = f(max(HF[s]["240"] for s in HF), 1)
gaps = []
for sh in SH:
    for m in ("linear", "gravity-ecef", "kepler"):
        gaps.append(100 * (S[sh][m]["tau99"] - S[sh][m]["tau_star"]) / S[sh][m]["tau99"])
    gaps.append(100 * (S[sh]["taylor"]["tau99"] - S[sh]["taylor"]["tau_star"]) / S[sh]["taylor"]["tau99"])
M["MaxBoundGapPct"], M["MinBoundGapPct"] = f(max(gaps), 0), f(min(gaps), 0)
ex = [S[sh][m]["bound_exceed_frac"] for sh in SH for m in ("linear", "gravity-ecef", "kepler")] + \
     [S[sh]["taylor"]["bound_exceed_frac"] for sh in SH]
M["BoundExceedMaxPct"] = f(100 * max(ex), 2)


# ---- ns-3 campaign, clustered by pass
def by_pass(cfg):
    return {(r["shell"], r["pass"]): r for r in R if r["config"] == cfg}


def boot_mean(vals):
    vals = np.asarray(vals, float)
    b = [vals[rng.integers(len(vals), size=len(vals))].mean() for _ in range(NB)]
    return vals.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


CFGS = sorted({r["config"] for r in R})
M["NPasses"] = str(len({(r["shell"], r["pass"]) for r in R}))
M["NPassShell"] = str(len({r["pass"] for r in R}))
M["NStress"] = str(len(by_pass("P-gw-b25")))
M["NStressShell"] = str(len(by_pass("P-gw-b25")) // 4)
M["NRuns"] = f"{len(R):,}".replace(",", "{,}")
M["NCfg"] = str(len(CFGS))
M["RuleOfThreeCore"] = f(100 * 3 / len(by_pass("P-gw-window")), 2)
M["RuleOfThreeStress"] = f(100 * 3 / max(1, len(by_pass("P-gw-b25"))), 1)
lat = np.array([abs(r["ca_lat"]) for r in by_pass("P-gw-window").values()])
M["LatMed"], M["LatMax"] = f(np.median(lat), 0), f(lat.max(), 0)
M["LatHighPct"] = f(100 * np.mean(lat > 45), 0)
for cfg in CFGS:
    P = by_pass(cfg)
    keys = sorted(P)
    reads = np.array([float(P[k]["reads"]) for k in keys])
    samp = np.array([float(P[k]["samples"]) for k in keys])
    viol = np.array([float(P[k]["viol_frac"]) for k in keys]) * samp
    vte = np.array([float(P[k]["viol_te"]) for k in keys]) * samp
    t = tag(cfg)
    m, lo, hi = boot_mean(reads)
    M[f"Reads{t}"], M[f"Reads{t}Lo"], M[f"Reads{t}Hi"] = f(m, 1), f(lo, 1), f(hi, 1)
    M[f"Avoid{t}"] = f(np.mean(reads - 1), 1)
    share = viol.sum() / samp.sum()
    shte = vte.sum() / samp.sum()
    M[f"Viol{t}"] = ("$<$0.01" if 0 < share < 5e-5 else f(100 * share, 2 if share < 0.1 else 1))
    M[f"ViolTe{t}"] = ("$<$0.01" if 0 < shte < 5e-5 else f(100 * shte, 2 if shte < 0.1 else 1))
    npass = int(np.sum(viol > 0))
    M[f"PassViol{t}"] = str(npass)
    M[f"PassViolTe{t}"] = str(int(np.sum(vte > 0)))
    M[f"PassViolBound{t}"] = f(100 * 3 / len(keys), 1) if npass == 0 else f(100 * npass / len(keys), 1)
    M[f"Pnn{t}"] = f(np.median([float(P[k]["p99_ns"]) for k in keys]), 0)
    M[f"Max{t}"] = f(max(float(P[k]["max_ns"]) for k in keys), 0)
    M[f"Val{t}"] = f(np.mean([float(P[k]["mean_validity"]) for k in keys]), 1)
    nf = int(sum(float(P[k]["infeasible"]) for k in keys))
    M[f"Infeas{t}"] = str(nf)


def ratio_of_means(a, b, shell=None):
    A, B = by_pass(a), by_pass(b)
    ks = sorted(k for k in A if k in B and (shell is None or k[0] == shell))
    x = np.array([float(A[k]["reads"]) for k in ks])
    y = np.array([float(B[k]["reads"]) for k in ks])
    bs = []
    for _ in range(NB):
        i = rng.integers(len(ks), size=len(ks))
        bs.append(x[i].mean() / y[i].mean())
    return x.mean() / y.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def gain(a, b, name, d=1, shell=None):
    g, lo, hi = ratio_of_means(a, b, shell)
    M[name], M[name + "Lo"], M[name + "Hi"] = f(g, d), f(lo, d), f(hi, d)
    return g


# The fit against Taylor at one validity and one grid; the rule against its strongest rivals
gain("T-shell", "W-shell", "GainFit")
gain("T-shell-L0", "W-shell-L0", "GainFitNoLead")
LTAG = {"0.32": "A", "1.28": "C", "2.56": "D", "5": "E"}
for L, t in LTAG.items():
    u = L.replace(".", "p")
    gain(f"T-shell-L{u}", f"W-shell-L{u}", f"GainLead{t}")
gain("W-shell-od10", "P-satf-od10-ign", "GainSatfTen")
WORDSA = {0: "Zero", 130: "OneThreeZero", 390: "ThreeNineZero"}
_g = [gain(f"T-shell-a{a}", f"W-shell-a{a}", f"GainFitA{WORDSA[a]}") for a in (0, 130, 390)]
_g.append(float(M["GainFit"]))
M["GainFitALo"], M["GainFitAHi"] = f(min(_g), 1), f(max(_g), 1)
gain("T5", "W-shell", "GainFitTFive")
gain("T-shell-grid", "W-shell-grid", "GainFitGrid")
gain("B1-r", "B1", "GainBOneR")
gain("P-gw-od30", "P-gw-od30-ign", "GainOdBudget")
_g = [gain("T-shell", "W-shell", f"GainFit{TAG[sh]}", 2, sh) for sh in SH]
M["GainFitShellLo"], M["GainFitShellHi"] = f(min(_g), 2), f(max(_g), 2)
_g = [gain("W-shell-full", "P-gw-window", f"GainRuleFull{TAG[sh]}", 2, sh) for sh in SH]
M["GainRuleFullShellLo"], M["GainRuleFullShellHi"] = f(min(_g), 2), f(max(_g), 2)
_a, _b = by_pass("W-shell-full"), by_pass("P-gw-window")
M["FullBeatsPct"] = f(100 * np.mean([float(_a[k]["reads"]) < float(_b[k]["reads"]) for k in _a]), 0)
gain("T5-grid", "T5", "GridLossGw", 2)
gain("B0", "B5", "GridLossSat", 2)
gain("T5", "P-gw-window", "GainWin")
gain("T5-grid", "P-gw-window", "GainWinGrid")
gain("P-gw-taylor", "P-gw-window", "GainWinTay", 2)
gain("W-shell", "P-gw-window", "GainRule", 2)
gain("W-shell-full", "P-gw-window", "GainRuleFull", 2)
gain("W-cell", "P-gw-window", "GainRuleCell", 2)
gain("W-shell-grid", "W-shell", "GridLossShell", 2)
gain("B5", "P-sat", "GainSat")
gain("B0", "P-sat", "GainSatGrid")
gain("S-kepler-j2", "P-sat", "RuleVsFix", 2)
_g = []
for sh in SH:
    _g.append(gain("W-shell", "P-gw-window", f"GainRule{TAG[sh]}", 2, sh))
M["GainRuleShellLo"], M["GainRuleShellHi"] = f(min(_g), 2), f(max(_g), 2)
M["SlotViolPct"] = M["ViolLSlot"]

# M1: ns-3 against the Monte Carlo at 470 km, weak-propagator stress bands (ns-3 has 3 m GNSS, the MC none)
Z = {}
if os.path.exists(RAWZ):
    Z = np.load(RAWZ)
if True:
    d = np.load(os.path.join(HERE, "out", "mc_starlink-480.npz"))
    taus = d["taus"]

    def band(cfg, mc, lo, hi):
        a, e = Z[f"starlink-480|{cfg}|age"], Z[f"starlink-480|{cfg}|err"]
        ns3 = np.percentile(e[(a >= lo) & (a < hi)], 99)
        col = np.abs(d[mc])[:, (taus >= lo) & (taus < hi)]
        return ns3, np.nanpercentile(col, 99) * 1e9

    a_, b_ = band("S-kepler", "eph_kepler", 60, 120)
    M["BandKepNs"], M["BandKepMc"] = f(a_, 0), f(b_, 0)
    a_, b_ = band("S-kepler-j2", "eph_kepler-j2", 2, 5)
    M["BandJTwoNs"], M["BandJTwoMc"] = f(a_, 0), f(b_, 0)
    a_, b_ = band("S-kepler-j2", "eph_kepler-j2", 60, 120)
    M["BandJTwoLongNs"], M["BandJTwoLongMc"] = f(a_, 0), f(b_, 0)
rt = os.path.join(HERE, "out_r5", "runtime.json")
if os.path.exists(rt):
    q = json.load(open(rt))
    M["RuleMsPerEpoch"] = f(q["ms_per_epoch"], 1)
else:
    raise SystemExit("run time_rule.py first")

M["PassAboveVsatWCell"] = str(sum(float(r["max_ns"]) > float(M["TePrimeVsatNs"]) for r in by_pass("W-cell").values()))
M["NaiveCfOverPct"] = f(100 * max(float(M[f"NaiveCf{t}"]) / float(M[f"NaiveMc{t}"]) - 1 for t in "BCDE"), 0)
M["GainRuleOne"] = M["GainRule"]
_E5 = (2 * 0.0839 * 5 + 2 * (0.5 * 1.3 * 3 ** .5 + 0.5 * 0.06 * 3 ** .5 * 5)) / T.C
_A = lambda sg: 2 * norm.isf(DELTA_G / (2 * np.ceil(5.64))) * 2 * sg / T.C
M["BetaMaxFive"] = f(1 - (_A(5) + _E5) / LIM, 3)
M["AuFiveTen"] = f(_A(10) * 1e9, 0)
assert 1 - (_A(10) + _E5) / LIM < 0
gain("B5", "S-900", "GainSatFix", 1)
gain("B0", "S-900", "GainSatGridFix", 1)
gain("P-sat", "S-900", "RuleOverFix", 2)
gain("P-sat-a0", "S-900", "RuleOverFixAZero", 2)
gain("P-sat-a390", "S-900", "RuleOverFixAThreeNineZero", 2)
gain("P-gw-od10", "W-shell-od10", "GainOdTenGw", 1)
gain("S-kepler-j2", "S-900", "TwoFortyOverFix", 2)
gain("W-shell-od10", "S-900-satf-od10", "GainSatfNine", 1)
gain("W-shell-g5", "P-gw-g5-b40", "GainRuleFiveFourty", 2)
gain("W-shell-g5", "P-gw-g5-b25", "GainRuleFive", 2)
_g = [gain("W-shell-g5", "P-gw-g5-b25", f"GainRuleFive{TAG[sh]}", 2, sh) for sh in SH]
M["GainRuleFiveShellLo"], M["GainRuleFiveShellHi"] = f(min(_g), 2), f(max(_g), 2)
M["MarginSNine"] = f(LIM * 1e9 - float(M["MaxSNineZeroZero"]), 0)
_W = 0.0839  # kWJ2 in ntn-sib19.cc, from out/ares_train.json rounded up at the third figure
assert json.load(open(os.path.join(HERE, "out", "ares_train.json")))["w_max"] <= _W
def _budget(V, h, sg=3.0):  # A_u + E_s of (vs) for a Kepler-J2 UE, at served ages
    r = T.RE + h; om = (T.MU / r ** 3) ** .5
    ph = np.arccos(T.RE * np.cos(np.radians(25)) / r) - np.radians(25)
    t = min(V, 2 * ph / (om - 7.292115e-5) + 5.0)
    return 2 * norm.isf(DELTA_G / (2 * np.ceil(t + 0.64))) * 2 * sg / T.C + (2 * _W * t + 2 * (0.5 * 1.3 * 3 ** .5 + 0.5 * 0.06 * 3 ** .5 * t)) / T.C, t
_es240 = _budget(240, 470e3)[0]; _au240 = 0.0
M["BudgetTwoForty"] = f((_au240 + _es240) * 1e9, 0)
gain("W-shell", "P-sat-kepler", "GainKepSat", 1)
# round 3
M["BandKepRep"] = f(json.load(open(os.path.join(HERE, "out", "replica.json")))["p99"], 0)
M["MaxGwOdThirty"] = f(max(float(r["max_ns"]) for r in R if r["config"] in ("P-gw-od30", "P-gw-od30-ign", "W-shell-od30")), 0)
BASE = [c for c in CFGS if not c.endswith(("-g1", "-g5", "-g10", "-tx"))]
_zero = [c for c in CFGS if not any(float(r["viol_frac"]) > 0 for r in by_pass(c).values())]
M["NZeroViol"] = str(len(_zero))
M["SimBoundCore"] = f(100 * (1 - (0.05 / len(CFGS)) ** (1 / 400)), 1)  # the whole family: the zero-violation ones are selected on outcome
M["SimBoundStress"] = f(100 * (1 - (0.05 / len(CFGS)) ** (1 / 160)), 1)
_w = by_pass("W-shell"); _t = by_pass("T-shell")
_ks = sorted(_w)
_rw = np.array([1 + (float(_w[k]["reads"]) - 1) * 1.5 if k[0] == "starlink-480" else float(_w[k]["reads"]) for k in _ks])
M["GainFitSens"] = f(np.mean([float(_t[k]["reads"]) for k in _ks]) / _rw.mean(), 1)
M["GainFitSensCell"] = f(_lb[SH.index("starlink-480")], 0) + "~s"
M["AirSavedS"] = f((float(M["ReadsTShell"]) - float(M["ReadsWShell"])) * 0.010, 2)
assert _lb[SH.index("starlink-480")] == 15
# round 4: per shell counts, per-SIB19 ranges, OneWeb bias
def _ow(c):
    return str(sum(float(r["viol_frac"]) > 0 for k, r in by_pass(c).items() if k[0] == "oneweb-1200"))
M["PassViolWShellFullOW"], M["PassViolWShellFullTxOW"], M["PassViolPGwWindowTxOW"] = _ow("W-shell-full"), _ow("W-shell-full-tx"), _ow("P-gw-window-tx")
_d, _d1, _, _ = T.range_derivs(1201e3, 0.0, T.central_angle_max(1201e3, np.radians(25)))
M["LightBiasOnewebNs"] = f(2 * abs(_d1) * _d / T.C ** 2 * 1e9, 0)
M["LightBiasOnewebPct"] = f(100 * 2 * abs(_d1) * _d / T.C ** 2 / LIM, 0)
def _persib(c):
    P_ = by_pass(c)
    n = sum(float(r["reads"]) for r in P_.values())
    return 100 * sum(float(r["viol_frac"]) > 0 for r in P_.values()) / n, 100 * min(n, sum(float(r["viol_frac"]) * float(r["samples"]) for r in P_.values())) / n
_l, _h = _persib("P-sat-vsat"); M["FRTwoSatLo"], M["FRTwoSatHi"] = f(_l, 2), f(_h, 2)
_l, _h = _persib("P-gw-vsat"); M["FRTwoGwLo"], M["FRTwoGwHi"] = f(_l, 2), f(_h, 2)
# round 5: the constant, its shape test on the training draws, budget at the longest served age by shell, velocity sweep,
# Monte Carlo check of no expiry, pass-length numerical orbit
_tr = json.load(open(os.path.join(HERE, "out", "ares_train.json")))
M["WRes"] = "0.0839"
M["WSlopeLo"], M["WSlopeHi"] = f(min(_tr["slopes"].values()), 2), f(max(_tr["slopes"].values()), 2)
for sh, h in (("starlink-vleo", 357e3), ("starlink-480", 470e3), ("starlink-550", 571e3), ("oneweb-1200", 1201e3)):
    _b, _t = _budget(900, h)
    M[f"BudgetCap{TAG[sh]}"], M[f"CapAge{TAG[sh]}"] = f(_b * 1e9, 0), f(_t, 0)
_mcs = {}
for sh in SH:
    _d = np.load(os.path.join(HERE, "out", f"mc_{sh}.npz"))
    _mcs[sh] = float(np.nanmax(np.abs(_d["eph_kepler-j2"])) * 1e9)
_a900 = 2 * norm.isf(DELTA_G / (2 * np.ceil(900.64))) * 6.0 / T.C * 1e9
M["AuNineH"] = f(_a900, 0)
M["McNineLo"], M["McNineHi"] = f(min(_mcs.values()) + _a900, 0), f(max(_mcs.values()) + _a900, 0)
_hp = json.load(open(os.path.join(HERE, "out", "hifi_pass.json")))
M["HifiPassLeo"], M["HifiPassOw"] = f(_hp["starlink-480"]["max_ns"], 0), f(_hp["oneweb-1200"]["max_ns"], 0)
M["CapOverFiveSeventy"] = f(float(M["BudgetCapFiveSeventy"]) - LIM * 1e9, 1) if False else f(_budget(900, 571e3)[0] * 1e9 - LIM * 1e9, 1)
M["WFlip"] = "0.0832"
_E10 = (2 * 0.0839 * 10 + 2 * (0.5 * 1.3 * 3 ** .5 + 0.5 * 0.06 * 3 ** .5 * 10)) / T.C
M["BetaMaxTen"] = f(1 - (2 * norm.isf(DELTA_G / (2 * np.ceil(10.64))) * 2 * 5 / T.C + _E10) / LIM, 3)
_r = {"starlink-vleo": 1.0, "starlink-480": 1.0, "starlink-550": 1.0}
_ps = by_pass("P-sat"); _s9 = by_pass("S-900")
M["RuleOverFixDg"] = f(np.mean([(1.0 if k[0] in _r else float(_ps[k]["reads"])) for k in _ps]) / np.mean([float(_s9[k]["reads"]) for k in _ps]), 2)
M["BinNew"], M["BinOld"] = BINHASH[:8], BINOLD[:8]
for c, t in (("P-sat", "PSat"), ("P-sat-g5", "PSatGF")):
    for sh in SH:
        x = [r for r in R if r["config"] == c and r["shell"] == sh]
        M[f"Val{t}{TAG[sh]}"] = f(np.mean([float(r["mean_validity"]) for r in x]), 0)
        M[f"Reads{t}{TAG[sh]}"] = f(np.mean([float(r["reads"]) for r in x]), 2)
# holdout: fresh element sets and a new seed, same frozen tables
_H5 = [json.loads(x) for x in open(os.path.join(HERE, "out_r5", "holdout_runs.jsonl"))]
_hn = {r["config"] for r in _H5}
H = [r for r in map(json.loads, open(os.path.join(HERE, "out_r4", "holdout_runs.jsonl"))) if r["config"] not in _hn] + _H5
assert not any(r.get("error") for r in H)
def _hm(c):
    return np.mean([float(r["reads"]) for r in H if r["config"] == c])
M["HoldDateText"] = "3 to 4~November~2026"
M["HoldPasses"] = str(len({(r["shell"], r["pass"]) for r in H}))
M["GainFitHold"] = f(_hm("T-shell") / _hm("W-shell"), 1)
M["GainRuleHold"] = f(_hm("W-shell") / _hm("P-gw-window"), 2)
for c, t in (("S-900", "SNine"), ("S-kepler-j2", "STwoForty"), ("W-shell", "WShell"), ("T-shell", "TShell"), ("P-gw-window", "PGw"), ("W-shell-full", "WFull"), ("P-sat", "PSat"), ("T5", "TFive")):
    x = [r for r in H if r["config"] == c]
    M[f"HoldMax{t}"] = f(max(float(r["max_ns"]) for r in x), 0)
    M[f"HoldPass{t}"] = str(sum(float(r["viol_frac"]) > 0 for r in x))
    M[f"HoldReads{t}"] = f(_hm(c), 1)
M["MaxWNaiveLFiveUs"] = f(float(M["MaxWNaiveLFive"]) / 1e3, 1)
with open(os.path.join(HERE, "..", "macros_p5.tex"), "w") as fo:
    fo.write("% written by sim/make_macros.py; do not edit\n")
    for k in sorted(M):
        fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(len(M), "macros;", len(R), "runs;", NERR, "errors; bin", BINHASH[:16])
