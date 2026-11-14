#!/usr/bin/env python3
"""Result figures of the TVT manuscript (Figs. 2-4), from out/mc_*.npz, out/mc_summary.json and
out_r4/ns3_runs.jsonl with the rule rows of out_r5, out_r4/ns3_raw.npz. Drawn at final size (3.5 in column, 7.16 in page)."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as FS  # noqa: E402
import ta_core as T  # noqa: E402
from figstyle import INK, INK2, COL, PAGE  # noqa: E402

FS.setup()
OUT = os.path.join(HERE, "out")
FIG = os.path.join(HERE, "..", "figures")
TE = T.TE[30]
LIM = TE - T.A_IMPL
SHELLS = ["starlink-vleo", "starlink-480", "starlink-550", "oneweb-1200"]
SHORT = {"starlink-vleo": "SL-357", "starlink-480": "SL-470", "starlink-550": "SL-571", "oneweb-1200": "OW-1201"}
LAB = {"linear": "Linear", "gravity-ecef": "Gravity-ECEF", "kepler": "Kepler", "kepler-j2": "Kepler-J2",
       "taylor": "Taylor", "econ": "Economized", "window": "Window fit", "sat": "Satellite ref."}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def p99(e, i):
    col = e[:, i]
    col = col[np.isfinite(col)]
    return np.percentile(col, 99) if len(col) >= 50 else np.nan


def te_line(ax, label=True):
    ax.axhline(TE * 1e9, **FS.THRESH)
    ax.axhline(LIM * 1e9, **FS.THRESH)
    if label:
        ax.text(0.02, TE * 1e9 * 1.15, r"$T_{\mathrm{e}}$", transform=ax.get_yaxis_transform(), color=FS.RED,
                fontsize=7, va="bottom")
        ax.text(0.02, LIM * 1e9 * 0.87, r"$T_{\mathrm{e}}'$", transform=ax.get_yaxis_transform(), color=FS.RED,
                fontsize=7, va="top")


def fig_validation(shell="starlink-480"):
    d = np.load(os.path.join(OUT, f"mc_{shell}.npz"))
    taus, tausf = d["taus"], d["tausf"]
    hmin = float(np.percentile(d["meta"][:, 0], 1))
    raw = np.load(os.path.join(HERE, "out_r4", "ns3_raw.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 2.05))
    ax = axs[0]
    pts = [2, 5, 10, 20, 30, 60, 120, 180, 240, 300]
    tt = np.linspace(1, 300, 400)
    qp, qv = 0.5 * 1.3 * np.sqrt(3), 0.5 * 0.06 * np.sqrt(3)
    bins = np.array([0, 3, 7, 14, 25, 45, 90, 150, 210, 270, 330])
    centers = np.array([2, 5, 10, 20, 30, 60, 120, 180, 240, 300])
    for m in ["linear", "gravity-ecef", "kepler", "kepler-j2"]:
        c_, mk, ls = FS.METHOD[m]
        if m != "kepler-j2":
            a = T.residual_acc(hmin, m, lat=np.pi / 2 if m == "kepler" else 0.0)
            ax.plot(tt, (a * tt ** 2 / T.C + 2 * (qp + qv * tt) / T.C) * 1e9, color=c_, ls=ls, lw=1.0)
        e = np.abs(d[f"eph_{m}"])
        ax.plot(pts, [p99(e, int(p)) * 1e9 for p in pts], ls="none", marker=mk, color=c_, mfc=c_, ms=3.6,
                label=LAB[m])
        ka, ke = f"{shell}|S-{m}|age", f"{shell}|S-{m}|err"
        if ka in raw.files:
            age, err = raw[ka], raw[ke]
            y = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (age >= lo) & (age < hi)
                y.append(np.percentile(err[sel], 99) if sel.sum() >= 50 else np.nan)
            ax.plot(centers, y, ls="none", marker=mk, color=c_, mfc="white", ms=4.6, mew=0.8)
    te_line(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 330)
    ax.set_ylim(0.1, 3e6)
    ax.set_xlabel(r"Information age $\tau$ (s)")
    ax.set_ylabel(r"UE-side error $|e_{\mathrm{s}}|$ (ns)")
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    ax.text(0.98, 0.04, "lines: closed-form bound\nfilled: MC, 99th pct.\nhollow: ns-3, 99th pct.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_title("(a) UE propagators", fontsize=8)
    ax = axs[1]
    m3 = T.ta3_max(hmin, np.radians(25))
    tt = np.linspace(0.5, 30, 200)
    c_, mk, ls = FS.METHOD["taylor"]
    ax.plot(tt, m3 * tt ** 3 / 6 * 1e9, color=c_, ls=ls)
    ef = np.abs(d["fine_taylor"])
    sel = np.arange(0, len(tausf), 4)
    ax.plot(tausf[sel][1:], [np.nanpercentile(ef[:, i], 99) * 1e9 for i in sel[1:]], ls="none", marker=mk,
            color=c_, mfc=c_, ms=3.0, label="Taylor")
    S = json.load(open(os.path.join(OUT, "mc_summary.json")))[shell]["win"]
    Vs = sorted(int(v) for v in S if int(v) <= 30)
    c_, mk, ls = FS.METHOD["econ"]
    ax.plot(tt, [T.econ_bound(hmin, np.radians(25), v) * 1e9 for v in tt], color=c_, ls=ls)
    ax.plot(Vs, [S[str(v)]["fit"]["p99_max_ns"] for v in Vs], ls="none", marker=mk, color=c_, mfc=c_,
            label="Economized")
    c_, mk, ls = FS.METHOD["window"]
    ax.plot(Vs, [S[str(v)]["win"]["p99_max_ns"] for v in Vs], ls="none", marker=mk, color=c_, mfc=c_,
            label="Window fit")
    te_line(ax)
    ax.set_yscale("log")
    ax.set_xlim(0, 31)
    ax.set_ylim(1, 1e5)
    ax.set_xlabel(r"Age $\tau$ (Taylor) or window $V$ (fits) (s)")
    ax.set_ylabel(r"Common-delay error $|e_{\mathrm{c}}|$ (ns)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.text(0.98, 0.04, "lines: closed-form bound\nmarkers: MC, 99th pct.", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_title("(b) Common delay, gateway reference", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig2_validation.pdf")


def fig_validity():
    S = json.load(open(os.path.join(OUT, "mc_summary.json")))
    hh = np.linspace(330e3, 1300e3, 60)
    m3 = np.array([T.ta3_max(h, np.radians(25)) for h in hh])
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    c_, mk, ls = FS.METHOD["taylor"]
    ax.plot(hh / 1e3, (6 * LIM / m3) ** (1 / 3), color=c_, ls=ls, label="Taylor, bound")
    vv = np.linspace(1, 80, 400)
    econ = [vv[np.argmax([T.econ_bound(h, np.radians(25), v) > LIM for v in vv]) - 1] for h in hh]
    c_, mk, ls = FS.METHOD["econ"]
    ax.plot(hh / 1e3, econ, color=c_, ls=ls, label="Economized, bound")
    for key, k2, dx in (("taylor", "taylor", -8), ("econ", "fit", 0), ("window", "win", 8)):
        c_, mk, _ = FS.METHOD[key]
        xs = [S[sh]["h_km"] + dx for sh in SHELLS]
        ys = [S[sh]["vmax_lim"][k2] for sh in SHELLS]
        ax.plot(xs, ys, marker=mk, color=c_, ls="none", mfc=c_, label=f"{LAB[key]}, MC")
    ax.set_yscale("log")
    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Validity of the common term (s)")
    ax.set_ylim(3, 100)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig3_validity.pdf")


def load_runs():
    R5 = [json.loads(x) for x in open(os.path.join(HERE, "out_r5", "ns3_runs.jsonl"))]
    new = {r["config"] for r in R5}
    R = [r for r in map(json.loads, open(os.path.join(HERE, "out_r4", "ns3_runs.jsonl"))) if r["config"] not in new] + R5
    return [r for r in R if not r.get("error")]


def mean_ci(x):
    x = np.asarray(x, float)
    from scipy import stats
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
    return m, h


SCHEMES = [("B0", INK, "B0", "///"), ("B1", FS.GREY, "B1", None), ("P-sat", FS.TEAL, "Rule, sat.", None), ("S-900", INK2, "V=900 s", "xxx"),
           ("T5-grid", FS.ORANGE, "T5-g", "///"), ("T5", FS.ORANGE, "T5", None), ("T60", FS.GOLD, "T60", None),
           ("L-slot", FS.RED, "L", None), ("T-shell", "#7f7f7f", "T-shell", "..."), ("W-cell", FS.PURPLE, "W-cell", None), ("W-shell", FS.GREEN, "W-shell", None),
           ("W-shell-full", "#8fbf8f", "W-shell-f", None), ("P-gw-taylor", "#c9a27e", "Rule, Taylor", None), ("P-gw-window", FS.BLUE, "Rule, window", None)]


def fig_baselines():
    R = load_runs()
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 2.0))
    w = 0.063
    n = len(SCHEMES)
    for k, (cfg, col, lab, hatch) in enumerate(SCHEMES):
        for j, sh in enumerate(SHELLS):
            rows = [r for r in R if r["config"] == cfg and r["shell"] == sh]
            if not rows:
                continue
            m, h = mean_ci([float(r["reads"]) for r in rows])
            axs[0].bar(j + (k - (n - 1) / 2) * w, m, w, yerr=h, color=col, edgecolor="white", lw=0.3, hatch=hatch,
                       error_kw=dict(lw=0.5, capsize=1), label=lab if j == 0 else None)
            v = 100 * np.sum([float(r["viol_frac"]) * float(r["samples"]) for r in rows]) / \
                np.sum([float(r["samples"]) for r in rows])
            if v > 0.01:
                axs[1].bar(j + (k - (n - 1) / 2) * w, v, w, color=col, edgecolor="white", lw=0.3, hatch=hatch)
    for ax in axs:
        ax.set_xticks(range(len(SHELLS)))
        ax.set_xticklabels([SHORT[s] for s in SHELLS])
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("SIB19 decodings per pass")
    axs[1].set_ylabel(r"Time with $|e|>T_{\mathrm{e}}'$ (%)")
    axs[1].set_ylim(0.01, 100)
    axs[0].set_title("(a) SIB19 decodings", fontsize=8)
    axs[1].set_title("(b) Violation share", fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=14, fontsize=6.2, bbox_to_anchor=(0.5, 1.07),
               handlelength=1.2, columnspacing=0.7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_baselines.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["validation", "validity", "baselines"]
    for w in which:
        globals()[f"fig_{w}"]()
