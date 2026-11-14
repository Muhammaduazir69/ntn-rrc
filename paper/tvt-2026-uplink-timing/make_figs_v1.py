#!/usr/bin/env python3
"""Result figures of the TVT manuscript (Figs. 2-5), from out/mc_*.npz, out/mc_summary.json,
out/ns3_runs.csv and out/ns3_age.json. Drawn at final size (3.5 in column, 7.16 in page)."""
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
    if label:
        ax.text(0.02, TE * 1e9 * 1.15, r"$T_{\mathrm{e}}$", transform=ax.get_yaxis_transform(), color=FS.RED,
                fontsize=7, va="bottom")


def fig_validation(shell="starlink-480"):
    d = np.load(os.path.join(OUT, f"mc_{shell}.npz"))
    taus = d["taus"]
    h = float(np.min(d["meta"][:, 0]))
    ages = json.load(open(os.path.join(OUT, "ns3_age.json")))
    bins = np.array(ages["bins"])
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 2.35))
    # (a) UE propagators
    ax = axs[0]
    pts = [2, 5, 10, 20, 30, 60, 120, 180, 240]
    tt = np.linspace(1, 240, 400)
    for m in ["linear", "gravity-ecef", "kepler", "kepler-j2"]:
        c, mk, ls = FS.METHOD[m]
        if m != "kepler-j2":
            a = T.residual_acc(h, m, lat=np.pi / 2 if m == "kepler" else 0.0)
            ax.plot(tt, (a * tt ** 2 / T.C + 2 * (0.65 + 0.03 * tt) / T.C) * 1e9, color=c, ls=ls, lw=1.0)
        e = np.abs(d[f"eph_{m}"])
        ax.plot(pts, [p99(e, int(p)) * 1e9 for p in pts], ls="none", marker=mk, color=c, mfc=c, ms=3.6,
                label=LAB[m])
        runs = ages["ages"].get(shell, {}).get(f"S-{m}", [])
        if runs:
            A = np.array(runs, float)
            y = [np.nanpercentile(A[:, i], 99) if np.isfinite(A[:, i]).sum() >= 5 else np.nan
                 for i in range(A.shape[1])]
            ax.plot(bins[1:], np.array(y) * 1.0, ls="none", marker=mk, color=c, mfc="white", ms=4.6, mew=0.8)
    te_line(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 260)
    ax.set_ylim(1, 3e6)
    ax.set_xlabel(r"Information age $\tau$ (s)")
    ax.set_ylabel(r"UE-side error $|e_{\mathrm{s}}|$ (ns)")
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    ax.text(0.98, 0.04, "lines: closed-form bound\nfilled: MC, 99th pct.\nhollow: ns-3, 99th pct.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_title("(a) UE propagators", fontsize=8)
    # (b) common delay, gateway reference point
    ax = axs[1]
    m3 = T.ta3_max(h, np.radians(25))
    tt = np.linspace(1, 30, 200)
    c, mk, ls = FS.METHOD["taylor"]
    ax.plot(tt, m3 * tt ** 3 / 6 * 1e9, color=c, ls=ls)
    e = np.abs(d["com_taylor"])
    pts = [2, 5, 10, 15, 20, 30]
    ax.plot(pts, [p99(e, p) * 1e9 for p in pts], ls="none", marker=mk, color=c, mfc=c, label="Taylor")
    c, mk, ls = FS.METHOD["econ"]
    ax.plot(tt, m3 * tt ** 3 / 192 * 1e9, color=c, ls=ls)
    S = json.load(open(os.path.join(OUT, "mc_summary.json")))[shell]["win"]
    Vs = [5, 10, 15, 20, 30]
    ax.plot(Vs, [S[str(v)]["fit"]["p99_max_ns"] for v in Vs], ls="none", marker=mk, color=c, mfc=c,
            label="Economized")
    c, mk, ls = FS.METHOD["window"]
    ax.plot(Vs, [S[str(v)]["win"]["p99_max_ns"] for v in Vs], ls="none", marker=mk, color=c, mfc=c,
            label="Window fit")
    te_line(ax)
    ax.set_yscale("log")
    ax.set_xlim(0, 31)
    ax.set_ylim(1, 1e5)
    ax.set_xlabel(r"Age $\tau$ (Taylor) or window $V$ (fits) (s)")
    ax.set_ylabel(r"Common-delay error $|e_{\mathrm{c}}|$ (ns)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.text(0.98, 0.04, "lines: closed-form bound\n(Taylor; economized, cubic part)\nmarkers: MC, 99th pct.", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_title("(b) Common delay, gateway reference", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig2_validation.pdf")


def fig_validity():
    S = json.load(open(os.path.join(OUT, "mc_summary.json")))
    hh = np.linspace(330e3, 1300e3, 120)
    m3 = np.array([T.ta3_max(h, np.radians(25)) for h in hh])
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for key, kap in (("taylor", 1), ("econ", 32)):
        c, mk, ls = FS.METHOD[key]
        ax.plot(hh / 1e3, (6 * kap * TE / m3) ** (1 / 3), color=c, ls=ls, label=f"{LAB[key]}, bound")
    for sh in SHELLS:
        r = S[sh]
        x = r["h_km"]
        c, mk, _ = FS.METHOD["taylor"]
        ax.plot(x, r["taylor"]["tau99"], marker=mk, color=c, ls="none", mfc=c)
        for key, k2 in (("econ", "fit"), ("window", "win")):
            c, mk, _ = FS.METHOD[key]
            ok = [int(v) for v, w in r["win"].items() if w[k2]["pass"] >= 0.99]
            ax.plot(x, max(ok) if ok else np.nan, marker=mk, color=c, ls="none", mfc=c,
                    label=f"{LAB[key]}, MC" if sh == SHELLS[0] else None)
    ax.plot([], [], marker=FS.METHOD["taylor"][1], color=FS.METHOD["taylor"][0], ls="none", label="Taylor, MC")
    ax.set_yscale("log")
    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Validity of the common term (s)")
    ax.set_ylim(3, 120)
    ax.legend(loc="upper left", fontsize=6.5, ncol=1)
    fig.tight_layout()
    save(fig, "fig3_validity.pdf")


def load_runs():
    R = list(csv.DictReader(open(os.path.join(OUT, "ns3_runs.csv"))))
    return [r for r in R if not r.get("error")]


def mean_ci(x):
    x = np.asarray(x, float)
    from scipy import stats
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
    return m, h


SCHEMES = [("B0-ran4", "B0", "B0 (RAN4)"), ("B1-fix1", "B1", "B1, 1 s"), ("B1-fix5", "B1", "B1, 5 s"),
           ("B2-ekf", "B2", "B2"), ("L-slot", "L", "L (slot/drift)"), ("P-gw-taylor", "taylor", "P, Taylor"),
           ("P-gw-econ", "econ", "P, econ."), ("P-gw-window", "window", "P, window"),
           ("P-sat", "sat", "P, sat. ref.")]


def fig_baselines():
    R = load_runs()
    fig, axs = plt.subplots(1, 2, figsize=(PAGE, 2.3))
    w = 0.09
    for k, (cfg, sty, lab) in enumerate(SCHEMES):
        c, mk, ls = FS.METHOD[sty]
        hatch = "///" if cfg in ("B1-fix5",) else None
        for j, sh in enumerate(SHELLS):
            rows = [r for r in R if r["config"] == cfg and r["shell"] == sh]
            if not rows:
                continue
            m, h = mean_ci([float(r["reads_per_min"]) for r in rows])
            axs[0].bar(j + (k - 4) * w, m, w, yerr=h, color=c, edgecolor="white", lw=0.3, hatch=hatch,
                       error_kw=dict(lw=0.5, capsize=1), label=lab if j == 0 else None)
            v = 100 * np.sum([float(r["viol_frac"]) * float(r["samples"]) for r in rows]) / \
                np.sum([float(r["samples"]) for r in rows])
            if v > 0:
                axs[1].bar(j + (k - 4) * w, v, w, color=c, edgecolor="white", lw=0.3, hatch=hatch)
    for ax in axs:
        ax.set_xticks(range(len(SHELLS)))
        ax.set_xticklabels([SHORT[s] for s in SHELLS])
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("SIB19 acquisitions per minute")
    axs[1].set_ylabel(r"Time with $|e|>T_{\mathrm{e}}$ (%)")
    axs[1].set_ylim(0.01, 100)
    axs[0].set_title("(a) Acquisition rate", fontsize=8)
    axs[1].set_title("(b) Violation share", fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=9, fontsize=6.5, bbox_to_anchor=(0.5, 1.07),
               handlelength=1.2, columnspacing=0.7)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_baselines.pdf")


def fig_vehicles():
    R = load_runs()
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    veh = [("car", "120 km/h"), ("hst", "500 km/h"), ("aircraft", "900 km/h")]
    for j, (v, lab) in enumerate(veh):
        for k, (suffix, fl, col) in enumerate((("", "1 s", FS.ORANGE), ("-fast", "0.2 s", FS.BLUE))):
            cfg = f"V-{v}{suffix}"
            rows = [r for r in R if r["config"] == cfg]
            if not rows:
                continue
            vv = 100 * np.sum([float(r["viol_frac"]) * float(r["samples"]) for r in rows]) / \
                np.sum([float(r["samples"]) for r in rows])
            x = j + (k - 0.5) * 0.36
            ax.bar(x, vv if vv > 0 else 0, 0.34, color=col, edgecolor="white",
                   label=f"GNSS fix every {fl}" if (j, k) in ((0, 0), (1, 1)) else None)
            if vv == 0:
                ax.text(x, 0.013, "0", ha="center", va="bottom", fontsize=7, color=col)
    ax.set_xticks(range(len(veh)))
    ax.set_xticklabels([l for _, l in veh])
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.set_ylabel(r"Time with $|e|>T_{\mathrm{e}}$ (%)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig5_vehicles.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["validation", "validity", "baselines", "vehicles"]
    for w in which:
        globals()[f"fig_{w}"]()
