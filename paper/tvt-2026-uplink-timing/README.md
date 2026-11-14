# Code and data: validity of broadcast assistance information for uplink timing pre-compensation

Companion material for M. Uzair, "Validity of Broadcast Assistance Information for Uplink Timing Pre-Compensation in LEO Non-Terrestrial Networks," 2026.

The study measures how long the SIB19 ephemeris and common timing-advance parameters stay valid for uplink timing pre-compensation on Starlink, OneWeb and very-low-orbit shells, with Monte Carlo over passes and packet-level ns-3 runs of the ntn-rrc module.

| Path | Contents |
|---|---|
| `ns3_patch/` | Campaign example source of record and build script |
| `run_ns3_campaign.py` | Packet-level ns-3 campaign driver |
| `ta_core.py`, `mc_ta.py`, `run_mc.sh`, `lookup_mc.py`, `lookup_lead_mc.py` | Timing-advance error model and Monte Carlo over passes |
| `analyze_mc.py`, `regress_budget.py`, `time_rule.py`, `hifi_*.py`, `cheap_gnss.py` | Analysis, validity rules and high-fidelity checks |
| `data/` | Element sets and inputs |
| `out.tar.xz`, `out_r3.tar.xz`, `out_r4.tar.xz`, `out_r2/`, `out_r5/` | Result files of each campaign stage |
| `make_macros.py`, `make_figs.py` | Every number and figure, generated from the results |

Unpack with `tar -xJf out.tar.xz` (same for the other archives). Monte Carlo arrays larger than 45 MB are not stored; `run_mc.sh` regenerates them.

Contact: Muhammad Uzair, muhammaduzairr69@gmail.com, ORCID 0009-0002-4104-2680. License: as the repository (GPL-2.0-only for code).
