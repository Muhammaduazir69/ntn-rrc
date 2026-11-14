#!/usr/bin/env python3
import sys; sys.exit("make_docs.py is out of date (old affiliation and AI wording); the package in final_submission/paper5_TVT is canonical.")
"""Fill the abstract, the headline numbers and the page count into cover_letter.md and submission_metadata.md
from macros_p5.tex and main.pdf, so that they cannot drift from the PDF."""
import os, re, subprocess
H = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(H, "..")
M = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(os.path.join(P, "macros_p5.tex")).read()))
pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", os.path.join(P, "main.pdf")], capture_output=True, text=True).stdout).group(1)
t = open(os.path.join(P, "sections", "abstract.tex")).read()
a = t[t.index("\\begin{abstract}") + 16:t.index("\\end{abstract}")]
a = a.replace("$\\mu$s", "us")
a = a.replace("$V=900$", "V = 900")
a = re.sub(r"\\(\w+)\\? ?", lambda m: M.get(m.group(1), m.group(0)) + (" " if m.group(0).endswith(" ") else ""), a)
a = a.replace("~", " ").replace("$J_2$", "J2").replace("\n", " ")
a = re.sub(r"\s+", " ", a).strip()
words = len(a.split())
assert words <= 250, words
letter = f"""Muhammad Uzair
Department of Computer Science, COMSATS University Islamabad
Islamabad 45550, Pakistan
muhammaduzairr69@gmail.com | ORCID 0009-0002-4104-2680

5 November 2026

The Editor-in-Chief
IEEE Transactions on Vehicular Technology

Dear Editor,

I submit the manuscript "Validity of Broadcast Assistance Information for Uplink Timing Pre-Compensation in LEO Non-Terrestrial Networks" for consideration as a regular paper in IEEE Transactions on Vehicular Technology.

A Release 17 UE in a non-terrestrial network times its uplink from system information block 19 and must stop transmitting when the information expires. The article shows that, with a ground reference point, most of the validity to be gained comes from what the network broadcasts: coefficients fitted over the validity window instead of expanded at the epoch, with no new signaling and no change at the UE, provided epochs are announced at most about {M['LeadS']} s ahead and the fit covers that lead. A closed-form error model against information age explains why, and a network-side sizing rule follows from it.

The evidence is an exact-geometry Monte Carlo on the October 2026 Starlink and OneWeb element sets and a kinematic ns-3 implementation over up to {M['NPasses']} SGP4-propagated passes across all orbital phases. A table of fixed validities per shell needs {M['GainFit']} times fewer SIB19 decodings with fitted coefficients than with Taylor ones. Sizing the validity at each epoch saves a further {M['GainRuleOne']} times over that table. With the reference point at the satellite, the simplest choice does best for a UE that propagates with J2: no expiry within a pass needs one decoding per pass, about {M['RuleOverFix']} times fewer than the per-epoch rule, peaking at {M['MaxSNineZeroZero']} ns against a {M['TePrimeNs']} ns limit with an exact orbit. These results held for orbit velocity errors up to the 0.03 m/s we tested, while some passes crossed the limit at 0.1 m/s, and at 5 m of GNSS error per horizontal axis the per-epoch rule loses its gain over the table. The article also states plainly its dependence on the UE's propagator, on the network's orbit knowledge and on fast vehicles projecting their position fix.

I believe the topic suits the journal's readership in vehicular and satellite communications: the results bear on handsets, IoT devices and vehicles served directly by LEO satellites, and the vehicle section gives the GNSS fix rate that trains and aircraft need.

The manuscript is original, has not been published, and is not under consideration elsewhere. Earlier work on the simulation toolkit used here appeared in the IEEE Open Journal of the Communications Society (DOI 10.1109/OJCOMS.2026.3745010); the present article does not overlap with it in results. The simulation code and scripts will be released upon acceptance. The use of AI is disclosed in the Acknowledgment.

Suggested topics for reviewers: NR non-terrestrial networks, uplink synchronization and timing advance, satellite orbit determination.

Sincerely,
Muhammad Uzair
"""
open(os.path.join(P, "cover_letter.md"), "w").write(letter)
mp = "/home/uzair/6g_ntn_ns3/papers/final_submission/paper5_TVT/submission_metadata.md"
s = open(mp).read()
s = re.sub(r"\*\*Manuscript length:\*\* \d+ pages", f"**Manuscript length:** {pages} pages", s)
s = re.sub(r"\*\*Abstract \(plain text, \d+ words\):\*\*\n.*?\n\n", f"**Abstract (plain text, {words} words):**\n{a}\n\n", s, flags=re.S)
s = re.sub(r"\(\d+ pages: met\)", f"({pages} pages: met)", s)
extra = int(pages) - 10
s = re.sub(r"\(\d+ pages: about US\$\d+\)", f"({pages} pages: about US${220 * extra})", s)
s = re.sub(r"compiles standalone to \d+ pages", f"compiles standalone to {pages} pages", s)
open(mp, "w").write(s)
print(words, "words;", pages, "pages")
