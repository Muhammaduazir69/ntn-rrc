#!/usr/bin/env python3
"""Static window-fit lookup from the Monte Carlo (Section VI): for each limit L, the largest enumerated V
per shell for which no Monte Carlo case exceeds L over the window. Writes out/lookup.json. The lookup is
fixed here, before the ns-3 campaign, and tested in ns-3 on passes it never saw."""
import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
SH=["starlink-vleo","starlink-480","starlink-550","oneweb-1200"]
TC=1/(480e3*4096); TE=22*64*TC
out={"worst_ns":{}, "n":{}}
for sh in SH:
    d=np.load(os.path.join(HERE,"out",f"mc_{sh}.npz"))
    w={}; n={}
    for V in range(5,61,5):
        e=np.abs(d[f"com_win{V}"])[:,:V+1]
        full=np.isfinite(e).all(axis=1)
        m=e[full].max(axis=1)
        w[V]=float(m.max()*1e9); n[V]=int(full.sum())
    out["worst_ns"][sh]=w; out["n"][sh]=n
    print(sh,{v:round(x) for v,x in w.items()},n[60])
    tt=np.abs(d["com_taylor"])
    out.setdefault("taylor_worst_ns",{})[sh]={V:float(np.nanmax(tt[np.isfinite(tt[:,:V+1]).all(axis=1),:V+1])*1e9) for V in (5,10,15,20)}
    print('  taylor',out["taylor_worst_ns"][sh])
json.dump(out,open(os.path.join(HERE,"out","lookup.json"),"w"),indent=1)
