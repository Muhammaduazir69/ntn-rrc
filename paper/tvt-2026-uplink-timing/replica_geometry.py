"""Round 3, m7: Python replica of the ns-3 pass geometry (40 passes, 470 km, Kepler UE, no GNSS error), ages 60 to 120 s.
Shows that the gap between ns-3 and the Monte Carlo at those ages is geometry. Writes out/replica.json."""
import sys,numpy as np
sys.dont_write_bytecode=True
H=__import__("os").path.dirname(__import__("os").path.abspath(__file__));sys.path.insert(0,H)
import run_ns3_campaign as R, ta_core as T, mc_ta
from sgp4.api import Satrec
rng=np.random.default_rng(2026)
P={sh:R.passes_for(sh,100,rng) for sh in R.SHELLS}
RE=6378137.0
errs=[];ages=[]
for p in P['starlink-480'][:40]:
    s=Satrec.twoline2rv(p['l1'],p['l2'])
    jd,fr=s.jdsatepoch,s.jdsatepochF
    start=p['start']
    tca=start+450
    s0,v0=T.sgp4_ecef(s,jd,fr,np.array([tca]));s0=s0[0];v0=v0[0]
    nad=s0/np.linalg.norm(s0);cr=np.cross(nad,v0);cr/=np.linalg.norm(cr)
    psi=p['ue_x']*1e3/RE
    ue=RE*(np.cos(psi)*nad+np.sin(psi)*cr)
    ts=start+0.05+np.arange(0,9000)*0.1
    Pt=T.sgp4_ecef(s,jd,fr,ts)[0]
    up=ue/np.linalg.norm(ue);los=Pt-ue;el=np.degrees(np.arcsin((los@up)/np.linalg.norm(los,axis=1)))
    idx=np.where(el>=25)[0]
    if len(idx)==0: continue
    t0=ts[idx[0]];k=0
    # first content: latest epoch on the 5 s grid, epoch=start+5k+lead-ish
    i=0
    tt=ts[idx]
    while i<len(tt):
        ep=start+5*np.floor((tt[i]-start+0.64)/5)
        ps,vs=T.sgp4_ecef(s,jd,fr,np.array([ep]));ps=ps[0];vs=vs[0]
        pq=T.quantize(ps,1.3,1e12);vq=T.quantize(vs,0.06,1e12)
        j=i
        sel=tt[(tt>=tt[i])&(tt-ep<240-0.32)]
        a=sel-ep
        grid=np.arange(0,241.0)
        Pp=T.propagate_pv(pq,vq,grid,'kepler',step=1.0)
        PP=np.stack([np.interp(a,grid,Pp[:,c]) for c in range(3)],1)
        Ps=T.sgp4_ecef(s,jd,fr,sel)[0]
        e=2*(np.linalg.norm(PP-ue,axis=1)-np.linalg.norm(Ps-ue,axis=1))/T.C
        errs.append(np.abs(e));ages.append(a)
        i+=len(sel)
e=np.concatenate(errs)*1e9;a=np.concatenate(ages)
m=(a>=60)&(a<120)
print('python replica, ns-3 geometry, no GNSS, 60-120 s: n',m.sum(),'p99',np.percentile(e[m],99),'median',np.median(e[m]))
import json;json.dump({'p99':float(np.percentile(e[m],99)),'median':float(np.median(e[m]))},open(H+'/out/replica.json','w'))
Z=np.load(H+'/out_r4/ns3_raw.npz');ag=Z['starlink-480|S-kepler|age'];er=Z['starlink-480|S-kepler|err']
mm=(ag>=60)&(ag<120);print('ns-3 raw same band: n',mm.sum(),'p99',np.percentile(er[mm],99),'median',np.median(er[mm]))
d=np.load(H+'/out/mc_starlink-480.npz');t=d['taus'];c=np.abs(d['eph_kepler'])[:,(t>=60)&(t<120)]*1e9
print('MC band p99',np.nanpercentile(c,99),'median',np.nanmedian(c))
