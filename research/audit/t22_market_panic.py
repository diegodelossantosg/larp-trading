import pandas as pd, numpy as np
from trader.lab import Lab
L=Lab(); d=L.dates
o=L.open.astype("float64")[:,L.col["SPY"]]; c=L.close.astype("float64")[:,L.col["SPY"]]
sso_o=L.open.astype("float64")[:,L.col["SSO"]]; sso_c=L.close.astype("float64")[:,L.col["SSO"]]
n=len(d); yrs=np.array([x.year for x in d])
r1=np.r_[np.nan,c[1:]/c[:-1]-1]; hi=pd.Series(c).rolling(252,min_periods=100).max().to_numpy(); dd=c/hi-1
COST=0.002   # switching SPY->SSO and back, both ways
def events(mask,H):
    out=[]; last=-999
    for t in np.flatnonzero(mask):
        if t-last<H or t+H>=n: continue
        out.append(t); last=t
    return out
def run(label,mask):
    for H in (5,10,20):
        ts=events(mask,H); 
        if not ts: continue
        ex=np.array([ (sso_c[t+H]/sso_o[t+1]-1)-(c[t+H]/o[t+1]-1)-COST for t in ts]); y=yrs[ts]
        A,B=ex[y<2021],ex[y>=2021]
        base=np.nanmean([c[t+H]/o[t+1]-1 for t in range(n-H-1)])   # any-day SPY H-day return
        spy=np.array([c[t+H]/o[t+1]-1 for t in ts])
        print(f"{label:34} H{H:2d}d n={len(ts):3d} | SPY after event {spy.mean()*100:+.2f}% (any day {base*100:+.2f}%) | extra from SSO swap: design {A.mean()*100 if len(A) else float('nan'):+.2f}% (n{len(A)}) valid {B.mean()*100 if len(B) else float('nan'):+.2f}% (n{len(B)}) wins {(ex>0).mean()*100:.0f}%")
for th in (0.02,0.03,0.04): run(f"SPY day <= -{th:.0%}",r1<=-th)
for th in (0.08,0.10,0.15): run(f"SPY {th:.0%}+ below 52w high",dd<=-th)
