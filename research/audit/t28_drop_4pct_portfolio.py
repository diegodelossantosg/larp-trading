import numpy as np, pandas as pd
from trader.lab import Lab
L=Lab(); dates=L.dates; didx={str(d.date()):i for i,d in enumerate(dates)}
COST,CAP=0.001,0.30
def load(name):
    ev=pd.read_csv(f"research/{name}.csv.gz",low_memory=False); ev["d"]=pd.to_datetime(ev.day)
    n=ev[ev.kind=="news"].copy(); n["i"]=n.day.map(didx); return n.dropna(subset=["i"])
def run(news,lo,hi,drop,atr=3,dv=20,per_day=2,size=0.10,h=10,extra=None):
    n=news[(news.d>=lo)&(news.d<hi)&(news.move_1h<=-drop)&(news.dollar_vol_m>=dv)&(news.atr_pct>=atr)]
    if extra is not None: n=n[extra(n)]
    col=f"hour_after_r{h}"; open_pos=[]; net=[]
    for i,g in n.sort_values(["i","move_1h"]).groupby("i",sort=True):
        i=int(i); open_pos=[p for p in open_pos if p[0]>i]
        exp=sum(p[1] for p in open_pos); held={p[2] for p in open_pos}; took=0
        for sym,r in zip(g.symbol.to_numpy(),g[col].to_numpy()):
            if took>=per_day or exp+size>CAP+1e-9: break
            if sym in held or np.isnan(r) or i+h>=len(dates): continue
            open_pos.append((i+h,size,sym)); held.add(sym); exp+=size; took+=1; net.append(r-COST)
    yrs=max((hi-lo).days/365.25,1e-9); net=np.array(net); k=len(net)
    return k/yrs, (size*net.sum()/yrs*100 if k else 0.0), (net.mean()*100 if k else np.nan)
files={"long 2018-26":"events_long","wide 2023-26":"events_wide","mid 2023-26":"events_mid","early 2016-18":"events_early"}
print(f"{'archivo':14s} {'regla':14s} {'compras/año':>11s} {'ganancia extra/año sobre SPY':>30s} {'por compra':>10s}")
for lab,f in files.items():
    n=load(f); lo,hi=n.d.min(),n.d.max()+pd.Timedelta(days=1)
    for name,kw in (("actual (2%)",dict(drop=.02)),("4% o más",dict(drop=.04)),("3% o más",dict(drop=.03)),("2% + ATR>=6",dict(drop=.02,atr=6)),("4% 3 compras/día",dict(drop=.04,per_day=3))):
        k,a,m=run(n,lo,hi,**kw); print(f"{lab:14s} {name:16s} {k:8.1f} {a:22.2f}% {m:12.2f}%")
