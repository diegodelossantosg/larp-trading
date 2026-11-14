import pandas as pd, numpy as np
from trader.lab import Lab
L=Lab(); dates=L.dates; didx={str(d.date()):i for i,d in enumerate(dates)}
COST=0.001; H=10
e=pd.concat([pd.read_csv(f"research/{f}.csv.gz",low_memory=False) for f in ("events_long","events_wide")]); e["d"]=pd.to_datetime(e.day)
n0=e[(e.kind=="news")&(e.atr_pct>=3)&(e.dollar_vol_m>=20)&(e.vol_ratio<=2)&(e.move_1h<=-0.02)].drop_duplicates(["symbol","day"]).dropna(subset=["hour_after_r10"]).copy()
n0["i"]=n0.day.map(didx); n0=n0.dropna(subset=["i"])
def sim(size_fn, cap, per_day=2):
    pos=[]; by={}
    for i,g in n0.sort_values(["i","move_1h"]).groupby("i"):
        i=int(i); pos=[p for p in pos if p[0]>i]; held={p[1] for p in pos}; took=0
        for sym,r,mv in zip(g.symbol,g.hour_after_r10,g.move_1h):
            sz=size_fn(mv); exposure=sum(p[2] for p in pos)
            if took>=per_day or exposure+sz>cap+1e-9: break
            if sym in held or i+H>=len(dates): continue
            pos.append((i+H,sym,sz)); held.add(sym); took+=1
            y=dates[i+H].year; by[y]=by.get(y,0)+sz*(r-COST)
    return by
def show(name,by):
    ys=[y for y in range(2018,2027) if y in by]; v=np.array([by[y]*100 for y in ys])
    print(f"{name:42} mean/yr {v.mean():+5.1f}%  worst {v.min():+6.1f}%  best {v.max():+5.1f}%  years>0 {int((v>0).sum())}/{len(v)}  mean/std {v.mean()/v.std():.2f}")
show("current: 10% each, cap 30%",sim(lambda m:0.10,0.30))
show("15% each, cap 45%",sim(lambda m:0.15,0.45))
show("20% each, cap 60%",sim(lambda m:0.20,0.60))
show("5% each, cap 15%",sim(lambda m:0.05,0.15))
show("graded: 15% if drop>=4% else 10%, cap 40%",sim(lambda m:0.15 if m<=-0.04 else 0.10,0.40))
show("10% each, 3 per day, cap 50%",sim(lambda m:0.10,0.50,per_day=3))
show("only drops>=4%: 15% each, cap 45%",sim(lambda m:0.15 if m<=-0.04 else 0.0,0.45))
