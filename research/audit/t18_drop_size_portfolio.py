import pandas as pd, numpy as np
from trader.lab import Lab
L=Lab(); dates=L.dates; didx={str(d.date()):i for i,d in enumerate(dates)}
COST=0.001; SIZE=0.10; CAP=0.30; PER_DAY=2; H=10
e=pd.concat([pd.read_csv(f"research/{f}.csv.gz",low_memory=False).assign(src=f) for f in ("events_long","events_wide")]); e["d"]=pd.to_datetime(e.day)
ctl=e[(e.kind=="control")&(e.atr_pct>=3)&(e.dollar_vol_m>=20)].drop_duplicates(["symbol","day"])
n0=e[(e.kind=="news")&(e.atr_pct>=3)&(e.dollar_vol_m>=20)&(e.vol_ratio<=2)].drop_duplicates(["symbol","day"]).dropna(subset=["hour_after_r10"]).copy()
n0["i"]=n0.day.map(didx); n0=n0.dropna(subset=["i"])
def sim(n):
    pos=[]; by={}; cnt=0
    for i,g in n.sort_values(["i","move_1h"]).groupby("i"):
        i=int(i); pos=[p for p in pos if p[0]>i]; held={p[1] for p in pos}; took=0
        for sym,r in zip(g.symbol,g.hour_after_r10):
            if took>=PER_DAY or (len(pos)+1)*SIZE>CAP+1e-9: break
            if sym in held or i+H>=len(dates): continue
            pos.append((i+H,sym)); held.add(sym); took+=1; cnt+=1
            y=dates[i+H].year; by[y]=by.get(y,0)+SIZE*(r-COST)
    return by,cnt
print("portfolio vs S&P (10%/buy, 2/day, 30% cap, 10d hold), % of capital by exit year:")
for lo in (0.02,0.03,0.04,0.05):
    n=n0[n0.move_1h<=-lo]; by,cnt=sim(n)
    yrs=sorted(by); d=sum(by[y] for y in yrs if y<=2022)/5*100; v=sum(by[y] for y in yrs if y>=2023)/3.9*100
    print(f" drop>={lo:.0%}: trades {cnt:4d} | per-year %:",{y:round(by[y]*100,1) for y in yrs}," | design avg/yr %.1f%%  valid avg/yr %.1f%%"%(d,v))
n4=n0[n0.move_1h<=-0.04]; base=ctl["h11_r10"].mean()
for k,g in n4.groupby("src"): print(" 4%+ file",k,len(g),f"edge {(g.hour_after_r10-COST-base).mean()*100:+.2f}%")
for y,g in n4.groupby(n4.d.dt.year): print(" 4%+ year",y,len(g),f"edge {(g.hour_after_r10-COST-base).mean()*100:+.2f}%")
