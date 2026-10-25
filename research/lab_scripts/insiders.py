import sys, numpy as np, pandas as pd
from trader.lab import Lab
L=Lab(); COST=0.001
d=pd.read_csv("research/insider_buys.csv.gz")
d=d[d.relationship.fillna("").str.contains("Director|Officer")]
d=d[d.symbol.isin(L.col)]
d["filed"]=pd.to_datetime(d.filed); d=d.sort_values("filed")
dates=L.dates.values
spy=L.col["SPY"]; C=L.close.astype(float); O=L.open.astype(float)
def entry_idx(f):  # first trading day strictly after filing date
    return int(np.searchsorted(dates, np.datetime64(f), side="right"))
def ret(j,i,H):
    if i+H>=len(dates) or np.isnan(O[i,j]): return np.nan
    c=C[i:i+H+1,j]; k=np.where(~np.isnan(c))[0]
    out=c[k[-1]]                       # delisted: last price
    return out/O[i,j]-1-(C[i+H,spy]/O[i,spy]-1)-COST
def member(j,i):
    y=L.dates[i].year
    return y in L.years and L.member[L.years.index(y),j]
# signals
def clusters(df, min_people=2, min_value=1e5, window=30):
    out=[]; last={}
    for sym,g in df.groupby("symbol"):
        g=g.sort_values("filed"); rows=g[["filed","owner","value"]].values
        for k,(f,o,v) in enumerate(rows):
            w=g[(g.filed>f-pd.Timedelta(days=window))&(g.filed<=f)]
            if w.owner.nunique()>=min_people and w.value.sum()>=min_value:
                if sym in last and (f-last[sym]).days<90: continue
                last[sym]=f; out.append((sym,f))
    return out
def ceo(df,min_value=1e5):
    out=[]; last={}
    t=df[df.title.fillna("").str.contains(r"CEO|Chief Executive|CFO|Chief Financial",case=False)]
    for sym,g in t.groupby("symbol"):
        for f,v in g.groupby("filed").value.sum().items():
            if v>=min_value and not (sym in last and (f-last[sym]).days<90):
                last[sym]=f; out.append((sym,f))
    return out
rng=np.random.default_rng(3)
def evaluate(sig,H,group):
    rs,ys=[],[]
    for sym,f in sig:
        j=L.col[sym]
        if L.groups[j]!=group: continue
        i=entry_idx(f)
        if i>=len(dates) or not member(j,i): continue
        r=ret(j,i,H)
        if not np.isnan(r): rs.append(r); ys.append(L.dates[i].year)
    # random baseline: member stocks of group, random days
    base=[]
    while len(base)<3000:
        i=int(rng.integers(260,len(dates)-H-1)); y=L.dates[i].year
        cand=np.flatnonzero(L.member[L.years.index(y)]&(L.groups==group))
        j=int(rng.choice(cand)); r=ret(j,i,H)
        if not np.isnan(r): base.append(r)
    rs=np.array(rs); b=np.mean(base)
    by=pd.Series(rs).groupby(ys).mean()-b if len(rs) else pd.Series(dtype=float)
    cnt=pd.Series(ys).value_counts()
    by=by[[cnt[y]>=5 for y in by.index]]
    return {"n":len(rs),"avg%":round(rs.mean()*100,2) if len(rs) else None,"random%":round(b*100,2),
            "edge%":round((rs.mean()-b)*100,2) if len(rs) else None,"yrs":f"{int((by>0).sum())}/{len(by)}",
            "t":round(rs.mean()/(rs.std()/len(rs)**.5),1) if len(rs)>2 else None}
groups=sys.argv[1].split(",")
S={"cluster 2+ insiders >=$100k":clusters(d),"CEO/CFO buy >=$100k":ceo(d)}
for name,sig in S.items():
    for H in (20,60):
        print(name,f"{H}d",{g:evaluate(sig,H,g) for g in groups})
