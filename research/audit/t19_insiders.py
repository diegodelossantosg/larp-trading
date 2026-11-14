import pandas as pd, numpy as np
from trader.lab import Lab
L=Lab(); dates=L.dates; c=L.close.astype("float64"); o=L.open.astype("float64"); v=L.volume.astype("float64")
spy=L.col["SPY"]; yrs=np.array([d.year for d in dates]); n_days=len(dates)
dvol=pd.DataFrame(c*v).rolling(20).mean().shift(1).to_numpy()
member=np.zeros_like(c,bool)
for i,y in enumerate(L.years): member[yrs==y]=L.member[i][None,:]
ins=pd.read_csv("research/insider_buys.csv.gz"); ins["filed"]=pd.to_datetime(ins.filed)
ins=ins[ins.symbol.isin(L.col)]
print("insider purchases:",len(ins),"| relationships:",ins.relationship.value_counts().head(5).to_dict())
dpos=pd.Series(np.arange(n_days),index=dates)
def entry_index(dt):  # first trading day strictly after the filing date
    k=dates.searchsorted(dt,side="right"); return k if k<n_days else None
COST=0.001; cut=2021
def run(sel,label,holds=(20,60)):
    # one signal per (symbol, filing day), aggregated
    g=sel.groupby(["symbol","filed"]).agg(value=("value","sum"),owners=("owner","nunique")).reset_index()
    for H in holds:
        rows=[]
        for sym,f,val,own in g.itertuples(index=False):
            e=entry_index(f)
            if e is None or e+H>=n_days: continue
            j=L.col[sym]
            if not (member[e,j] and c[e,j]>=5 and dvol[e,j]>=20e6): continue
            ent=o[e,j]
            if not np.isfinite(ent) or not np.isfinite(c[e+H,j]): continue
            r=c[e+H,j]/ent-1; s=c[e+H,spy]/o[e,spy]-1
            rows.append((dates[e].year,r-s-COST,r-COST))
        if not rows: print(label,H,"no trades"); continue
        d=pd.DataFrame(rows,columns=["y","ex","plain"])
        a,b=d[d.y<cut],d[d.y>=cut]
        t=lambda x: x.ex.mean()/(x.ex.std()/len(x)**.5) if len(x)>2 else 0
        ys=[g_.ex.mean() for _,g_ in b.groupby("y") if len(g_)>=15]
        print(f"{label:46} hold {H:2d}d | design {a.ex.mean()*100:+.2f}% n{len(a):4d} | valid {b.ex.mean()*100:+.2f}% n{len(b):4d} t{t(b):+.1f} yrs+ {sum(1 for x in ys if x>0)}/{len(ys)} win {(b.ex>0).mean()*100:.0f}%")
off=ins[ins.relationship.str.contains("Officer",case=False,na=False)]
ceo=ins[ins.title.fillna("").str.contains("CEO|Chief Executive|President|CFO|Chief Financial",case=False)]
run(ins,"any insider purchase")
run(off,"officers only")
run(ceo,"CEO/CFO/President only")
for minv in (100_000,500_000):
    run(ins[ins.value>=minv],f"any insider, purchase >= ${minv:,}")
    run(ceo[ceo.value>=minv],f"CEO/CFO, purchase >= ${minv:,}")
# clusters: 3+ distinct insiders buying within 30 days; the signal is the filing day that completes the cluster
sig=[]
for sym,gr in ins.sort_values("filed").groupby("symbol"):
    last=None
    for f in sorted(gr.filed.unique()):
        w=gr[(gr.filed<=f)&(gr.filed>f-pd.Timedelta(days=30))]
        if w.owner.nunique()>=3 and (last is None or (f-last).days>60):
            sig.append(gr[gr.filed==f].iloc[:1]); last=f
cl=pd.concat(sig) if sig else ins.iloc[:0]
run(cl,"cluster: 3+ insiders within 30 days")
