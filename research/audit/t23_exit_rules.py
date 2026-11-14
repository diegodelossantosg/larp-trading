import pandas as pd, numpy as np
COST=0.001
e=pd.concat([pd.read_csv(f"research/{f}.csv.gz",low_memory=False) for f in ("events_long","events_wide")]); e["d"]=pd.to_datetime(e.day)
n=e[(e.kind=="news")&(e.atr_pct>=3)&(e.dollar_vol_m>=20)&(e.vol_ratio<=2)&(e.move_1h<=-0.02)].drop_duplicates(["symbol","day"])
n=n.dropna(subset=["hour_after_r1","hour_after_r5","hour_after_r10","hour_after_r20"]).copy()
r1,r5,r10,r20=(n[f"hour_after_r{h}"].to_numpy() for h in (1,5,10,20)); cut=pd.Timestamp("2023-01-01")
pol={"A hold 10 days (current)":r10}
for X in (0.03,0.05,0.08): pol[f"B take profit at day 5 if +{X:.0%}, else hold to 10"]=np.where(r5>=X,r5,r10)
for Y in (0.05,0.08,0.10): pol[f"C cut at day 5 if {-Y:.0%} or worse, else hold to 10"]=np.where(r5<=-Y,r5,r10)
for Z in (0.03,0.05): pol[f"D cut at day 1 if {-Z:.0%} or worse, else hold to 10"]=np.where(r1<=-Z,r1,r10)
pol["E at day 10 keep to day 20 only if ahead"]=np.where(r10>0,r20,r10)
pol["F at day 10 keep to day 20 only if behind"]=np.where(r10<0,r20,r10)
pol["G hold 5 days"]=r5; pol["H hold 20 days"]=r20
d=(n.d<cut).to_numpy(); v=~d
base=r10-COST
print(f"{'policy':58} {'design avg':>10} {'diff vs A':>9} | {'valid avg':>9} {'diff vs A':>9} {'t(diff)':>7} n={d.sum()}/{v.sum()}")
for k,x in pol.items():
    y=x-COST; df_=y-base
    t=df_[v].mean()/(df_[v].std()/v.sum()**.5) if df_[v].std()>0 else 0
    print(f"{k:58} {y[d].mean()*100:+10.2f} {df_[d].mean()*100:+9.2f} | {y[v].mean()*100:+9.2f} {df_[v].mean()*100:+9.2f} {t:+7.1f}")
