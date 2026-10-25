import re, pandas as pd, numpy as np
from trader.research import load, evaluate
pd.set_option('display.width',250)
df=load()
EPSR=re.compile(r"EPS \$?(-?\d+(?:\.\d+)?)\s+(?:Beats?|Miss(?:es)?|In-Line|Inline|Up|Down)\w*\s+(?:From |With )?\$?(-?\d+(?:\.\d+)?)\s+Estimate",re.I)
SALR=re.compile(r"(?:Sales|Revenue)\s+\$?(\d+(?:\.\d+)?)([BMK])\s+(?:Beats?|Miss(?:es)?|In-Line|Inline)\w*\s+\$?(\d+(?:\.\d+)?)([BMK])",re.I)
mult={"B":1e9,"M":1e6,"K":1e3}
def eps(h):
    m=EPSR.search(str(h)); 
    if not m: return np.nan
    a,e=float(m.group(1)),float(m.group(2)); return (a-e)/max(abs(e),0.10)
def sales(h):
    m=SALR.search(str(h))
    if not m: return np.nan
    a=float(m.group(1))*mult[m.group(2).upper()]; e=float(m.group(3))*mult[m.group(4).upper()]
    return a/e-1
E=df.kind=='earnings'
df.loc[E,'eps_surprise']=df.loc[E,'headline'].map(eps); df.loc[E,'sales_surprise']=df.loc[E,'headline'].map(sales)
print("parsed eps:",df.loc[E,'eps_surprise'].notna().mean().round(2)," sales:",df.loc[E,'sales_surprise'].notna().mean().round(2))
def show(name,m,entry="close",hold=20,sets=("big","dev")):
    r=evaluate(df,m,entry,hold,True,sets)
    print(f"{name:55s}",{s:(r[s].get('trades'),r[s].get('avg_pct'),r[s].get('edge_pct'),r[s].get('years_edge')) for s in sets})
S=df.eps_surprise; SS=df.sales_surprise
# pre-registered
R1=E&(S>=0.10)&(SS>0); R2=R1&(df.day_move>=0)
print("\n-- pre-registered (design groups) --")
for h in (10,20): show(f"R1 eps>=+10% & sales beat, {h}d",R1,hold=h); show(f"R2 R1 & day up, {h}d",R2,hold=h)
print("\n-- diagnosis by surprise size (design groups, close, 20d) --")
for lo,hi in [(-9,-0.05),(-0.05,0),(0,0.03),(0.03,0.10),(0.10,0.25),(0.25,9)]:
    show(f"eps surprise {lo:+.0%}..{hi:+.0%}",E&(S>=lo)&(S<hi))
