import pandas as pd, numpy as np, time
from trader.lab import Lab, pick_backtest, equal_weight, stats, SCORES
L=Lab()
spy=pd.Series(np.nan_to_num(L.ret[:,L.col["SPY"]]),index=L.dates)
ew=equal_weight(L,"A")
rows=[]
for nm,r in [("SPY",spy),("equal-weight A",ew)]:
    s=stats(r,spy); rows.append({"rule":nm,**{k:s[k] for k in ("yearly_pct","worst_drop_pct","years_beating")}})
for sc in SCORES:
    r=pick_backtest(L,sc,"A",n=10)
    s=stats(r,spy); s2=stats(r,ew)
    rows.append({"rule":sc,**{k:s[k] for k in ("yearly_pct","worst_drop_pct","years_beating")},"beats_EW_years":s2["years_beating"],"by_year_vs_spy":s["by_year_excess_pct"]})
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',200)
print(pd.DataFrame(rows).to_string())
