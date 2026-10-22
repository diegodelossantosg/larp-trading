import pandas as pd, numpy as np
import trader.lab as lab
from trader.lab import Lab, pick_backtest, equal_weight, stats
L=Lab()
spy=pd.Series(np.nan_to_num(L.ret[:,L.col["SPY"]]),index=L.dates)
rows=[]
for cost in (0.001,0.002):
    lab.COST=cost
    for n in (10,20):
        r=pick_backtest(L,"mom_12_1_risk_adj","C",n=n); s=stats(r,spy)
        rows.append({"group":"C","cost_per_side":cost,"n":n,**{k:s[k] for k in ("yearly_pct","bench_yearly_pct","worst_drop_pct","years_beating","by_year_excess_pct")}})
lab.COST=0.001
ew=equal_weight(L,"C"); s=stats(ew,spy)
rows.append({"group":"C equal-weight","n":"all",**{k:s[k] for k in ("yearly_pct","worst_drop_pct","years_beating")}})
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',200)
print(pd.DataFrame(rows).to_string())
