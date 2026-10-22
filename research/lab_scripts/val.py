import pandas as pd, numpy as np, sys
from trader.lab import Lab, pick_backtest, equal_weight, stats
L=Lab(); g=sys.argv[1]
spy=pd.Series(np.nan_to_num(L.ret[:,L.col["SPY"]]),index=L.dates)
ew=equal_weight(L,g)
rows=[{"rule":"SPY",**{k:stats(spy,spy)[k] for k in ("yearly_pct","worst_drop_pct")}},
      {"rule":f"equal-weight {g}",**{k:stats(ew,spy)[k] for k in ("yearly_pct","worst_drop_pct","years_beating")}}]
for sc,n in [("mom_12_1_risk_adj",10),("mom_12_1_risk_adj",20),("mom_and_low_vol",10)]:
    r=pick_backtest(L,sc,g,n=n); s=stats(r,spy)
    rows.append({"rule":f"{sc} n={n}",**{k:s[k] for k in ("yearly_pct","worst_drop_pct","years_beating","by_year_excess_pct")}})
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',200)
print(pd.DataFrame(rows).to_string())
