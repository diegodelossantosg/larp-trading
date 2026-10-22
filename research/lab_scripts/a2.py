import pandas as pd, numpy as np, sys
from trader.lab import Lab, pick_backtest, equal_weight, stats
L=Lab(); g=sys.argv[1] if len(sys.argv)>1 else "A"
spy=pd.Series(np.nan_to_num(L.ret[:,L.col["SPY"]]),index=L.dates)
rows=[]
for sc in ("mom_12_1_risk_adj","mom_and_low_vol","mom_12_1"):
  for n in (10,20,30):
    for every in (1,3):
      for stop in (None,0.15,0.25):
        if sc!="mom_12_1_risk_adj" and (stop or every!=1): continue
        r=pick_backtest(L,sc,g,n=n,stop=stop,every=every); s=stats(r,spy)
        rows.append({"rule":sc,"n":n,"every_m":every,"stop":stop,**{k:s[k] for k in ("yearly_pct","worst_drop_pct","years_beating")}})
pd.set_option('display.width',300)
print(pd.DataFrame(rows).to_string())
