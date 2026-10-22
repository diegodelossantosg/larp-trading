import pandas as pd, numpy as np
from trader.lab import Lab, timing, stats, TIMING
L=Lab()
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',120)
for base,lev in [("SPY","SSO"),("QQQ","QLD"),("IWM","UWM")]:
    b=L.series(base).pct_change().fillna(0)
    rows=[]
    for rule in TIMING:
        r=timing(L,base,lev,rule); s=stats(r,b)
        rows.append({"rule":rule,**{k:s[k] for k in ("yearly_pct","worst_drop_pct","years_beating")},"by_year_vs_1x":s["by_year_excess_pct"]})
    print(f"\n=== {base}/{lev}"); print(pd.DataFrame(rows).to_string())
