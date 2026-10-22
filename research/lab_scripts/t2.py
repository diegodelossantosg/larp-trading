import pandas as pd, numpy as np
from trader.lab import Lab, timing, stats
L=Lab()
pd.set_option('display.width',300)
for base,lev in [("SPY","SSO"),("QQQ","QLD"),("IWM","UWM")]:
    b=L.series(base).pct_change().fillna(0)
    always=timing(L,base,lev,"always 2x"); trend=timing(L,base,lev,"2x when above 200d avg")
    rows=[]
    for w in (0.3,0.4,0.5,0.6):
        for nm,core in (("SSO always",always),("SSO trend rule",trend)):
            r=w*core+(1-w)*b   # daily-rebalanced blend
            s=stats(r,b); rows.append({"core %":int(w*100),"core":nm,**{k:s[k] for k in ("yearly_pct","worst_drop_pct","years_beating")}})
    print(f"\n=== {base}: X% in the 2x fund, rest in {base}"); print(pd.DataFrame(rows).to_string())
