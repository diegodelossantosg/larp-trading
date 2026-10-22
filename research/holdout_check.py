"""Final one-time check on the held-out companies (see trader/research.py). Run: PYTHONPATH=. python research/holdout_check.py"""
import pandas as pd
from trader.research import load, evaluate
pd.set_option('display.width',250)
df=load(); N=df.kind=='news'; E=df.kind=='earnings'
mk=df.spy_above_sma200.fillna(0)==1
rules={
 "NEW: news drop 4-8% + market ok (1h after, 10d)": (N&(df.move_1h<=-0.04)&(df.move_1h>-0.08)&mk, "hour_after", 10),
 "NEW, bought at close instead": (N&(df.day_move<=-0.04)&(df.day_move>-0.08)&mk, "close", 10),
 "OLD: news drop <=-2% (1h after, 10d)": (N&(df.move_1h<=-0.02), "hour_after", 10),
 "OLD: earnings beat & +2% (close, 20d)": (E&(df.beat==1)&(df.day_move>=0.02), "close", 20),
}
for name,(m,e,h) in rules.items():
    r=evaluate(df,m,e,h,True,sets=("holdout","dev","big"))
    print(f"\n{name}")
    for s in ("holdout","dev","big"):
        print(f"  {s:8s}", {k:r[s].get(k) for k in ("trades","avg_pct","win","random_pct","edge_pct","t","years_edge","edge_by_year")})
