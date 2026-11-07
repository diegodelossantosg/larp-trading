"""T2: permutation test. How often would random days of the same kind of stocks (volatile, liquid) do as well as the News Hunter's picks?"""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
ev = pd.read_csv("research/events.csv.gz", low_memory=False)
ev = ev[ev["set"].isin(["dev", "holdout"])]
news, ctl = ev[ev.kind == "news"], ev[ev.kind == "control"]
rules = {"current rule (any stock, dv>=300)": lambda d: d.dollar_vol_m >= 300,
         "volatile only (ATR>=3%)": lambda d: (d.dollar_vol_m >= 300) & (d.atr_pct >= 3),
         "volatile + below 50d avg": lambda d: (d.dollar_vol_m >= 300) & (d.atr_pct >= 3) & (d.above_sma50 == 0)}
rows = []
for name, f in rules.items():
    for s in ("dev", "holdout", "both"):
        sets = ("dev", "holdout") if s == "both" else (s,)
        x = news[news["set"].isin(sets) & (news.move_1h <= -0.02)]; x = x[f(x)]["hour_after_r10"].dropna().to_numpy() - 0.001
        c = ctl[ctl["set"].isin(sets)]; c = c[f(c)]["h11_r10"].dropna().to_numpy() - 0.001
        n = len(x)
        null = np.array([rng.choice(c, n, replace=True).mean() for _ in range(20000)])
        rows.append({"rule": name, "set": s, "trades": n, "avg_net_pct": round(x.mean() * 100, 2), "random_days_avg_pct": round(c.mean() * 100, 2),
                     "p_value": round(float((null >= x.mean()).mean()), 4)})
d = pd.DataFrame(rows); d.to_csv("research/audit/out_t2.csv", index=False)
print(d.to_string(index=False))
print("\nWith ~34 rules/slices tried, a single p-value must be below ~0.0015 (Bonferroni 0.05/34) to count as clearly not luck.")
