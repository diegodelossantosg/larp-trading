"""T12: (a) liquidity floor with the volatile-stock filter (do more, thinner companies now help?)  (b) how many years of live results are needed to confirm the edge."""
import numpy as np, pandas as pd
ev = pd.read_csv("research/events.csv.gz", low_memory=False)
ev = ev[ev["set"].isin(["dev", "holdout"])]
news, ctl = ev[ev.kind == "news"], ev[ev.kind == "control"]
rows = []
for dv in (30, 50, 100, 150, 300, 600, 1000):
    for s in ("dev", "holdout"):
        d = news[(news["set"] == s) & (news.move_1h <= -0.02) & (news.dollar_vol_m >= dv) & (news.atr_pct >= 3)]["hour_after_r10"].dropna() - 0.001
        c = ctl[(ctl["set"] == s) & (ctl.dollar_vol_m >= dv) & (ctl.atr_pct >= 3)]["h11_r10"].dropna() - 0.001
        rows.append({"dv_floor_M$": dv, "set": s, "n": len(d), "edge_pct": round((d.mean() - c.mean()) * 100, 2), "avg_net_pct": round(d.mean() * 100, 2), "t": round(d.mean() / (d.std() / len(d) ** .5), 1)})
t = pd.DataFrame(rows); t.to_csv("research/audit/out_t12_liquidity.csv", index=False)
pd.set_option("display.width", 180)
print("(a) liquidity floor, volatile stocks only:"); print(t.pivot_table(index="dv_floor_M$", columns="set", values=["n", "edge_pct", "t"]).round(2).to_string())
# (b) power: trades per year, per-trade std, size 10%
x = news[(news.move_1h <= -0.02) & (news.dollar_vol_m >= 300) & (news.atr_pct >= 3)]["hour_after_r10"].dropna() - 0.001
yrs = 3.6; per_year = len(x) / yrs; sd = x.std(); mean = x.mean()
print("\n(b) POWER: the rule makes about %.0f trades a year in the whole event universe; per-trade mean %.2f%%, std %.1f%%" % (per_year, mean * 100, sd * 100))
for cap_trades in (40, 60, 100):                                      # trades a year the live agent can actually take (2/day, cap 30%, 10%, 10 days)
    t_per_year = mean / sd * np.sqrt(cap_trades)
    print(f"   if the live agent takes ~{cap_trades} trades a year: t grows by {t_per_year:.2f} per sqrt(year) -> about {(2 / t_per_year) ** 2:.1f} years of live results to reach t=2 (if the edge is real and equal to the measured one)")
half = mean / 2
for cap_trades in (40, 60, 100):
    t_per_year = half / sd * np.sqrt(cap_trades)
    print(f"   with the edge HALVED: about {(2 / t_per_year) ** 2:.1f} years for t=2 at {cap_trades} trades a year")
