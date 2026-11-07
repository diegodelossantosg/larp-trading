"""T1: does the edge survive buying later than the ideal moment (the live agent acts hourly + Opus time) and bigger trading costs?
Volatile stocks (ATR>=3%) that fell >=2% vs SPY an hour after the headline, dv>=300M, dev and holdout separately."""
import numpy as np, pandas as pd
ev = pd.read_csv("research/events.csv.gz", low_memory=False)
ev = ev[ev["set"].isin(["dev", "holdout"])]
news, ctl = ev[ev.kind == "news"], ev[ev.kind == "control"]
sel = news[(news.move_1h <= -0.02) & (news.dollar_vol_m >= 300) & (news.atr_pct >= 3)]
rows = []
for entry in ("hour_after", "h11", "close", "next_open"):
    for h in (5, 10):
        for cost in (0.001, 0.003, 0.006, 0.010):
            for s in ("dev", "holdout"):
                x = sel[sel["set"] == s][f"{entry}_r{h}"].dropna()
                cbase = "h11" if entry == "hour_after" else entry
                c = ctl[(ctl["set"] == s) & (ctl.dollar_vol_m >= 300) & (ctl.atr_pct >= 3)][f"{cbase}_r{h}"].dropna()
                net = x - cost
                rows.append({"entry": entry, "hold": h, "cost_roundtrip": cost, "set": s, "n": len(x), "avg_net_pct": round(net.mean() * 100, 2),
                             "random_days_pct": round((c.mean() - cost) * 100, 2), "edge_pct": round((net.mean() - (c.mean() - cost)) * 100, 2),
                             "t": round(net.mean() / (net.std() / len(x) ** .5), 1)})
d = pd.DataFrame(rows); d.to_csv("research/audit/out_t1.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
print("EDGE vs random days (%/trade) by entry timing (cost 0.1%), hold 10:")
print(d[(d.cost_roundtrip == 0.001) & (d.hold == 10)].pivot_table(index="entry", columns="set", values=["edge_pct", "avg_net_pct", "n", "t"], sort=False).round(2).to_string())
print("\nAVERAGE NET %/trade by cost, entry hour_after, hold 10:")
print(d[(d.entry == "hour_after") & (d.hold == 10)].pivot_table(index="cost_roundtrip", columns="set", values=["avg_net_pct", "t"]).round(2).to_string())
print("\nsame for entry close (3-5h later):")
print(d[(d.entry == "close") & (d.hold == 10)].pivot_table(index="cost_roundtrip", columns="set", values=["avg_net_pct", "t"]).round(2).to_string())
