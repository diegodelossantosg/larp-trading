"""More News Hunter ideas on the current event file, each judged on dev and holdout separately (volatile stocks only, 10%, 2/day, dv>=300).
Selection rule among the day's candidates, early exit on a quick rebound, market filters, hold length."""
import numpy as np, pandas as pd
from research.audit import run_news_lib as R

SIZE, PER_DAY, DV, DROP = 0.10, 2, 300, 0.02
rng = np.random.default_rng(0)
def sim(d, h, order, ret_fn, per_day=PER_DAY):
    open_pos, net = [], []
    for i, g in d.groupby("i", sort=True):
        g = order(g)
        open_pos = [p for p in open_pos if p[0] > i]
        exposure, held, took = sum(p[1] for p in open_pos), {p[2] for p in open_pos}, 0
        for _, row in g.iterrows():
            if took >= per_day or exposure + SIZE > R.CAP + 1e-9: break
            r = ret_fn(row)
            if row.symbol in held or np.isnan(r) or i + h >= len(R.dates): continue
            open_pos.append((i + h, SIZE, row.symbol)); held.add(row.symbol); exposure += SIZE; took += 1; net.append(r - R.COST_RT)
    return np.array(net)
ORD = {"biggest drop (current)": lambda g: g.sort_values("move_1h"),
       "most liquid": lambda g: g.sort_values("dollar_vol_m", ascending=False),
       "most volatile": lambda g: g.sort_values("atr_pct", ascending=False),
       "weakest before (20d fall)": lambda g: g.sort_values("ret20_before"),
       "random (average of 1 draw)": lambda g: g.sample(frac=1, random_state=int(g.i.iloc[0]))}
rows = []
for s in ("dev", "holdout"):
    yrs = R.YEARS[s]
    base_d = R.news[(R.news["set"] == s) & (R.news.move_1h <= -DROP) & (R.news.dollar_vol_m >= DV) & (R.news.atr_pct >= 3)]
    def add(name, group, d, h, order, fn):
        net = sim(d, h, order, fn); n = len(net)
        rows.append({"test": group, "variant": name, "set": s, "trades_yr": round(n / yrs, 1), "alpha_yr": round(SIZE * net.sum() / yrs * 100, 2) if n else 0,
                     "avg_trade_pct": round(net.mean() * 100, 2) if n else np.nan,
                     "t": round(net.mean() / (net.std() / n ** .5), 1) if n > 10 and net.std() > 0 else np.nan})
    cur = lambda h: (lambda row: row[f"hour_after_r{h}"])
    for name, o in ORD.items():
        add(name, "which candidate first", base_d, 10, o, cur(10))
    for h in (5, 10, 20):
        add(f"hold {h} days", "hold length", base_d, h, ORD["biggest drop (current)"], cur(h))
    for thr in (0.02, 0.03, 0.05):
        add(f"hold 10, but sell at day 5 if already +{int(thr*100)}%", "early exit", base_d, 10, ORD["biggest drop (current)"],
            lambda row, thr=thr: row["hour_after_r5"] if row["hour_after_r5"] >= thr else row["hour_after_r10"])
    add("sell at day 5 if +3%, else hold 20", "early exit", base_d, 20, ORD["biggest drop (current)"],
        lambda row: row["hour_after_r5"] if row["hour_after_r5"] >= 0.03 else row["hour_after_r20"])
    for name, mask in {"all days": base_d.move_1h.notna(), "market above 200d avg": base_d.spy_above_sma200 == 1,
                       "market below 200d avg": base_d.spy_above_sma200 == 0, "market fell >5% in 20d": base_d.spy_ret20_before < -0.05,
                       "market not falling (20d > -3%)": base_d.spy_ret20_before > -0.03}.items():
        add(name, "market filter", base_d[mask], 10, ORD["biggest drop (current)"], cur(10))
out = pd.DataFrame(rows); out.to_csv("research/audit/out_more_news.csv", index=False)
pd.set_option("display.width", 230); pd.set_option("display.max_rows", 100)
p = out.pivot_table(index=["test", "variant"], columns="set", values=["alpha_yr", "avg_trade_pct", "trades_yr", "t"], sort=False).round(2)
print(p[[("alpha_yr", "dev"), ("alpha_yr", "holdout"), ("avg_trade_pct", "dev"), ("avg_trade_pct", "holdout"), ("trades_yr", "dev"), ("trades_yr", "holdout")]].to_string())
