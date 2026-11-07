"""News Hunter with extra filters on the stock's own state, judged on dev and holdout separately (alpha vs SPY, idle money in SPY)."""
import itertools, numpy as np, pandas as pd
from research.audit import run_news_lib as R

FILTERS = {"none (current rule)": lambda d: d.move_1h.notna(),
           "below 50d avg": lambda d: d.above_sma50 == 0,
           "below 200d avg": lambda d: d.above_sma200 == 0,
           "fell >5% in prior 20d": lambda d: d.ret20_before < -0.05,
           "wild (atr>=3%)": lambda d: d.atr_pct >= 3,
           "below 50d AND wild": lambda d: (d.above_sma50 == 0) & (d.atr_pct >= 3),
           "below 50d OR fell>5%": lambda d: (d.above_sma50 == 0) | (d.ret20_before < -0.05)}
def sim(d, per_day, size, h, col, base):
    open_pos, net = [], []
    for i, g in d.sort_values(["i", "move_1h"]).groupby("i", sort=True):
        open_pos = [p for p in open_pos if p[0] > i]
        exposure, held, took = sum(p[1] for p in open_pos), {p[2] for p in open_pos}, 0
        for sym, r in zip(g.symbol.to_numpy(), g[col].to_numpy()):
            if took >= per_day or exposure + size > R.CAP + 1e-9: break
            if sym in held or np.isnan(r) or i + h >= len(R.dates): continue
            open_pos.append((i + h, size, sym)); held.add(sym); exposure += size; took += 1; net.append(r - R.COST_RT)
    return np.array(net)
rows = []
for s in ("dev", "holdout"):
    yrs = R.YEARS[s]
    for (fname, f), h, size, per_day, dv in itertools.product(FILTERS.items(), (5, 10), (0.075, 0.10), (2, 3), (300, 1000)):
        d = R.news[(R.news["set"] == s) & (R.news.move_1h <= -0.02) & (R.news.dollar_vol_m >= dv)]
        d = d[f(d)]
        col = f"hour_after_r{h}"
        net = sim(d, per_day, size, h, col, R.base[(s, dv, h, "r")])
        n = len(net); b = R.base[(s, dv, h, "r")]
        rows.append({"set": s, "filter": fname, "hold": h, "size": size, "per_day": per_day, "dv": dv, "trades_yr": round(n / yrs, 1),
                     "alpha_yr": round(size * net.sum() / yrs * 100, 2) if n else 0.0,
                     "edge_pct": round((net.mean() - (0 if np.isnan(b) else b)) * 100, 2) if n else np.nan,
                     "t": round(net.mean() / (net.std() / n ** .5), 1) if n > 10 and net.std() > 0 else np.nan})
out = pd.DataFrame(rows); out.to_csv("research/audit/out_news_filters.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
m = out.pivot_table(index="filter", columns="set", values=["alpha_yr", "edge_pct", "trades_yr"], aggfunc="mean", sort=False).round(2)
print(m.to_string())
