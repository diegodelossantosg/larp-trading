"""News Hunter: every combination of liquidity floor, drop size, buys/day, size, hold and stop, on the three
company sets of research/events.csv.gz (dev = design, big = biased-up check, holdout = final). Writes out_news.csv.

Each trade's result is the stock's return minus SPY's over the hold (the event file stores it that way), minus 0.1%
round trip. 'edge' also subtracts what random days of the same companies earned the same way (luck check).
"""
import itertools, time, numpy as np, pandas as pd
from trader.lab import Lab

L = Lab(); dates = L.dates; didx = {str(d.date()): i for i, d in enumerate(dates)}
ev = pd.read_csv("research/events.csv.gz", low_memory=False)
news, ctl = ev[ev.kind == "news"].copy(), ev[ev.kind == "control"].copy()
news["i"] = news.day.map(didx)
COST_RT, CAP = 0.001, 0.30
DVS, DROPS, PERDAY, SIZES, HOLDS, STOPS = (20, 50, 100, 150, 300, 600, 1000), (0.01, 0.015, 0.02, 0.03, 0.04), (1, 2, 3, 5, 10), (0.025, 0.05, 0.075, 0.10, 0.15), (5, 10, 20), ("r", "s")
YEARS = {s: (pd.Timestamp(g.day.max()) - pd.Timestamp(g.day.min())).days / 365.25 for s, g in news.groupby("set")}

# random-day average per (set, liquidity floor, hold, stop): the luck baseline
base = {}
for s in ("dev", "big", "holdout"):
    for dv in DVS:
        c = ctl[(ctl["set"] == s) & (ctl.dollar_vol_m >= dv)]
        for h in HOLDS:
            for st in STOPS:
                v = c[f"h11_{st}{h}"].dropna()
                base[(s, dv, h, st)] = float(v.mean()) if len(v) > 30 else np.nan

def simulate(sub: pd.DataFrame, per_day: int, size: float, h: int, st: str, ret_col: str, base_ret: float):
    """sub: candidate events (already filtered), sorted by day then biggest drop. Returns (alpha_yr pct, trades, edge_yr pct, years +)"""
    open_pos = []                       # (exit_i, size, symbol)
    pnl, edge, ys = 0.0, 0.0, {}
    n = 0
    for i, g in sub.groupby("i", sort=True):
        open_pos = [p for p in open_pos if p[0] > i]
        exposure = sum(p[1] for p in open_pos)
        held = {p[2] for p in open_pos}
        took = 0
        for sym, r in zip(g.symbol.to_numpy(), g[ret_col].to_numpy()):
            if took >= per_day or exposure + size > CAP + 1e-9: break
            if sym in held or np.isnan(r) or i + h >= len(dates): continue
            open_pos.append((i + h, size, sym)); held.add(sym); exposure += size; took += 1; n += 1
            gain = size * (r - COST_RT)
            pnl += gain; edge += size * (r - COST_RT - base_ret)
            y = dates[i + h].year; ys[y] = ys.get(y, 0.0) + gain
    return pnl, edge, n, ys

rows, t0 = [], time.time()
for s in ("dev", "big", "holdout"):
    ns = news[news["set"] == s]
    for dv, drop in itertools.product(DVS, DROPS):
        cand = ns[(ns.move_1h <= -drop) & (ns.dollar_vol_m >= dv)].sort_values(["i", "move_1h"])
        if len(cand) < 20: continue
        for h, st in itertools.product(HOLDS, STOPS):
            col = f"hour_after_{st}{h}"; b = base[(s, dv, h, st)]
            for per_day, size in itertools.product(PERDAY, SIZES):
                pnl, edge, n, ys = simulate(cand, per_day, size, h, st, col, 0.0 if np.isnan(b) else b)
                yrs = YEARS[s]
                rows.append({"set": s, "dv": dv, "drop": drop, "per_day": per_day, "size": size, "hold": h, "stop": st,
                             "trades": n, "trades_yr": round(n / yrs, 1), "alpha_yr": round(pnl / yrs * 100, 3), "edge_yr": round(edge / yrs * 100, 3),
                             "avg_trade_pct": round(pnl / max(n, 1) / size * 100, 3), "years_pos": sum(v > 0 for v in ys.values()), "years": len(ys)})
    print(s, round(time.time() - t0), "s", len(rows), flush=True)
pd.DataFrame(rows).to_csv("research/audit/out_news.csv", index=False)
