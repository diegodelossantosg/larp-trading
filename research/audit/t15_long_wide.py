"""T15: the News Hunter rule on the two big downloads (events_long: 2018-2026, ~190 companies; events_wide: 2023-2026,
~530 companies). Run from the repo root:  python -m research.audit.t15_long_wide

Idle money is SPY, so every number is the stock's return minus SPY's (the event file stores it that way), minus 0.1%
round trip. 'edge' also subtracts what random days of the same companies earned the same way (luck check).
Periods: 'pre' = before 2022-07 (days the first 40-month file never saw: a real out-of-sample test), 'post' = after.
"""
import itertools, numpy as np, pandas as pd
from trader.lab import Lab

L = Lab(); dates = L.dates; didx = {str(d.date()): i for i, d in enumerate(dates)}
COST, CAP = 0.001, 0.30
SPLIT = pd.Timestamp("2022-07-01")


def load(name):
    ev = pd.read_csv(f"research/{name}.csv.gz", low_memory=False)
    ev["d"] = pd.to_datetime(ev.day)
    news, ctl = ev[ev.kind == "news"].copy(), ev[ev.kind == "control"].copy()
    news["i"] = news.day.map(didx)
    return news.dropna(subset=["i"]), ctl


def run(news, ctl, lo, hi, atr, drop, dv, per_day, size, h, st):
    n = news[(news.d >= lo) & (news.d < hi) & (news.move_1h <= -drop) & (news.dollar_vol_m >= dv) & (news.atr_pct >= atr)]
    c = ctl[(ctl.d >= lo) & (ctl.d < hi) & (ctl.dollar_vol_m >= dv) & (ctl.atr_pct >= atr)][f"h11_{st}{h}"].dropna()
    base = float(c.mean()) if len(c) > 30 else 0.0
    col, open_pos, net, days = f"hour_after_{st}{h}", [], [], set()
    for i, g in n.sort_values(["i", "move_1h"]).groupby("i", sort=True):
        i = int(i); open_pos = [p for p in open_pos if p[0] > i]
        exposure, held, took = sum(p[1] for p in open_pos), {p[2] for p in open_pos}, 0
        for sym, r in zip(g.symbol.to_numpy(), g[col].to_numpy()):
            if took >= per_day or exposure + size > CAP + 1e-9: break
            if sym in held or np.isnan(r) or i + h >= len(dates): continue
            open_pos.append((i + h, size, sym)); held.add(sym); exposure += size; took += 1
            net.append(r - COST)
    yrs = max((hi - lo).days / 365.25, 1e-9)
    net = np.array(net); k = len(net)
    return {"trades_yr": round(k / yrs, 1), "alpha_yr": round(size * net.sum() / yrs * 100, 2) if k else 0.0,
            "edge_pct": round((net.mean() - base) * 100, 2) if k else np.nan,
            "t": round((net.mean() - base) / (net.std() / k ** .5), 1) if k > 10 and net.std() > 0 else np.nan, "n": k}


rows = []
for name in ("events_long", "events_wide"):
    news, ctl = load(name)
    end = news.d.max() + pd.Timedelta(days=1)
    periods = {"all": (news.d.min(), end), "pre": (news.d.min(), SPLIT), "post": (SPLIT, end)}
    if name == "events_wide":
        periods = {"all": periods["all"]}
    for pname, (lo, hi) in periods.items():
        for atr, dv, h, st in itertools.product((0, 3), (20, 100, 300, 600, 1000), (5, 10), ("r", "s")):
            r = run(news, ctl, lo, hi, atr, 0.02, dv, 2, 0.10, h, st)
            rows.append({"file": name, "period": pname, "atr>=": atr, "floor_M": dv, "hold": h, "stop": st, **r})
out = pd.DataFrame(rows); out.to_csv("research/audit/out_t15.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
print(out.groupby(["file", "period", "atr>=", "floor_M"])[["trades_yr", "alpha_yr", "edge_pct", "t"]].mean().round(2).to_string())
print("\ncurrent rule (atr>=3, floor 20, hold 5/10, r):")
print(out[(out["atr>="] == 3) & (out.floor_M == 20) & (out.stop == "r")].to_string(index=False))
