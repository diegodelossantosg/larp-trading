"""Would day trading "stocks in play" (sold by the close every day) have made money after costs?

    python -m trader.inplay_data --out <dir>      # download once (about an hour)
    python -m trader.inplay_backtest --raw <dir>

The rule is the "opening range breakout on stocks in play" from Zarattini, Barbon & Aziz (2024),
which reports large profits from 2016 on:
1. Universe: that year's 500 most-traded stocks (research/lab.npz, delisted ones included), price
   at least $5, 14-day average volume at least 1M shares, 14-day ATR at least $0.50.
2. In play: the 20 stocks whose first 5 minutes traded the most volume relative to their own last 14
   days (relative volume at least 1).
3. Direction: opening range closes up -> buy-stop at its high; closes down -> sell-short-stop at its low.
4. Stop loss 10% of the ATR from the entry. Everything is closed at 16:00.
"random" applies the same rule to 20 random universe stocks each day: in-play selection only adds
something if it clearly beats that. Each stock-day is tested at opening ranges of 5, 15 and 30
minutes (1-minute bars, so the tight stop is checked minute by minute), and long and short are
reported apart (a cash account cannot short).

Fills: the entry fills at the trigger, or the bar's open if it gapped through. If the entry bar also
touches the stop, the trade is counted as stopped out (the worst case). Costs per side are spread +
slippage, tested at 0.02%, 0.05% (main) and 0.10%.

Money: "pot_1x" splits the pot evenly over the day's candidates, never borrowing. "pot_paper" sizes
like the paper: each trade risks 1% of the pot, with at most 4x leverage in total.
Anti-luck: results are shown per symbol group (A design, B validation, C final check), per year,
and for the first and second half of the period.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import statistics
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("inplay")

COSTS = (0.0002, 0.0005, 0.001)
MAIN_COST = 0.0005
RANGES = (5, 15, 30)        # opening range in minutes (1-minute bars)
STOP_ATR = 0.10
MIN_PRICE, MIN_AVG_VOL, MIN_ATR, MIN_RVOL = 5.0, 1e6, 0.5, 1.0
TOP = 20
POT = 1000.0
RISK_PER_TRADE, MAX_LEVERAGE = 0.01, 4.0


def simulate(o, h, l, c, k: int, atr: float) -> tuple[int, float, float] | None:
    """One stock-day. Returns (direction, entry, exit) or None if there was no trade.

    o/h/l/c: the day's 1-minute bars from 09:30 (the last bar ends at 16:00); k: bars in the opening range.
    """
    n = len(c)
    if n <= k or atr <= 0:
        return None
    if c[k - 1] > o[0]:
        d, trig = 1, float(np.max(h[:k]))
    elif c[k - 1] < o[0]:
        d, trig = -1, float(np.min(l[:k]))
    else:
        return None
    for i in range(k, n):
        if d == 1 and h[i] >= trig:
            entry = max(trig, float(o[i]))
            stop = entry - STOP_ATR * atr
        elif d == -1 and l[i] <= trig:
            entry = min(trig, float(o[i]))
            stop = entry + STOP_ATR * atr
        else:
            continue
        if (d == 1 and l[i] <= stop) or (d == -1 and h[i] >= stop):
            return d, entry, stop
        for j in range(i + 1, n):
            if d == 1 and l[j] <= stop:
                return d, entry, min(stop, float(o[j]))
            if d == -1 and h[j] >= stop:
                return d, entry, max(stop, float(o[j]))
        return d, entry, float(c[-1])
    return None


def daily_stats(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per symbol: yesterday's 14-day ATR and average volume (known before the open), indexed by date."""
    out = {}
    for sym, d in daily.groupby(level="symbol"):
        d = d.droplevel("symbol")
        d.index = d.index.tz_convert("America/New_York").date
        prev = d["close"].shift(1)
        tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1).max(axis=1)
        out[sym] = pd.DataFrame({"atr": tr.rolling(14).mean().shift(1), "avg_vol": d["volume"].rolling(14).mean().shift(1)})
    return out


def collect(raw: str, groups: dict[str, str]) -> list[dict]:
    """Every simulated trade: day, symbol, group, set (inplay/random), range, direction, gross return, R."""
    daily = pd.read_pickle(os.path.join(raw, "daily.pkl"))
    stats = daily_stats(daily)
    trades = []
    for path in sorted(glob.glob(os.path.join(raw, "days", "*.pkl"))):
        rec = pd.read_pickle(path)
        day = datetime.strptime(os.path.basename(path)[:10], "%Y-%m-%d").date()
        rvol = rec.get("rvol", {})
        for set_name, syms in (("inplay", rec["picks"]), ("random", rec["random"])):
            chosen = []
            for s in syms:
                st = stats.get(s)
                bars = rec.get("bars1", {}).get(s)
                if st is None or bars is None or day not in st.index or len(bars) < 350:
                    continue
                atr, avg_vol = st.at[day, "atr"], st.at[day, "avg_vol"]
                if not (atr >= MIN_ATR and avg_vol >= MIN_AVG_VOL and bars["open"].iloc[0] >= MIN_PRICE):
                    continue
                if set_name == "inplay" and not rvol.get(s, 0) >= MIN_RVOL:
                    continue
                chosen.append((s, bars, float(atr)))
            for s, bars, atr in chosen[:TOP]:
                o, h, l, c = (bars[f].to_numpy(dtype=float) for f in ("open", "high", "low", "close"))
                for k in RANGES:
                    r = simulate(o, h, l, c, k, atr)
                    if r is None:
                        trades.append({"day": day, "sym": s, "group": groups.get(s, "?"), "set": set_name, "k": k,
                                       "n_cand": len(chosen[:TOP]), "dir": 0})
                        continue
                    d, entry, exit_ = r
                    trades.append({"day": day, "sym": s, "group": groups.get(s, "?"), "set": set_name, "k": k,
                                   "n_cand": len(chosen[:TOP]), "dir": d, "gross": d * (exit_ / entry - 1),
                                   "risk": STOP_ATR * atr / entry})
    return trades


def _summary(t: pd.DataFrame, cost: float) -> dict:
    t = t[t["dir"] != 0]
    if len(t) < 3:
        return {"trades": int(len(t))}
    net = t["gross"] - 2 * cost
    sd = float(net.std())
    return {"trades": int(len(t)), "win_rate": round(float((net > 0).mean()), 3),
            "avg_gross_pct": round(float(t["gross"].mean()) * 100, 4), "avg_net_pct": round(float(net.mean()) * 100, 4),
            "t_stat": round(float(net.mean()) / (sd / math.sqrt(len(net))), 2) if sd > 0 else None}


def _pots(t: pd.DataFrame, cost: float, years: float) -> dict:
    """Compounded $1,000 over the days, with even split (1x) and with the paper's risk sizing (up to 4x)."""
    p1 = pp = POT
    for _day, g in t.groupby("day"):
        n = int(g["n_cand"].iloc[0]) or 1
        tr = g[g["dir"] != 0]
        net = (tr["gross"] - 2 * cost).to_numpy()
        p1 *= 1 + net.sum() / n
        if len(tr):
            w = np.minimum(RISK_PER_TRADE / tr["risk"].to_numpy(), MAX_LEVERAGE)
            w *= min(1.0, MAX_LEVERAGE / w.sum())
            pp *= max(1 + float((w * net).sum()), 0.0)
    return {"pot_1x_end": round(p1, 2), "pot_1x_yearly_pct": round(((p1 / POT) ** (1 / years) - 1) * 100, 1),
            "pot_paper_end": round(pp, 2),
            "pot_paper_yearly_pct": round(((max(pp, 1e-9) / POT) ** (1 / years) - 1) * 100, 1)}


def report(trades: list[dict], spy: pd.Series | None = None) -> dict:
    df = pd.DataFrame(trades)
    days = sorted(df["day"].unique())
    years = max(len(days) / 252, 1e-9)
    mid = days[len(days) // 2]
    out = {"days": len(days), "first_day": str(days[0]), "last_day": str(days[-1]), "cost_per_side_main": MAIN_COST,
           "variants": {}}
    for set_name in ("inplay", "random"):
        for k in RANGES:
            base = df[(df["set"] == set_name) & (df["k"] == k)]
            for side, part in (("both", base), ("long", base[base["dir"] >= 0]), ("short", base[base["dir"] <= 0])):
                v = {"by_cost": {f"{c * 100:.2f}%": _summary(part, c) for c in COSTS}}
                v.update(_pots(part, MAIN_COST, years))
                v["groups"] = {g: _summary(part[part["group"] == g], MAIN_COST) for g in "ABC"}
                v["halves"] = {"first": _summary(part[part["day"] < mid], MAIN_COST),
                               "second": _summary(part[part["day"] >= mid], MAIN_COST)}
                v["years"] = {str(y): _summary(part[pd.to_datetime(part["day"]).dt.year == y], MAIN_COST)
                              for y in sorted({d.year for d in days})}
                out["variants"][f"{set_name}_{k}min_{side}"] = v
    if spy is not None:
        s = spy[(spy.index >= pd.Timestamp(days[0])) & (spy.index <= pd.Timestamp(days[-1]))]
        out["spy_hold_end"] = round(POT * float(s.iloc[-1] / s.iloc[0]), 2)
        out["spy_hold_yearly_pct"] = round(((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100, 1)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--lab", default="research/lab.npz")
    ap.add_argument("--out", default="research/inplay_backtest.json")
    args = ap.parse_args()
    z = np.load(args.lab)
    groups = dict(zip(z["symbols"], z["groups"]))
    spy = pd.Series(z["close"][:, list(z["symbols"]).index("SPY")].astype(float), index=pd.DatetimeIndex(z["dates"]))
    result = report(collect(args.raw, groups), spy.dropna())
    result["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    v = result["variants"]
    for name in v:
        m = v[name]["by_cost"][f"{MAIN_COST * 100:.2f}%"]
        print(f"{name:28s} trades={m.get('trades')} net={m.get('avg_net_pct')}% t={m.get('t_stat')} "
              f"pot1x={v[name]['pot_1x_end']} potpaper={v[name]['pot_paper_end']}")
    print("SPY hold:", result.get("spy_hold_end"))


if __name__ == "__main__":
    main()
