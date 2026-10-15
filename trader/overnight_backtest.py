"""Overnight trades: buy near the close, sell the next morning. Tested on real 5-minute bars, after costs.

    python -m trader.overnight_backtest --years 3

Every combination of
  entry:  15:30, or the 16:00 close
  exit:   next day's 09:30 open, 10:00, or 10:30
  filter: which ETFs to buy that evening (all of them, only after an up or down day, only in an
          uptrend, after a 3-day drop, the day's biggest losers or winners, skip Fridays, ...)
is tested on the same ETFs as the day-trading test. Signals only use what is known at entry time.

For each combination: number of trades, win rate and average return after costs, a t-statistic
(how sure we can be it isn't luck; with this many combinations, only |t| above ~3 is convincing),
whether it made money in both halves of the period, and a $200 pot that each night is split evenly
over that night's trades. The pot is compared with holding $200 of SPY the whole time.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .daytrade_backtest import COST_PER_SIDE, DEFAULT_COST, fetch

log = logging.getLogger("overnight")
POT = 200.0

ENTRIES = {"15:30": "p1530", "close": "close"}
EXITS = {"open": "n_open", "10:00": "n_1000", "10:30": "n_1030"}


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per full trading day with the prices each strategy needs (5-minute regular-session bars)."""
    ny = df.index.tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    rows = {}
    for d, idx in pd.Series(range(len(df)), index=ny.date).groupby(level=0):
        day = df.iloc[idx.to_numpy()]
        m = np.asarray(mins[idx.to_numpy()])
        if m[0] != 570 or m[-1] < 955:                 # needs the 09:30 and 15:55 bars (skips half days)
            continue

        def at(t):
            i = int(np.searchsorted(m, t))
            return float(day["open"].iloc[i]) if i < len(m) else np.nan

        upto = m < 930                                 # bars before 15:30
        rows[d] = {"open": float(day["open"].iloc[0]), "p1000": at(600), "p1030": at(630), "p1500": at(900),
                   "p1530": at(930), "close": float(day["close"].iloc[-1]),
                   "hi": float(day["high"][upto].max()), "lo": float(day["low"][upto].min())}
    t = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    t["prev_close"] = t["close"].shift(1)
    t["sma20"] = t["close"].rolling(20).mean().shift(1)
    t["close_3ago"] = t["close"].shift(3)
    for col, src in (("n_open", "open"), ("n_1000", "p1000"), ("n_1030", "p1030")):
        t[col] = t[src].shift(-1)
    t["weekday"] = [d.weekday() for d in t.index]
    return t


def _filters(t: pd.DataFrame, px: pd.Series, spy_up: pd.Series, rank: pd.Series, n_syms: int) -> dict[str, pd.Series]:
    """Which days to buy, using only information available at the entry price `px`."""
    day = px / t["prev_close"] - 1
    pos = (px - t["lo"]) / (t["hi"] - t["lo"]).replace(0, np.nan)
    trend = px > t["sma20"]
    return {
        "all": pd.Series(True, index=t.index),
        "up_day": day > 0,
        "down_day": day < 0,
        "big_up_day(>1%)": day > 0.01,
        "big_down_day(<-1%)": day < -0.01,
        "uptrend": trend,
        "downtrend": ~trend & t["sma20"].notna(),
        "market_uptrend": spy_up.reindex(t.index).fillna(False).astype(bool),
        "dip_in_uptrend": trend & (day < -0.005),
        "last_30min_up": t["p1530"] / t["p1500"] - 1 > 0,
        "last_30min_down": t["p1530"] / t["p1500"] - 1 < 0,
        "near_day_high": pos > 0.8,
        "near_day_low": pos < 0.2,
        "3day_drop(<-3%)": px / t["close_3ago"] - 1 < -0.03,
        "skip_fridays": t["weekday"] != 4,
        "fridays_only": t["weekday"] == 4,
        "3_biggest_losers": rank <= 3,
        "3_biggest_winners": rank > n_syms - 3,
    }


def run(bars: dict[str, pd.DataFrame]) -> dict:
    tables = {s: daily_table(df) for s, df in bars.items()}
    tables = {s: t for s, t in tables.items() if len(t) > 25}
    if "SPY" not in tables:
        raise ValueError("SPY is needed for the market filter and the benchmark")
    spy = tables["SPY"]
    results = []
    for ename, ecol in ENTRIES.items():
        spy_up = spy[ecol] > spy["sma20"]
        day_moves = pd.DataFrame({s: t[ecol] / t["prev_close"] - 1 for s, t in tables.items()})
        ranks = day_moves.rank(axis=1)
        n_syms = day_moves.notna().sum(axis=1)
        for xname, xcol in EXITS.items():
            trades: dict[str, list] = {}               # filter -> [(date, symbol, net return)]
            for sym, t in tables.items():
                cost = 2 * COST_PER_SIDE.get(sym, DEFAULT_COST)
                ret = t[xcol] / t[ecol] - 1
                ok = ret.notna()
                for fname, mask in _filters(t, t[ecol], spy_up, ranks[sym].reindex(t.index),
                                            n_syms.reindex(t.index)).items():
                    sel = ret[ok & mask.fillna(False).astype(bool)]
                    trades.setdefault(fname, []).extend((d, sym, r - cost) for d, r in sel.items())
            for fname, tr in trades.items():
                results.append(_score(f"buy {ename}, sell {xname}, {fname}", tr))
    results.sort(key=lambda r: -r.get("pot_end", 0))
    days = spy.index[spy["n_open"].notna()]
    years = len(days) / 252
    spy_hold = POT * spy["close"].iloc[-1] / spy["close"].iloc[0]
    spy_overnight = POT * float(np.prod(1 + (spy["n_open"] / spy["close"] - 1).dropna()))
    spy_daytime = POT * float(np.prod(1 + (spy["close"] / spy["open"] - 1)))
    return {"days": len(days), "first_day": str(spy.index[0]), "last_day": str(spy.index[-1]),
            "years": round(years, 2), "symbols": sorted(tables), "pot_start": POT,
            "spy_hold_end": round(spy_hold, 2),
            "spy_hold_yearly_pct": round(((spy_hold / POT) ** (1 / years) - 1) * 100, 1),
            "spy_only_nights_no_costs": round(spy_overnight, 2),
            "spy_only_daytime_no_costs": round(spy_daytime, 2),
            "combinations": len(results), "strategies": results}


def _score(name: str, trades: list) -> dict:
    if len(trades) < 20:
        return {"name": name, "trades": len(trades)}
    nets = [r for _, _, r in trades]
    by_day: dict = {}
    for d, _, r in trades:
        by_day.setdefault(d, []).append(r)
    days = sorted(by_day)
    pot = POT
    for d in days:
        pot *= 1 + statistics.mean(by_day[d])
    all_days = sorted({d for d, _, _ in trades})
    mid = all_days[len(all_days) // 2]
    first = [r for d, _, r in trades if d < mid]
    second = [r for d, _, r in trades if d >= mid]
    sd = statistics.stdev(nets)
    span_years = max((days[-1] - days[0]).days / 365.25, 0.1)
    return {
        "name": name,
        "trades": len(nets),
        "nights_traded": len(days),
        "win_rate": round(sum(r > 0 for r in nets) / len(nets), 3),
        "avg_net_pct": round(statistics.mean(nets) * 100, 4),
        "t_stat": round(statistics.mean(nets) / (sd / len(nets) ** 0.5), 2) if sd > 0 else None,
        "first_half_avg_pct": round(statistics.mean(first) * 100, 4) if first else None,
        "second_half_avg_pct": round(statistics.mean(second) * 100, 4) if second else None,
        "pot_end": round(pot, 2),
        "pot_yearly_pct": round(((pot / POT) ** (1 / span_years) - 1) * 100, 1),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--out", default="docs/data/overnight_backtest.json")
    args = ap.parse_args()
    result = run(fetch(args.years))
    result["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "strategies"}, indent=1))
    for r in result["strategies"][:20]:
        print(r)


if __name__ == "__main__":
    main()
