"""Old way (ETFs only) vs new way (ETFs + company stocks), on past data.

    python -m trader.stocks_backtest --years 9 --watch-months 12

Claude's daily choices can't be replayed on the past (it would know what happened next), so two
rule-based stand-ins are compared, each run once with ETFs only and once with stocks allowed:

1. Picker (stand-in for the daily decision), daily prices since 2017: every month, 50% goes to SSO
   and 50% to the 6 strongest trends (6-month return divided by volatility, only names above their
   200-day average). New way: the big stocks compete with the ETFs, with the live rules: at most
   4 stocks (4 x 8.3% < 40% total). A trailing stop (6% ETFs, 10% stocks) sells a pick early;
   its money waits in cash until the next month. Trading costs 0.05% per side.
2. Watcher, 15-minute prices: the live buy signals with ETFs only vs ETFs plus big stocks
   (see trader/backtest.py).

Big caveat, survivorship bias: the stock list is today's big, well-known companies. Some are big
precisely because they soared (Nvidia, Palantir, Broadcom...), and companies that shrank aren't on
the list. That flatters the new way. The "without the biggest winners" run removes the six largest
winners of the period to show how much the result depends on them.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .universe import BIG_STOCKS, CORE_ETFS, LEVERAGED

log = logging.getLogger("stocks_backtest")

CAPITAL = 1000.0
COST = 0.0005
N_PICKS, MAX_STOCKS = 6, 4
ETF_STOP, STOCK_STOP = 0.06, 0.10


def picker(closes: pd.DataFrame, candidates: list[str], stocks: set[str]) -> pd.Series:
    """Daily portfolio value of: 50% SSO + 50% in the month's strongest trends, with trailing stops."""
    rets = closes.pct_change().fillna(0.0)
    mom = closes / closes.shift(126) - 1
    vol = rets.rolling(126).std() * np.sqrt(252)
    above = closes > closes.rolling(200).mean()
    score = (mom / vol).where(above)
    month_ends = set(closes.groupby(closes.index.to_period("M")).tail(1).index)

    value, weights, highs = CAPITAL, {}, {}
    out = []
    for i, day in enumerate(closes.index):
        if i > 0 and weights:
            value *= 1 + sum(w * rets.at[day, s] for s, w in weights.items())
            # Weights drift with prices.
            grown = {s: w * (1 + rets.at[day, s]) for s, w in weights.items()}
            total = sum(grown.values()) + (1 - sum(weights.values()))
            weights = {s: w / total for s, w in grown.items()}
            for s in list(weights):
                if s == "SSO":
                    continue
                highs[s] = max(highs.get(s, 0), closes.at[day, s])
                stop = STOCK_STOP if s in stocks else ETF_STOP
                if closes.at[day, s] <= highs[s] * (1 - stop):
                    value *= 1 - COST * weights[s]
                    del weights[s]
        out.append(value)
        if day in month_ends and i >= 200:
            row = score.loc[day, candidates].dropna().sort_values(ascending=False)
            picks, n_stocks = [], 0
            for s in row.index:
                if s in stocks:
                    if n_stocks >= MAX_STOCKS:
                        continue
                    n_stocks += 1
                picks.append(s)
                if len(picks) == N_PICKS:
                    break
            new = {"SSO": 0.5} | {s: 0.5 / N_PICKS for s in picks}
            turnover = sum(abs(new.get(s, 0) - weights.get(s, 0)) for s in set(new) | set(weights))
            value *= 1 - COST * turnover
            out[-1] = value
            weights = new
            highs = {s: closes.at[day, s] for s in picks}
    return pd.Series(out, index=closes.index)


def _summary(series: pd.Series) -> dict:
    years = (series.index[-1] - series.index[0]).days / 365.25
    dd = float((series / series.cummax() - 1).min())
    return {"end_value": round(float(series.iloc[-1]), 2),
            "yearly_pct": round(((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100, 1),
            "worst_drop_pct": round(dd * 100, 1)}


def run_picker(closes: pd.DataFrame) -> dict:
    closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.5)).ffill()
    etfs = [s for s in CORE_ETFS if s in closes and s not in LEVERAGED]
    stocks = [s for s in BIG_STOCKS if s in closes]
    start = closes.index[200]
    total = (closes.iloc[-1] / closes.loc[start] - 1).reindex(stocks).sort_values(ascending=False)
    top_winners = list(total.index[:6])
    variants = {
        "old: ETFs only": picker(closes, etfs, set()),
        "new: ETFs + big stocks": picker(closes, etfs + stocks, set(stocks)),
        "new, without the 6 biggest winners": picker(closes, etfs + [s for s in stocks if s not in top_winners],
                                                     set(stocks)),
    }
    spy = closes["SPY"] / closes["SPY"].loc[start] * CAPITAL
    variants = {k: v.loc[start:] / v.loc[start] * CAPITAL for k, v in variants.items()}
    out = {k: _summary(v) for k, v in variants.items()}
    out["just hold the S&P 500"] = _summary(spy.loc[start:])
    by_year = {}
    for k, v in list(variants.items()) + [("just hold the S&P 500", spy.loc[start:])]:
        yearly = v.groupby(v.index.year).last() / v.groupby(v.index.year).first() - 1
        by_year[k] = {str(y): round(float(r) * 100, 1) for y, r in yearly.items()}
    return {"start": str(start.date()), "end": str(closes.index[-1].date()), "results": out,
            "by_year_pct": by_year, "biggest_winners_removed": top_winners,
            "stocks_with_data": len(stocks), "etfs_with_data": len(etfs)}


def fetch_daily(years: float, symbols: list[str]) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(years * 365.25))
    cols = {}
    for i in range(0, len(symbols), 40):
        df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols[i:i + 40], timeframe=TimeFrame.Day,
                                                    start=start, end=end, adjustment=Adjustment.ALL,
                                                    feed=DataFeed.SIP)).df
        for sym in df.index.get_level_values("symbol").unique():
            s = df.xs(sym, level="symbol")["close"]
            s.index = s.index.tz_convert("America/New_York").normalize().tz_localize(None)
            cols[sym] = s
        log.info("daily bars: %d/%d symbols", min(i + 40, len(symbols)), len(symbols))
    return pd.DataFrame(cols).sort_index()


def main() -> None:
    from . import backtest

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=9.5)
    ap.add_argument("--watch-months", type=int, default=12)
    ap.add_argument("--out", default="docs/data/stocks_backtest.json")
    args = ap.parse_args()

    symbols = sorted((set(CORE_ETFS) - LEVERAGED) | set(BIG_STOCKS) | {"SSO"})
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
              "picker": run_picker(fetch_daily(args.years, symbols))}
    log.info("picker: %s", json.dumps(report["picker"]["results"]))

    watch_syms = sorted((set(CORE_ETFS) - LEVERAGED) | set(BIG_STOCKS))
    intraday, daily = backtest.fetch(args.watch_months, watch_syms)
    etfs = set(CORE_ETFS) - LEVERAGED
    report["watcher"] = {
        "months": args.watch_months,
        "old: ETFs only": backtest.run(intraday, daily, min_scores=(5,), horizons=(3, 5), universe=etfs),
        "new: ETFs + big stocks": backtest.run(intraday, daily, min_scores=(5,), horizons=(3, 5),
                                               universe=etfs | set(BIG_STOCKS)),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
