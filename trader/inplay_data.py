"""Download the data for the "stocks in play" day-trading test (see inplay_backtest).

    python -m trader.inplay_data --out research/inplay_raw

For every trading day 2016-2026 and every stock in that year's lab universe (research/lab.npz: the
500 most-traded stocks each January, delisted ones included):
1. daily bars with the high (lab.npz has no high), for the 14-day ATR;
2. the first 5-minute bar (09:30-09:35), for the relative volume of the open;
3. full-day 1-minute bars for the 20 stocks with the highest opening relative volume that day, plus
   20 random members of the universe (the "no selection" comparison).
Each day is saved to its own file, so a stopped download resumes where it left off.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger("inplay_data")
NY = "America/New_York"
TOP, RANDOM, LOOKBACK = 20, 20, 14


def _client():
    from alpaca.data.historical import StockHistoricalDataClient

    from .config import load_config

    cfg = load_config()
    return StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)


def _bars(client, symbols, start, end, minutes: int | None):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = TimeFrame.Day if minutes is None else TimeFrame(minutes, TimeFrameUnit.Minute)
    for attempt in range(4):
        try:
            return client.get_stock_bars(StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=tf, start=start,
                                                          end=end, adjustment=Adjustment.ALL,
                                                          feed=DataFeed.SIP)).df
        except Exception as e:  # network hiccups and rate limits
            log.warning("request failed (%s), retrying", e)
            import time as _t
            _t.sleep(2 ** (attempt + 1))
    return pd.DataFrame()


def ny_dt(day, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, t)).tz_localize(NY)


def _full_day(client, day, symbols) -> dict[str, pd.DataFrame]:
    """1-minute bars 09:30-16:00 for these symbols on this day."""
    full = {}
    if symbols:
        df = _bars(client, symbols, ny_dt(day, time(9, 30)), ny_dt(day, time(16, 0)), 1)
        for sym, g in (df.groupby(level="symbol") if not df.empty else []):
            full[sym] = g.droplevel("symbol")[["open", "high", "low", "close", "volume"]].astype("float32")
    return full


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default="research/lab.npz")
    ap.add_argument("--out", default="research/inplay_raw")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "days"), exist_ok=True)
    z = np.load(args.lab)
    dates = pd.DatetimeIndex(z["dates"])
    symbols = list(z["symbols"])
    etfs = set(z["etfs"])
    years = list(z["years"])
    member = z["member"]
    client = _client()

    daily_path = os.path.join(args.out, "daily.pkl")
    if not os.path.exists(daily_path):
        stocks = [s for s in symbols if s not in etfs]
        parts = []
        for k in range(0, len(stocks), 200):
            df = _bars(client, stocks[k:k + 200], pd.Timestamp("2015-10-01", tz="UTC"), pd.Timestamp.now(tz="UTC")
                       - timedelta(minutes=20), None)
            if not df.empty:
                parts.append(df[["open", "high", "low", "close", "volume"]].astype("float32"))
            log.info("daily: %d/%d", min(k + 200, len(stocks)), len(stocks))
        pd.concat(parts).to_pickle(daily_path)

    first = {}   # day -> {symbol: first-bar volume}, for the relative volume
    rng = random.Random(7)
    for d in dates:
        if d.year not in years:
            continue
        day = d.date()
        path = os.path.join(args.out, "days", f"{day}.pkl")
        uni = [s for s, m in zip(symbols, member[years.index(d.year)]) if m and s not in etfs]
        if os.path.exists(path):
            rec = pd.read_pickle(path)
            first[day] = rec["first"]
            if "bars1" not in rec:
                rec["bars1"] = _full_day(client, day, sorted(set(rec["picks"]) | set(rec["random"])))
                rec.pop("bars", None)
                pd.to_pickle(rec, path)
                log.info("%s: 1-minute bars added", day)
            continue
        fb = {}
        for k in range(0, len(uni), 250):
            df = _bars(client, uni[k:k + 250], ny_dt(day, time(9, 30)), ny_dt(day, time(9, 35)), 5)
            if not df.empty:
                for (sym, _ts), row in df.iterrows():
                    fb[sym] = (float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
                               float(row["volume"]))
        if not fb:
            continue  # holiday or missing day
        vols = pd.Series({s: v[4] for s, v in fb.items()})
        past = [first[p] for p in sorted(first)[-LOOKBACK:]]
        rvol = pd.Series(dtype=float)
        if len(past) == LOOKBACK:
            avg = pd.DataFrame([pd.Series({s: v[4] for s, v in p.items()}) for p in past]).mean(skipna=False)
            rvol = (vols / avg).dropna().sort_values(ascending=False)
        picks = list(rvol.index[:TOP])
        rand = rng.sample(sorted(fb), min(RANDOM, len(fb)))
        want = sorted(set(picks) | set(rand))
        pd.to_pickle({"first": fb, "rvol": rvol.head(60).to_dict(), "picks": picks, "random": rand,
                      "bars1": _full_day(client, day, want)}, path)
        first[day] = fb
        log.info("%s: %d stocks, picks %s", day, len(fb), ",".join(picks[:5]))


if __name__ == "__main__":
    main()
