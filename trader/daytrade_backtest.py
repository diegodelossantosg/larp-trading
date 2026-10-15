"""Would same-day trades (buy during the day, always sold by the close) have made money after costs?

    python -m trader.daytrade_backtest --years 3

Tested on real 5-minute bars, long only, every position closed at the 16:00 close:

1. orb30: after 10:00, price breaks above the 09:30-10:00 high -> buy. Stop at the range low.
2. orb30_trend: same, but only on days SPY closed yesterday above its 20-day average.
3. morning_momentum: up 0.5% or more from yesterday's close at 10:00 -> buy at 10:00. Stop -1%.
4. last_half_hour: up from yesterday's close at 10:00 -> buy at 15:30 (the "first half hour
   predicts the last half hour" effect from academic studies).
5. dip_buy: down 1% or more at 10:00 while the ETF is above its 20-day average -> buy at 10:00. Stop -1.5%.
baseline: buy every ETF at 10:00 and sell at the close, no signal. A strategy only has an edge if
it clearly beats this.

Entries fill at the next bar's open; when a bar touches the stop it fills at the stop (or the open if
it gapped through). Costs per side: 0.01% for SPY/QQQ, 0.03% for most ETFs, 0.05% for the thinner ones.

The "$200 pot" compounds each strategy's trades: every day the pot is split evenly over that day's
trades (at most 4), and compared with simply holding $200 of SPY over the same dates.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("daytrade")

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "SSO", "QLD", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
           "SMH", "XBI", "GLD", "TLT"]
COST_PER_SIDE = {"SPY": 0.0001, "QQQ": 0.0001, "XBI": 0.0005, "SMH": 0.0005}
DEFAULT_COST = 0.0003
STRATEGIES = ["orb30", "orb30_trend", "morning_momentum", "last_half_hour", "dip_buy", "baseline"]
POT, MAX_TRADES_PER_DAY = 200.0, 4


def _run_to_close(o, l, c, i0, stop, last):
    """Hold from bar i0's open until the stop or the close of bar `last`. Returns the exit price."""
    for i in range(i0, last + 1):
        if l[i] <= stop:
            return min(stop, o[i])
    return c[last]


def simulate_day(day: pd.DataFrame, prev_close: float, sma20: float, spy_trend_up: bool) -> dict[str, float]:
    """Gross return of each strategy that traded this ETF on this day (5-minute regular-session bars)."""
    o, h, l, c = (day[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    ny = day.index.tz_convert("America/New_York")
    minutes = np.array([t.hour * 60 + t.minute for t in ny])
    k = int(np.searchsorted(minutes, 10 * 60))        # first bar at/after 10:00
    k_late = int(np.searchsorted(minutes, 15 * 60 + 30))
    last = len(c) - 1
    out: dict[str, float] = {}
    if k < 3 or k >= last or minutes[last] < 15 * 60 + 55 or prev_close is None or pd.isna(prev_close):
        return out  # half days and gappy data are skipped
    entry = o[k]
    move = c[k - 1] / prev_close - 1                  # yesterday's close -> 10:00

    out["baseline"] = c[last] / entry - 1
    rh, rl = h[:k].max(), l[:k].min()
    for i in range(k, last):
        if c[i] > rh:
            e = o[i + 1]
            r = _run_to_close(o, l, c, i + 1, min(rl, e * 0.999), last) / e - 1
            out["orb30"] = r
            if spy_trend_up:
                out["orb30_trend"] = r
            break
    if move >= 0.005:
        out["morning_momentum"] = _run_to_close(o, l, c, k, entry * 0.99, last) / entry - 1
    if move > 0 and k_late < last:
        out["last_half_hour"] = c[last] / o[k_late] - 1
    if move <= -0.01 and sma20 and prev_close > sma20:
        out["dip_buy"] = _run_to_close(o, l, c, k, entry * 0.985, last) / entry - 1
    return out


def run(bars: dict[str, pd.DataFrame]) -> dict:
    """bars: 5-minute regular-session bars per symbol (must include SPY for the trend filter)."""
    daily = {s: df.groupby(df.index.tz_convert("America/New_York").date)["close"].last() for s, df in bars.items()}
    spy = daily.get("SPY")
    spy_up = (spy > spy.rolling(20).mean()).shift(1, fill_value=False) if spy is not None else None

    by_day: dict = {}                                  # date -> strategy -> [net returns]
    gross = {s: [] for s in STRATEGIES}
    net = {s: [] for s in STRATEGIES}
    for sym, df in bars.items():
        cost = 2 * COST_PER_SIDE.get(sym, DEFAULT_COST)
        closes = daily[sym]
        prev = closes.shift(1)
        sma = closes.rolling(20).mean().shift(1)
        for d, day in df.groupby(df.index.tz_convert("America/New_York").date):
            trend = bool(spy_up.get(d, False)) if spy_up is not None else False
            for strat, g in simulate_day(day, prev.get(d), sma.get(d), trend).items():
                gross[strat].append(g)
                net[strat].append(g - cost)
                by_day.setdefault(d, {}).setdefault(strat, []).append(g - cost)

    days = sorted(by_day)
    years = max(len(days) / 252, 1e-9)
    report = {}
    for strat in STRATEGIES:
        g, nt = gross[strat], net[strat]
        if not nt:
            report[strat] = {"trades": 0}
            continue
        pot = POT
        for d in days:
            rets = by_day[d].get(strat, [])[:MAX_TRADES_PER_DAY]
            if rets:
                pot *= 1 + statistics.mean(rets)
        sd = statistics.stdev(nt) if len(nt) > 2 else 0
        report[strat] = {
            "trades": len(nt),
            "trades_per_day": round(len(nt) / max(len(days), 1), 2),
            "win_rate_after_costs": round(sum(x > 0 for x in nt) / len(nt), 3),
            "avg_gross_pct": round(statistics.mean(g) * 100, 4),
            "avg_net_pct": round(statistics.mean(nt) * 100, 4),
            "t_stat_net": round(statistics.mean(nt) / (sd / len(nt) ** 0.5), 2) if sd > 0 else None,
            "pot_end": round(pot, 2),
            "pot_yearly_pct": round(((pot / POT) ** (1 / years) - 1) * 100, 1),
        }
    spy_hold = None
    if spy is not None and len(days) > 1:
        s = spy[spy.index.isin(days)]
        spy_hold = round(POT * s.iloc[-1] / s.iloc[0], 2)
    return {"days": len(days), "first_day": str(days[0]) if days else None, "last_day": str(days[-1]) if days else None,
            "symbols": sorted(bars), "pot_start": POT, "spy_hold_end": spy_hold, "strategies": report}


def fetch(years: float) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(years * 365))
    out = {}
    for sym in SYMBOLS:
        df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                                                    start=start, end=end, adjustment=Adjustment.ALL,
                                                    feed=DataFeed.SIP)).df
        if df.empty:
            continue
        df = df.xs(sym, level="symbol").sort_index()
        ny = df.index.tz_convert("America/New_York")
        out[sym] = df[[(t.hour, t.minute) >= (9, 30) and t.hour < 16 for t in ny]]
        log.info("fetched %s: %d bars", sym, len(out[sym]))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--out", default="docs/data/daytrade_backtest.json")
    args = ap.parse_args()
    result = run(fetch(args.years))
    result["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
