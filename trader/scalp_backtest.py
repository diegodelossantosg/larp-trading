"""Do minute-level "scalping" trades make money after costs? Tested on real 1-minute bars.

    python -m trader.scalp_backtest --months 3

Three common retail scalping setups, long only, one position per ETF at a time:

1. breakout: the close breaks above the previous 15-minute high on 2x normal volume.
   Take profit +0.15%, stop -0.10%, exit after 15 minutes at the latest.
2. vwap_reversion: price is 0.30% or more below today's VWAP and the 1-minute RSI is under 25.
   Take profit at VWAP (at most +0.30%), stop -0.20%, exit after 30 minutes at the latest.
3. opening_range: after 09:45, the close breaks above the 09:30-09:45 high (once a day).
   Take profit +0.30%, stop at the range low (at most -0.25%), exit at 15:55 at the latest.

Entries fill at the next bar's open. Inside a bar we can't see whether the high or the low came
first, so when both the target and the stop are touched we assume the stop (the cautious choice).
Costs per side: 0.01% for SPY/QQQ, 0.03% for the other ETFs (half the bid-ask spread plus slippage).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("scalp")

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "GLD", "TLT"]
COST_PER_SIDE = {"SPY": 0.0001, "QQQ": 0.0001}
DEFAULT_COST = 0.0003


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    up = pd.Series(np.clip(delta, 0, None)).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    down = pd.Series(np.clip(-delta, 0, None)).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(down == 0, 100.0, 100 - 100 / (1 + up / down))


def _exit(o, h, l, c, i0, target, stop, last_i):
    """Walk bars from i0 until target/stop/time; returns (exit_price, bars_held, why)."""
    for i in range(i0, last_i + 1):
        hit_stop, hit_target = l[i] <= stop, h[i] >= target
        if hit_stop:  # cautious: stop first if both touched
            return min(stop, o[i]), i - i0, "stop"
        if hit_target:
            return max(target, o[i]), i - i0, "target"
    return c[last_i], last_i - i0, "time"


def simulate_day(day: pd.DataFrame) -> dict[str, list[float]]:
    """Gross returns of each strategy's trades in one ETF-day (regular session minutes)."""
    o, h, l, c, v = (day[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    out = {"breakout": [], "vwap_reversion": [], "opening_range": []}
    if n < 60:
        return out
    pv = np.cumsum(day["vwap"].to_numpy(dtype=float) * v)
    cv = np.cumsum(v)
    vwap = np.where(cv > 0, pv / np.maximum(cv, 1), c)
    r = rsi(c)
    minutes = np.array([t.hour * 60 + t.minute for t in day.index.tz_convert("America/New_York")])
    end = int(np.searchsorted(minutes, 15 * 60 + 55)) - 1
    end = min(max(end, 0), n - 1)

    # 1. breakout
    i, busy_until = 15, -1
    while i < end - 1:
        if i > busy_until:
            prev_high = h[i - 15:i].max()
            avg_vol = v[i - 15:i].mean()
            if c[i] > prev_high and avg_vol > 0 and v[i] >= 2 * avg_vol:
                e = o[i + 1]
                px, held, _ = _exit(o, h, l, c, i + 1, e * 1.0015, e * 0.999, min(i + 15, end))
                out["breakout"].append(px / e - 1)
                busy_until = i + 1 + held
        i += 1

    # 2. vwap reversion
    i, busy_until = 15, -1
    while i < end - 1:
        if i > busy_until and c[i] <= vwap[i] * 0.997 and r[i] < 25:
            e = o[i + 1]
            target = min(vwap[i], e * 1.003)
            if target > e:
                px, held, _ = _exit(o, h, l, c, i + 1, target, e * 0.998, min(i + 30, end))
                out["vwap_reversion"].append(px / e - 1)
                busy_until = i + 1 + held
        i += 1

    # 3. opening range breakout (first 15 minutes)
    k = int(np.searchsorted(minutes, 9 * 60 + 45))
    if 5 <= k < end - 1:
        rh, rl = h[:k].max(), l[:k].min()
        for i in range(k, end - 1):
            if c[i] > rh:
                e = o[i + 1]
                stop = max(rl, e * 0.9975)
                px, _, _ = _exit(o, h, l, c, i + 1, e * 1.003, stop, end)
                out["opening_range"].append(px / e - 1)
                break
    return out


def run(bars: dict[str, pd.DataFrame]) -> dict:
    gross: dict[str, list[float]] = {"breakout": [], "vwap_reversion": [], "opening_range": []}
    net: dict[str, list[float]] = {k: [] for k in gross}
    days = set()
    for sym, df in bars.items():
        cost = 2 * COST_PER_SIDE.get(sym, DEFAULT_COST)
        for d, day in df.groupby(df.index.tz_convert("America/New_York").date):
            days.add(d)
            for strat, rets in simulate_day(day).items():
                gross[strat] += rets
                net[strat] += [x - cost for x in rets]
    n_days = max(len(days), 1)
    report = {}
    for strat in gross:
        g, nt = gross[strat], net[strat]
        if not g:
            report[strat] = {"trades": 0}
            continue
        sd = statistics.stdev(nt) if len(nt) > 2 else 0
        report[strat] = {
            "trades": len(g),
            "trades_per_day": round(len(g) / n_days, 1),
            "win_rate_after_costs": round(sum(x > 0 for x in nt) / len(nt), 3),
            "avg_gross_pct": round(statistics.mean(g) * 100, 4),
            "avg_net_pct": round(statistics.mean(nt) * 100, 4),
            "total_net_pct_sum": round(sum(nt) * 100, 2),
            "t_stat_net": round(statistics.mean(nt) / (sd / len(nt) ** 0.5), 2) if sd > 0 else None,
            "per_1000_usd_per_trade_net": round(1000 * statistics.mean(nt), 3),
        }
    return {"days": n_days, "symbols": sorted(bars), "strategies": report}


def fetch(months: int) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=int(months * 30.5))
    out = {}
    for sym in SYMBOLS:
        df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                                    start=start, end=end, adjustment=Adjustment.ALL,
                                                    feed=DataFeed.SIP)).df
        if df.empty:
            continue
        df = df.xs(sym, level="symbol").sort_index()
        ny = df.index.tz_convert("America/New_York")
        mask = [(t.hour, t.minute) >= (9, 30) and t.hour < 16 for t in ny]
        out[sym] = df[mask]
        log.info("fetched %s: %d minute bars", sym, len(out[sym]))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--out", default="docs/data/scalp_backtest.json")
    args = ap.parse_args()
    result = run(fetch(args.months))
    result["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    result["months"] = args.months
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
