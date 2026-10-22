"""Download the "strategy lab" data once: every US stock 2016-2026, crashed and delisted ones included.

    python -m trader.lab_data

Daily bars for all US-listed common stocks (active and inactive), plus the main ETFs and their 2x
versions. Each January, the 500 most-traded stocks *at that time* (by average dollar volume in the
first 20 trading days) form that year's universe, so strategies only ever pick from what a trader
could have seen then. Saved to research/lab.npz (open, close, low, volume as float32, dates x
symbols), with each symbol's group: A (design), B (validation), C (final check), by a fixed hash.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("lab")

ETFS = ["SPY", "SSO", "QQQ", "QLD", "IWM", "UWM", "DIA", "DDM", "MDY", "TLT", "IEF", "SHY", "GLD",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH", "MTUM", "USMV", "QUAL"]
TOP = 500


def group_of(symbol: str) -> str:
    return "ABC"[int(hashlib.sha1(symbol.encode()).hexdigest(), 16) % 3]


def all_stocks() -> list[str]:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    from .config import load_config
    from .universe import looks_like_fund

    cfg = load_config()
    trading = TradingClient(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=True, raw_data=True)
    assets = []
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        assets += trading.get_all_assets(GetAssetsRequest(status=status, asset_class=AssetClass.US_EQUITY))
    return sorted({a["symbol"] for a in assets
                   if str(a.get("exchange", "")).upper() in ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
                   and a["symbol"].isalpha() and len(a["symbol"]) <= 5 and not looks_like_fund(a.get("name") or "")})


def fetch(symbols: list[str], start, end, batch: int = 200) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    out = {}
    for k in range(0, len(symbols), batch):
        try:
            df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols[k:k + batch], timeframe=TimeFrame.Day,
                                                        start=start, end=end, adjustment=Adjustment.ALL,
                                                        feed=DataFeed.SIP)).df
        except Exception as e:
            log.warning("batch %d failed: %s", k, e)
            continue
        if df.empty:
            continue
        df = df[["open", "close", "low", "volume"]].astype("float32")
        for sym, d in df.groupby(level="symbol"):
            d = d.droplevel("symbol")
            d.index = d.index.tz_convert("America/New_York").normalize().tz_localize(None)
            out[sym] = d
        log.info("bars: %d/%d symbols", min(k + batch, len(symbols)), len(symbols))
    return out


def yearly_universe(bars: dict[str, pd.DataFrame], years: range) -> dict[int, list[str]]:
    uni = {}
    for y in years:
        dv = {}
        for sym, d in bars.items():
            jan = d.loc[f"{y}-01-01":f"{y}-02-15"].head(20)
            if len(jan) >= 15 and float(jan["close"].iloc[-1]) >= 5:
                dv[sym] = float((jan["close"] * jan["volume"]).mean())
        uni[y] = sorted(dv, key=lambda s: -dv[s])[:TOP]
    return uni


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-06-01")
    ap.add_argument("--out", default="research/lab.npz")
    args = ap.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    stocks = all_stocks()
    log.info("candidate stocks: %d", len(stocks))
    bars = fetch(stocks, start, end)
    years = range(2016, end.year + 1)
    uni = yearly_universe(bars, years)
    keep = sorted({s for v in uni.values() for s in v})
    bars = {s: bars[s] for s in keep}
    bars.update(fetch(ETFS, start, end))
    dates = sorted({d for b in bars.values() for d in b.index})
    idx = pd.DatetimeIndex(dates)
    syms = sorted(bars)
    arrays = {}
    for field in ("open", "close", "low", "volume"):
        m = np.full((len(idx), len(syms)), np.nan, dtype="float32")
        for j, s in enumerate(syms):
            m[:, j] = bars[s][field].reindex(idx).to_numpy("float32")
        arrays[field] = m
    member = np.zeros((len(years), len(syms)), dtype=bool)
    pos = {s: j for j, s in enumerate(syms)}
    for i, y in enumerate(years):
        for s in uni[y]:
            member[i, pos[s]] = True
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, dates=np.array([d.strftime("%Y-%m-%d") for d in idx]), symbols=np.array(syms),
                        groups=np.array([group_of(s) for s in syms]), years=np.array(list(years)), member=member,
                        etfs=np.array(ETFS), **arrays)
    log.info("saved %d dates x %d symbols (%d ever in a yearly top %d)", len(idx), len(syms), len(keep), TOP)


if __name__ == "__main__":
    main()
