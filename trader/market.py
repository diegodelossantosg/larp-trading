"""Market data and news from Alpaca."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .config import Config
from .indicators import atr_pct, candlestick_patterns

log = logging.getLogger(__name__)


class Market:
    def __init__(self, cfg: Config):
        self.bars_client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
        self.news_client = NewsClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)

    def daily_bars(self, symbols: list[str], days: int = 400) -> dict[str, pd.DataFrame]:
        """Daily OHLCV per symbol, oldest first.

        Uses the consolidated (SIP) feed ending 20 minutes ago, which free accounts
        may query; falls back to the IEX-only feed if that is refused.
        """
        now = datetime.now(timezone.utc)
        base = dict(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=now - timedelta(days=days),
            adjustment=Adjustment.ALL,
        )
        try:
            barset = self.bars_client.get_stock_bars(
                StockBarsRequest(**base, end=now - timedelta(minutes=20), feed=DataFeed.SIP)
            )
        except APIError as e:
            log.warning("SIP bars refused (%s); falling back to IEX feed", e)
            barset = self.bars_client.get_stock_bars(StockBarsRequest(**base, feed=DataFeed.IEX))
        df = barset.df
        out: dict[str, pd.DataFrame] = {}
        if df.empty:
            return out
        for sym in df.index.get_level_values("symbol").unique():
            out[sym] = df.xs(sym, level="symbol")[["open", "high", "low", "close", "volume"]].sort_index()
        return out

    def news(self, hours: int = 36, symbols: list[str] | None = None, limit: int = 50) -> list[dict]:
        req = NewsRequest(
            start=datetime.now(timezone.utc) - timedelta(hours=hours),
            symbols=",".join(symbols) if symbols else None,
            limit=limit,
            sort="desc",
        )
        try:
            articles = self.news_client.get_news(req).data.get("news", [])
        except APIError as e:
            log.warning("news request failed: %s", e)
            return []
        return [
            {
                "time": a.created_at.strftime("%Y-%m-%d %H:%M UTC"),
                "headline": a.headline,
                "summary": (a.summary or "")[:300],
                "symbols": a.symbols[:6],
                "source": a.source,
            }
            for a in articles
        ]

    def quotes(self, symbols: list[str]):
        """Live price plus the daily-trend and intraday-candle inputs the watcher scores."""
        from .watch import Quote

        snaps = self.bars_client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        )
        today = datetime.now(timezone.utc).date()
        daily = self.daily_bars(symbols, days=90)
        intraday = self.intraday_bars(symbols)
        out = {}
        for sym, snap in snaps.items():
            df = daily.get(sym)
            if snap is None or snap.latest_trade is None or snap.previous_daily_bar is None or df is None:
                continue
            past = df[[ts.date() < today for ts in df.index]]
            if len(past) < 50:
                continue
            closes = past["close"]
            q = Quote(
                price=float(snap.latest_trade.price),
                prev_close=float(snap.previous_daily_bar.close),
                high_20d=float(closes.tail(20).max()),
                atr_pct=atr_pct(past),
                sma20=float(closes.tail(20).mean()),
                sma50=float(closes.tail(50).mean()),
            )
            bars = intraday.get(sym)
            if bars is not None and len(bars) >= 2:
                done = bars.iloc[:-1]  # the last bar is still forming
                q.bars_today = len(done)
                vol = bars["volume"].sum()
                q.vwap = float((bars["vwap"] * bars["volume"]).sum() / vol) if vol else 0.0
                q.candles = candlestick_patterns(done) if len(done) >= 3 else []
                avg = done["volume"].mean()
                q.volume_ratio = float(done["volume"].iloc[-1] / avg) if avg else 0.0
            out[sym] = q
        return out

    def intraday_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Today's 15-minute bars (IEX feed, real time), oldest first."""
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            df = self.bars_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbols, timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start, feed=DataFeed.IEX,
            )).df
        except APIError as e:
            log.warning("intraday bars failed: %s", e)
            return {}
        if df.empty:
            return {}
        return {sym: df.xs(sym, level="symbol").sort_index()
                for sym in df.index.get_level_values("symbol").unique()}
