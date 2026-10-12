"""Market data and news from Alpaca."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .config import Config

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
