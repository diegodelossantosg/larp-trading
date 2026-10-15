"""Account state and order execution on Alpaca."""

from __future__ import annotations

import logging
import math
import time

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, TimeInForce
from alpaca.trading.requests import GetCalendarRequest, GetPortfolioHistoryRequest, MarketOrderRequest

from .config import Config
from .risk import Order
from .universe import BIG_STOCKS, CORE_ETFS, is_leveraged, looks_like_fund

log = logging.getLogger(__name__)


class Broker:
    def __init__(self, cfg: Config):
        self.client = TradingClient(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=cfg.paper)
        self._assets: dict[str, object] = {}

    def market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def is_trading_day(self, day: str) -> bool:
        return bool(self.client.get_calendar(GetCalendarRequest(start=day, end=day)))

    def equity_history(self) -> list[tuple[int, float]]:
        """(unix seconds, account equity) for each regular-hours minute of the last week."""
        h = self.client.get_portfolio_history(GetPortfolioHistoryRequest(
            period="1W", timeframe="1Min", intraday_reporting="market_hours"))
        return [(t, float(e)) for t, e in zip(h.timestamp or [], h.equity or []) if e]

    def account(self) -> dict:
        a = self.client.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash), "buying_power": float(a.buying_power)}

    def positions(self) -> dict[str, dict]:
        return {
            p.symbol: {
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in self.client.get_all_positions()
        }

    def asset(self, symbol: str):
        if symbol not in self._assets:
            try:
                self._assets[symbol] = self.client.get_asset(symbol)
            except APIError:
                self._assets[symbol] = None
        return self._assets[symbol]

    def vet(self, symbol: str) -> tuple[bool, bool, str, str]:
        """(ok, leveraged, kind, reason) for any ETF or stock Claude wants to trade.

        kind is "etf", "big" (a big, well-known company) or "small" (any other company).
        """
        a = self.asset(symbol)
        if a is None:
            return False, False, "", "unknown symbol"
        if a.status != AssetStatus.ACTIVE or not a.tradable or a.asset_class != AssetClass.US_EQUITY:
            return False, False, "", "not active/tradable"
        if symbol in CORE_ETFS or looks_like_fund(a.name or ""):
            return True, is_leveraged(symbol, a.name or ""), "etf", ""
        if str(getattr(a.exchange, "value", a.exchange)).upper() in ("OTC", ""):
            return False, False, "", "only trades over the counter"
        return True, False, "big" if symbol in BIG_STOCKS else "small", ""

    def execute(self, orders: list[Order], prices: dict[str, float], positions: dict[str, dict]) -> list[dict]:
        """Submit sells, wait for them to settle into buying power, then buys."""
        results = []
        sells = [o for o in orders if o.side == "sell"]
        buys = [o for o in orders if o.side == "buy"]
        for o in sells:
            results.append(self._submit(o, prices, positions))
        if sells and buys:
            time.sleep(10)
        for o in buys:
            results.append(self._submit(o, prices, positions))
        return results

    def _submit(self, o: Order, prices: dict[str, float], positions: dict[str, dict]) -> dict:
        side = OrderSide.BUY if o.side == "buy" else OrderSide.SELL
        asset = self.asset(o.symbol)
        held = positions.get(o.symbol, {})
        record = {"symbol": o.symbol, "side": o.side, "notional": o.notional, "reason": o.reason}
        try:
            if o.side == "sell" and held and o.notional >= held["market_value"] * 0.98:
                # Full exit: close by quantity so no dust is left behind.
                order = self.client.close_position(o.symbol)
            elif asset is not None and asset.fractionable:
                order = self.client.submit_order(
                    MarketOrderRequest(symbol=o.symbol, notional=round(o.notional, 2), side=side,
                                       time_in_force=TimeInForce.DAY)
                )
            else:
                qty = math.floor(o.notional / prices[o.symbol])
                if qty < 1:
                    return {**record, "status": "skipped", "error": "less than one whole share"}
                order = self.client.submit_order(
                    MarketOrderRequest(symbol=o.symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
                )
            return {**record, "status": str(order.status.value), "order_id": str(order.id)}
        except (APIError, KeyError) as e:
            log.error("order %s %s failed: %s", o.side, o.symbol, e)
            return {**record, "status": "error", "error": str(e)}
