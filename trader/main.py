"""Entry point.

    python -m trader.main trade      # research with Claude, then rebalance (market hours)
    python -m trader.main snapshot   # after the close: record equity vs SPY for the dashboard

Flags for trade: --dry-run (decide but place no orders), --force (run while the market is closed).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from . import journal
from .brain import decide
from .broker import Broker
from .config import load_config
from .indicators import summarize
from .market import Market
from .risk import apply_limits, drawdown, plan_orders
from .universe import CORE_ETFS, LEVERAGED

log = logging.getLogger("trader")
ET = ZoneInfo("America/New_York")
MIN_DOLLAR_VOLUME_M = 5.0  # extra ETFs Claude proposes must trade at least $5M/day


def _today() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _performance(rows: list[dict]) -> dict:
    if len(rows) < 2:
        return {"days_tracked": len(rows)}
    first, last = rows[0], rows[-1]
    agent = last["equity"] / first["equity"] - 1
    spy = last["spy_close"] / first["spy_close"] - 1
    return {
        "since": first["date"],
        "days_tracked": len(rows),
        "agent_return_pct": round(agent * 100, 2),
        "spy_return_pct": round(spy * 100, 2),
        "excess_vs_spy_pct": round((agent - spy) * 100, 2),
    }


def snapshot(cfg, broker: Broker, market: Market) -> None:
    if not broker.is_trading_day(_today()):
        log.info("not a trading day; no snapshot")
        return
    acct = broker.account()
    positions = broker.positions()
    spy = market.daily_bars(["SPY"], days=10)["SPY"]
    rows = journal.upsert_equity({
        "date": _today(),
        "equity": round(acct["equity"], 2),
        "cash": round(acct["cash"], 2),
        "spy_close": round(float(spy["close"].iloc[-1]), 2),
    })
    _write_status(cfg, acct, positions, rows)
    log.info("snapshot: equity %.2f, %d days tracked", acct["equity"], len(rows))


def _write_status(cfg, acct: dict, positions: dict, rows: list[dict]) -> None:
    equity = acct["equity"] or 1
    journal.write("status.json", {
        "updated": datetime.now(ET).isoformat(timespec="minutes"),
        "mode": "paper" if cfg.paper else "LIVE",
        "model": cfg.claude_model,
        "equity": round(acct["equity"], 2),
        "cash": round(acct["cash"], 2),
        "performance": _performance(rows),
        "positions": sorted(
            ({"symbol": s, **{k: round(v, 4) for k, v in p.items()},
              "weight": round(p["market_value"] / equity, 4)} for s, p in positions.items()),
            key=lambda p: -p["market_value"],
        ),
    })


def trade(cfg, broker: Broker, market: Market, dry_run: bool, force: bool) -> int:
    if not force and not broker.market_open():
        log.info("market closed; nothing to do")
        return 0

    acct = broker.account()
    positions = broker.positions()
    equity = acct["equity"]
    history = journal.read("equity.json", [])
    record: dict = {"date": _today(), "time": datetime.now(ET).strftime("%H:%M"), "equity": equity,
                    "dry_run": dry_run, "orders": [], "notes": []}

    dd = drawdown([r["equity"] for r in history] + [equity])
    if dd >= cfg.max_drawdown_halt:
        record["error"] = f"HALTED: drawdown {dd:.1%} >= limit {cfg.max_drawdown_halt:.0%}. Review before resuming."
        journal.append_decision(record)
        log.error(record["error"])
        return 1

    universe = sorted(set(CORE_ETFS) | set(positions) | {cfg.idle_asset})
    bars = market.daily_bars(universe)
    technicals = {s: {"name": CORE_ETFS.get(s, ""), **summarize(df)} for s, df in bars.items() if len(df) > 30}

    previous = [
        {k: d.get(k) for k in ("date", "market_view", "confidence", "trade_today")}
        | {"positions": [(p["symbol"], p["weight"]) for p in d.get("positions", [])]}
        for d in journal.read("decisions.json", [])[-5:] if "market_view" in d
    ]
    briefing = {
        "date": _today(),
        "account": {"equity": equity, "cash": acct["cash"]},
        "holdings": {s: {"weight": round(p["market_value"] / equity, 4),
                         "unrealized_pl_pct": round(p["unrealized_plpc"] * 100, 2)} for s, p in positions.items()},
        "risk_limits": {
            "idle_asset": cfg.idle_asset,
            "max_weight_per_etf": cfg.max_position_weight,
            "max_positions": cfg.max_positions,
            "max_total_leveraged_or_inverse": cfg.max_leveraged_weight,
            "max_daily_turnover": cfg.max_daily_turnover,
            "rebalance_band": cfg.rebalance_band,
        },
        "performance_vs_spy": _performance(history),
        "previous_decisions": previous,
        "headlines": market.news(hours=36, limit=50),
        "holding_headlines": market.news(hours=72, symbols=list(positions), limit=20) if positions else [],
        "technicals": technicals,
    }

    decision = decide(cfg, briefing)
    record["ai_cost_usd"] = decision.usage.cost_usd()
    record["usage"] = vars(decision.usage)
    record["sources"] = decision.sources[:30]
    if decision.data is None:
        record["error"] = decision.error
        journal.append_decision(record)
        log.error("no decision: %s", decision.error)
        return 1
    record.update(decision.data)

    if not decision.data["trade_today"]:
        record["notes"].append("Claude chose to hold; no orders.")
        journal.append_decision(record)
        _write_status(cfg, acct, positions, history)
        return 0

    proposed = {p["symbol"].upper(): float(p["weight"]) for p in decision.data["positions"]}
    allowed = set(technicals)
    leveraged = set(LEVERAGED)
    extra = [s for s in proposed if s not in technicals]
    if extra:
        extra_bars = market.daily_bars(extra)
        for s in extra:
            ok, lev, why = broker.vet_etf(s)
            df = extra_bars.get(s)
            if ok and df is not None and len(df) > 30:
                summary = summarize(df)
                if summary["avg_dollar_vol_m"] >= MIN_DOLLAR_VOLUME_M:
                    technicals[s] = summary
                    allowed.add(s)
                    if lev:
                        leveraged.add(s)
                    continue
                why = f"too illiquid (${summary['avg_dollar_vol_m']}M/day)"
            record["notes"].append(f"rejected {s}: {why or 'no price history'}")
    for s in allowed - set(CORE_ETFS) - leveraged:  # held ETFs from outside the core list
        if broker.vet_etf(s)[1]:
            leveraged.add(s)

    targets, notes = apply_limits(
        proposed, allowed=allowed, leveraged=leveraged, idle_asset=cfg.idle_asset,
        max_position_weight=cfg.max_position_weight, max_positions=cfg.max_positions,
        max_leveraged_weight=cfg.max_leveraged_weight,
    )
    record["notes"] += notes
    record["final_targets"] = targets

    current = {s: p["market_value"] for s, p in positions.items()}
    orders, notes = plan_orders(
        targets, current, equity, rebalance_band=cfg.rebalance_band,
        min_order_usd=cfg.min_order_usd, max_daily_turnover=cfg.max_daily_turnover,
    )
    record["notes"] += notes
    prices = {s: t["price"] for s, t in technicals.items()}

    if dry_run or not cfg.trading_enabled:
        record["orders"] = [{"symbol": o.symbol, "side": o.side, "notional": o.notional, "status": "not sent"}
                            for o in orders]
        record["notes"].append("dry run: no orders sent" if dry_run else "TRADING_ENABLED=false: no orders sent")
    else:
        record["orders"] = broker.execute(orders, prices, positions)

    journal.append_decision(record)
    _write_status(cfg, broker.account(), broker.positions(), history)
    for o in record["orders"]:
        log.info("%s %s $%.2f -> %s", o["side"], o["symbol"], o["notional"], o["status"])
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="trader")
    parser.add_argument("command", choices=["trade", "snapshot"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    broker, market = Broker(cfg), Market(cfg)
    log.info("mode: %s", "PAPER" if cfg.paper else "LIVE MONEY")
    if args.command == "snapshot":
        snapshot(cfg, broker, market)
        return 0
    return trade(cfg, broker, market, args.dry_run, args.force)


if __name__ == "__main__":
    sys.exit(main())
