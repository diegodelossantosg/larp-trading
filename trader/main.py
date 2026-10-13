"""Entry point.

    python -m trader.main trade      # research with Claude, then rebalance (market hours)
    python -m trader.main snapshot   # after the close: record equity vs SPY for the dashboard
    python -m trader.main watch      # every 10 minutes: trailing stops and signal-based buys (no Claude)

Flags for trade: --dry-run (decide but place no orders), --force (run while the market is closed).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from . import journal
from .brain import claude_available, decide, review_buys
from .broker import Broker
from .config import load_config
from .indicators import summarize
from .market import Market
from .risk import Order, apply_limits, drawdown, plan_orders
from .universe import CORE_ETFS, LEVERAGED
from . import health as market_health
from . import scorecard
from . import watch as rules

log = logging.getLogger("trader")
ET = ZoneInfo("America/New_York")
MIN_DOLLAR_VOLUME_M = 5.0  # extra ETFs Claude proposes must trade at least $5M/day


def _today() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _recent_watch_buys(days: int) -> set[str]:
    """ETFs the watcher bought within the last `days` trading days."""
    import pandas as pd

    cutoff = (pd.Timestamp(_today()) - pd.offsets.BDay(days)).strftime("%Y-%m-%d")
    return {o["symbol"] for d in journal.read("decisions.json", [])
            if d.get("source") == "watch" and not d.get("dry_run") and d["date"] > cutoff
            for o in d.get("orders", []) if o["side"] == "buy" and o.get("status") not in ("error", "skipped")}


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


def _account(cfg, broker: Broker, positions: dict) -> dict:
    """The agent's own money: CAPITAL_USD plus everything it has gained or lost since it started.

    The Alpaca paper account may hold much more (e.g. $100,000); the agent only ever uses its
    capital. Since nothing else trades in the account, its profit equals the account's change
    in equity since the first run.
    """
    real = broker.account()
    if not cfg.capital_usd:
        return real
    state = journal.read("capital.json", None)
    if state is None or state.get("capital") != cfg.capital_usd:
        state = {"capital": cfg.capital_usd, "baseline_equity": real["equity"], "started": _today()}
        journal.write("capital.json", state)
    equity = cfg.capital_usd + (real["equity"] - state["baseline_equity"])
    invested = sum(p["market_value"] for p in positions.values())
    return {"equity": equity, "cash": equity - invested, "buying_power": equity - invested}


def snapshot(cfg, broker: Broker, market: Market) -> None:
    if not broker.is_trading_day(_today()):
        log.info("not a trading day; no snapshot")
        return
    positions = broker.positions()
    acct = _account(cfg, broker, positions)
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
        "capital": cfg.capital_usd,
        "model": cfg.claude_model,
        "equity": round(acct["equity"], 2),
        "cash": round(acct["cash"], 2),
        "performance": _performance(rows),
        "scorecard": scorecard.summary(journal.read("scorecard.json", [])),
        "health": journal.read("health.json", None),
        "positions": sorted(
            ({"symbol": s, **{k: round(v, 4) for k, v in p.items()},
              "name": CORE_ETFS.get(s, ""),
              "weight": round(p["market_value"] / equity, 4)} for s, p in positions.items()),
            key=lambda p: -p["market_value"],
        ),
    })


def trade(cfg, broker: Broker, market: Market, dry_run: bool, force: bool) -> int:
    if not force and not broker.market_open():
        log.info("market closed; nothing to do")
        return 0

    positions = broker.positions()
    acct = _account(cfg, broker, positions)
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
    watcher = [
        {k: d.get(k) for k in ("date", "time", "notes")}
        | {"orders": [(o["side"], o["symbol"], o["notional"], o.get("status")) for o in d.get("orders", [])]}
        for d in journal.read("decisions.json", [])[-30:] if d.get("source") == "watch"
    ][-10:]
    health = market_health.build(technicals, bars)
    journal.write("health.json", health)
    briefing = {
        "date": _today(),
        "market_health": health,
        "account": {"equity": equity, "cash": acct["cash"]},
        "holdings": {s: {"weight": round(p["market_value"] / equity, 4),
                         "unrealized_pl_pct": round(p["unrealized_plpc"] * 100, 2)} for s, p in positions.items()},
        "risk_limits": {
            "idle_asset": cfg.idle_asset,
            "max_weight_in_idle_asset": cfg.max_idle_weight,
            "max_weight_per_etf": cfg.max_position_weight,
            "max_positions": cfg.max_positions,
            "max_total_leveraged_or_inverse": cfg.max_leveraged_weight,
            "max_daily_turnover": cfg.max_daily_turnover,
            "rebalance_band": cfg.rebalance_band,
        },
        "performance_vs_spy": _performance(history),
        "previous_decisions": previous,
        "watcher_actions_recent": watcher,
        "protected_recent_watcher_buys": sorted(_recent_watch_buys(cfg.min_hold_days) & set(positions)),
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
        max_leveraged_weight=cfg.max_leveraged_weight, max_idle_weight=cfg.max_idle_weight,
    )
    record["notes"] += notes
    record["final_targets"] = targets

    current = {s: p["market_value"] for s, p in positions.items()}
    orders, notes = plan_orders(
        targets, current, equity, rebalance_band=cfg.rebalance_band,
        min_order_usd=cfg.min_order_usd, max_daily_turnover=cfg.max_daily_turnover,
        protected=_recent_watch_buys(cfg.min_hold_days) & set(positions),
    )
    record["notes"] += notes
    prices = {s: t["price"] for s, t in technicals.items()}

    if dry_run or not cfg.trading_enabled:
        record["orders"] = [{"symbol": o.symbol, "side": o.side, "notional": o.notional, "status": "not sent"}
                            for o in orders]
        record["notes"].append("dry run: no orders sent" if dry_run else "TRADING_ENABLED=false: no orders sent")
    else:
        record["orders"] = broker.execute(orders, prices, positions)
        scorecard.record(record["orders"], "claude", prices, prices.get("SPY"), _today(), cfg.idle_asset)

    journal.append_decision(record)
    positions = broker.positions()
    scorecard.update(prices, prices.get("SPY"), set(positions), _today())
    _write_status(cfg, _account(cfg, broker, positions), positions, history)
    for o in record["orders"]:
        log.info("%s %s $%.2f -> %s", o["side"], o["symbol"], o["notional"], o["status"])
    return 0


def watch(cfg, broker: Broker, market: Market, dry_run: bool, force: bool = False) -> int:
    """10-minute check: trailing stops and signal-based buys. Rules only, no Claude."""
    if not force and not broker.market_open():
        log.info("market closed; nothing to do")
        return 0
    if cfg.capital_usd and journal.read("capital.json", None) is None:
        log.info("waiting for the first daily run to set up the agent's capital")
        return 0
    positions = broker.positions()
    acct = _account(cfg, broker, positions)
    history = journal.read("equity.json", [])
    if drawdown([r["equity"] for r in history] + [acct["equity"]]) >= cfg.max_drawdown_halt:
        log.error("halted by drawdown limit; watcher does nothing")
        return 1

    today = _today()
    state = journal.read("watch.json", {"highs": {}, "breakouts": {}})
    quotes = market.quotes(sorted(set(CORE_ETFS) | set(positions)))
    highs = rules.update_highs(state.get("highs", {}), positions, quotes)

    orders = rules.stop_orders(positions, highs, quotes, cfg.idle_asset, cfg.stop_trail_pct)
    stopped = {o.symbol for o in orders}
    candidates: list = []
    used = state.get("buys", state.get("breakouts", {})).get(today, 0)
    slots = cfg.watch_max_buys_per_day - used
    if slots > 0:
        equity = acct["equity"] or 1
        weights = {s: p["market_value"] / equity for s, p in positions.items() if s not in stopped}
        mood = (journal.read("health.json", {}) or {}).get("mood")
        min_score = min(6, cfg.watch_min_score + 1) if mood == "stressed" else cfg.watch_min_score
        candidates = rules.buy_candidates(quotes, weights, cfg.idle_asset, min_score,
                                          cfg.max_position_weight, cfg.watch_buy_size)
        orders += rules.buy_orders(
            candidates, equity=acct["equity"],
            cash=acct["cash"] + sum(o.notional for o in orders),
            idle_value=positions.get(cfg.idle_asset, {}).get("market_value", 0.0),
            idle_asset=cfg.idle_asset, held=set(positions) - stopped,
            max_positions=cfg.max_positions, size=cfg.watch_buy_size, slots=slots,
        )

    notes: list[str] = []
    if any(o.side == "buy" for o in orders):
        orders, review_notes = _review(cfg, orders, quotes, positions, acct, candidates)
        notes += review_notes

    new_buys = sum(1 for o in orders if o.side == "buy")
    state = {"highs": highs, "buys": {today: used + new_buys}, "last_check": datetime.now(ET).isoformat(timespec="minutes")}
    if not orders:
        if highs != journal.read("watch.json", {}).get("highs"):
            journal.write("watch.json", state)
        scorecard.update({s: q.price for s, q in quotes.items()}, quotes["SPY"].price if "SPY" in quotes else None, set(positions), _today())
        _write_status(cfg, acct, positions, history)  # live value for the dashboard
        log.info("watch: no action (%d quotes checked)", len(quotes))
        return 0

    prices = {s: q.price for s, q in quotes.items()}
    if dry_run or not cfg.trading_enabled:
        results = [{"symbol": o.symbol, "side": o.side, "notional": o.notional, "reason": o.reason,
                    "status": "not sent"} for o in orders]
    else:
        results = broker.execute(orders, prices, positions)
        scorecard.record(results, "watch", prices, prices.get("SPY"), today, cfg.idle_asset)
    for r in results:
        if r["side"] == "buy" and r["status"] not in ("error", "skipped", "not sent"):
            highs[r["symbol"]] = prices.get(r["symbol"], 0.0)
    journal.write("watch.json", state)
    journal.append_decision({
        "date": today, "time": datetime.now(ET).strftime("%H:%M"), "source": "watch", "dry_run": dry_run,
        "orders": results, "notes": notes + [f"{o.side} {o.symbol}: {o.reason}" for o in orders],
    })
    positions = broker.positions()
    scorecard.update(prices, prices.get("SPY"), set(positions), _today())
    _write_status(cfg, _account(cfg, broker, positions), positions, history)
    for r in results:
        log.info("%s %s $%.2f -> %s (%s)", r["side"], r["symbol"], r["notional"], r["status"], r["reason"])
    return 0


def _review(cfg, orders, quotes, positions, acct, candidates):
    """Keep only the watcher buys Claude approves; drop the SPY sale that funded rejected ones.

    Protective sells (trailing stops) never wait for Claude.
    """
    buys = [o for o in orders if o.side == "buy"]
    if not claude_available(cfg):
        if not cfg.watch_require_claude:
            return orders, []
        kept = [o for o in orders if o.side == "sell" and o.reason.startswith("trailing stop")]
        return kept, [f"skipped {len(buys)} buy idea(s): Claude isn't available to approve them "
                      "(add the CLAUDE_CODE_OAUTH_TOKEN secret)"]
    equity = acct["equity"] or 1
    last = next((d for d in reversed(journal.read("decisions.json", [])) if d.get("market_view")), {})
    context = {
        "candidates": [{"symbol": s, "name": CORE_ETFS.get(s, ""), "signals": f"{sc}/6: " + "; ".join(r),
                        "price": quotes[s].price, "change_today_pct": round((quotes[s].price / quotes[s].prev_close - 1) * 100, 2)}
                       for s, sc, r in candidates if any(o.symbol == s for o in buys)],
        "holdings": {s: round(p["market_value"] / equity, 3) for s, p in positions.items()},
        "todays_market_view": last.get("market_view"),
        "spy_change_today_pct": round((quotes["SPY"].price / quotes["SPY"].prev_close - 1) * 100, 2) if "SPY" in quotes else None,
    }
    verdicts, error = review_buys(cfg, context)
    if verdicts is None:
        kept = [o for o in orders if o.side == "sell" and o.reason.startswith("trailing stop")]
        return kept, [f"skipped buys: {error}"]
    notes, approved = [], []
    for o in buys:
        ok, why = verdicts.get(o.symbol, (False, "no verdict"))
        notes.append(f"Claude {'approved' if ok else 'rejected'} {o.symbol}: {why}")
        if ok:
            approved.append(Order(o.symbol, o.side, o.notional, o.reason + " · Claude: " + why))
    stops = [o for o in orders if o.side == "sell" and o.reason.startswith("trailing stop")]
    funding = [o for o in orders if o.side == "sell" and not o.reason.startswith("trailing stop")]
    if funding and approved:
        share = sum(o.notional for o in approved) / sum(o.notional for o in buys)
        funding = [Order(f.symbol, "sell", round(f.notional * share, 2), f.reason) for f in funding]
    else:
        funding = []
    return stops + funding + approved, notes


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="trader")
    parser.add_argument("command", choices=["trade", "snapshot", "watch", "market-open"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    broker, market = Broker(cfg), Market(cfg)
    log.info("mode: %s", "PAPER" if cfg.paper else "LIVE MONEY")
    if args.command == "snapshot":
        snapshot(cfg, broker, market)
        return 0
    if args.command == "market-open":  # exit code 0 while the market is open, 1 when closed
        return 0 if broker.market_open() else 1
    if args.command == "watch":
        return watch(cfg, broker, market, args.dry_run, args.force)
    return trade(cfg, broker, market, args.dry_run, args.force)


if __name__ == "__main__":
    sys.exit(main())
