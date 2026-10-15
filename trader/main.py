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
from .brain import claude_available, decide, hunt_news, review_buys
from .broker import Broker
from .config import load_config
from .indicators import summarize
from .market import Market
from .risk import Order, apply_limits, drawdown, earnings_blackout, plan_orders
from .universe import BIG_STOCKS, CORE_ETFS, LEVERAGED, NAMES
from . import health as market_health
from . import feeds, scorecard
from . import watch as rules

log = logging.getLogger("trader")
ET = ZoneInfo("America/New_York")
MIN_DOLLAR_VOLUME_M = 5.0  # extra ETFs Claude proposes must trade at least $5M/day
MIN_SMALL_STOCK_DOLLAR_VOLUME_M = 20.0  # smaller companies: at least $20M/day...
MIN_SMALL_STOCK_PRICE = 5.0  # ...and no penny stocks


def _today() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _write_intraday(cfg, broker: Broker, market: Market) -> None:
    """Minute-by-minute value of the agent's money and SPY's price, last 5 sessions, for the chart."""
    import pandas as pd

    try:
        hist = broker.equity_history()
    except Exception as e:  # the chart is nice to have; never break the watcher for it
        log.warning("intraday history failed: %s", e)
        return
    state = journal.read("capital.json", None)
    if not hist or (cfg.capital_usd and state is None):
        return
    try:
        hist = _repair_history(hist, broker.account()["equity"])
    except Exception as e:
        log.warning("account check for the chart failed: %s", e)
    offset = cfg.capital_usd - state["baseline_equity"] if cfg.capital_usd else 0.0
    spy = market.minute_closes("SPY")
    ts = pd.to_datetime([t for t, _ in hist], unit="s", utc=True)
    spy_at = spy.reindex(spy.index.union(ts)).ffill().reindex(ts) if len(spy) else pd.Series(index=ts, dtype=float)
    points = [[t.isoformat(timespec="minutes"), round(e + offset, 2), None if pd.isna(p) else round(float(p), 2)]
              for t, (_, e), p in zip(ts, hist, spy_at.to_numpy())]
    days = sorted({p[0][:10] for p in points})[-5:]
    journal.write("intraday.json", {"points": [p for p in points if p[0][:10] in days]})
    if state is not None and "spy_start" not in state:
        # SPY's price when the capital started, so the chart can show what the same money in the S&P 500 is worth.
        first = next((p for p in points if p[0][:10] >= state.get("started", "") and p[2] is not None), None)
        if first:
            journal.write("capital.json", {**state, "spy_start": first[2]})


def _repair_history(hist: list[tuple[int, float]], equity_now: float) -> list[tuple[int, float]]:
    """Line Alpaca's minute history up with the account's live equity.

    Alpaca's paper-account history has been seen reporting equity $100,000 too high from one day
    on (a step that never happened). When the latest history point is more than 1% off the live
    value, shift every point since the last such step (a >20% jump between two minutes) by the gap.
    """
    if not hist or not equity_now:
        return hist
    gap = equity_now - hist[-1][1]
    if abs(gap) <= 0.01 * equity_now:
        return hist
    start = 0
    for i in range(1, len(hist)):
        if abs(hist[i][1] - hist[i - 1][1]) > 0.2 * hist[i - 1][1]:
            start = i
    return hist[:start] + [(t, e + gap) for t, e in hist[start:]]


def _recent_watch_buys(days: int) -> set[str]:
    """ETFs the watcher bought within the last `days` trading days."""
    import pandas as pd

    cutoff = (pd.Timestamp(_today()) - pd.offsets.BDay(days)).strftime("%Y-%m-%d")
    return {o["symbol"] for d in journal.read("decisions.json", [])
            if d.get("source") in ("watch", "news") and not d.get("dry_run") and d["date"] > cutoff
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
        "spy_start": (journal.read("capital.json", None) or {}).get("spy_start"),
        "idle_asset": cfg.idle_asset,
        "model": cfg.claude_model,
        "equity": round(acct["equity"], 2),
        "cash": round(acct["cash"], 2),
        "performance": _performance(rows),
        "scorecard": scorecard.summary(journal.read("scorecard.json", [])),
        "health": journal.read("health.json", None),
        "positions": sorted(
            ({"symbol": s, **{k: round(v, 4) for k, v in p.items()},
              "name": NAMES.get(s, ""),
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

    universe = sorted(set(CORE_ETFS) | set(BIG_STOCKS) | set(positions) | {cfg.idle_asset})
    bars = market.daily_bars(universe)
    technicals = {s: {"name": NAMES.get(s, ""), **summarize(df)} for s, df in bars.items() if len(df) > 30}

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
    health = market_health.build({s: t for s, t in technicals.items() if s in CORE_ETFS}, bars)  # breadth: ETFs only
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
            "max_weight_per_big_company_stock": cfg.max_stock_weight,
            "max_weight_per_smaller_company_stock": cfg.max_small_stock_weight,
            "max_total_in_smaller_companies": cfg.max_small_stocks_total,
            "max_total_in_single_stocks": cfg.max_stocks_total,
            "no_stock_buys_this_many_trading_days_before_earnings": cfg.earnings_blackout_days,
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
        # Holding keeps Claude's picks as they are, but the risk rules still apply (for example
        # moving spare money into a newly chosen idle asset). Small drifts stay inside the band.
        record["notes"].append("Claude chose to hold its picks; only rule-based adjustments are made.")

    proposed = {p["symbol"].upper(): float(p["weight"]) for p in decision.data["positions"]}
    allowed = set(technicals)
    leveraged = set(LEVERAGED)
    stocks = {s: "big" for s in BIG_STOCKS}
    for s in set(positions) - set(CORE_ETFS) - set(BIG_STOCKS):  # holdings from outside our lists
        ok, lev, kind, _ = broker.vet(s)
        if lev:
            leveraged.add(s)
        if kind in ("big", "small"):
            stocks[s] = kind
    extra = [s for s in proposed if s not in technicals]
    if extra:
        extra_bars = market.daily_bars(extra)
        for s in extra:
            ok, lev, kind, why = broker.vet(s)
            df = extra_bars.get(s)
            if ok and df is not None and len(df) > 30:
                summary = summarize(df)
                small = kind == "small"
                min_vol = MIN_SMALL_STOCK_DOLLAR_VOLUME_M if small else MIN_DOLLAR_VOLUME_M
                if small and summary["price"] < MIN_SMALL_STOCK_PRICE:
                    why = f"price ${summary['price']} is too low (penny stock)"
                elif summary["avg_dollar_vol_m"] >= min_vol:
                    technicals[s] = {"name": "", **summary}
                    allowed.add(s)
                    if lev:
                        leveraged.add(s)
                    if kind in ("big", "small"):
                        stocks[s] = kind
                    continue
                else:
                    why = f"too illiquid (${summary['avg_dollar_vol_m']}M/day)"
            record["notes"].append(f"rejected {s}: {why or 'no price history'}")
    stocks = {s: k for s, k in stocks.items() if s in allowed}

    held_weights = {s: p["market_value"] / (equity or 1) for s, p in positions.items()}
    earnings = {p["symbol"].upper(): p.get("next_earnings", "") for p in decision.data["positions"]}
    proposed, notes = earnings_blackout(proposed, held_weights, set(stocks), earnings, _today(),
                                        cfg.earnings_blackout_days)
    record["notes"] += notes

    targets, notes = apply_limits(
        proposed, allowed=allowed, leveraged=leveraged, idle_asset=cfg.idle_asset,
        max_position_weight=cfg.max_position_weight, max_positions=cfg.max_positions,
        max_leveraged_weight=cfg.max_leveraged_weight, max_idle_weight=cfg.max_idle_weight,
        stocks=stocks, max_stock_weight=cfg.max_stock_weight, max_small_stock_weight=cfg.max_small_stock_weight,
        max_small_stocks_total=cfg.max_small_stocks_total, max_stocks_total=cfg.max_stocks_total,
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
    stocks = set(BIG_STOCKS) | {s for s in set(positions) - set(CORE_ETFS) - set(BIG_STOCKS)
                                if broker.vet(s)[2] in ("big", "small")}

    orders = rules.stop_orders(positions, highs, quotes, cfg.idle_asset, cfg.stop_trail_pct,
                               stocks=stocks, stock_trail_pct=cfg.stop_trail_pct_stock)
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
                                          cfg.max_position_weight, cfg.watch_buy_size,
                                          max_stock_weight=cfg.max_stock_weight,
                                          max_stocks_total=cfg.max_stocks_total, stocks=stocks)
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
        _write_intraday(cfg, broker, market)
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
    _write_intraday(cfg, broker, market)
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
        "candidates": [{"symbol": s, "name": NAMES.get(s, ""), "signals": f"{sc}/6: " + "; ".join(r),
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


NEWS_BUY_SIZE = 0.05  # share of the agent's money per News Hunter buy
NEWS_MAX_BUYS_PER_DAY = 2


def news(cfg, broker: Broker, market: Market, dry_run: bool, force: bool = False) -> int:
    """Hourly News Hunter: Opus reads the latest headlines and may buy what it thinks isn't priced in."""
    if not force and not broker.market_open():
        log.info("market closed; nothing to do")
        return 0
    if cfg.capital_usd and journal.read("capital.json", None) is None:
        log.info("waiting for the first daily run to set up the agent's capital")
        return 0
    if not claude_available(cfg):
        log.info("News Hunter needs Claude; skipping")
        return 0
    positions = broker.positions()
    acct = _account(cfg, broker, positions)
    history = journal.read("equity.json", [])
    if drawdown([r["equity"] for r in history] + [acct["equity"]]) >= cfg.max_drawdown_halt:
        log.error("halted by drawdown limit; News Hunter does nothing")
        return 1
    today = _today()
    state = journal.read("news.json", {})
    used = state.get("buys", {}).get(today, 0)
    if used >= NEWS_MAX_BUYS_PER_DAY:
        log.info("News Hunter already made %d buys today", used)
        return 0
    headlines = market.news(hours=2, limit=50)
    extra, sources = feeds.fetch_all(hours=2)
    log.info("news sources: Alpaca/Benzinga %d, %s", len(headlines), sources)
    headlines += extra
    if not headlines:
        log.info("no fresh headlines")
        return 0

    equity = acct["equity"] or 1
    weights = {s: p["market_value"] / equity for s, p in positions.items()}
    mentioned = sorted({s for h in headlines for s in h["symbols"] if s.isalpha() and len(s) <= 5})[:40]
    quotes = market.quotes(sorted(set(mentioned) | set(BIG_STOCKS) | {"SPY"}))
    spy_move = quotes["SPY"].price / quotes["SPY"].prev_close - 1 if "SPY" in quotes else 0.0
    vs_spy = {s: q.price / q.prev_close - 1 - spy_move for s, q in quotes.items() if s in BIG_STOCKS}
    big_moves = sorted((s for s, m in vs_spy.items() if abs(m) >= 0.02), key=lambda s: vs_spy[s])[:15]
    last = next((d for d in reversed(journal.read("decisions.json", [])) if d.get("market_view")), {})
    context = {
        "now_new_york": datetime.now(ET).strftime("%Y-%m-%d %H:%M"),
        "headlines": headlines,
        "news_sources_status": {"Alpaca (Benzinga)": f"ok ({len(headlines) - len(extra)})", **sources},
        "prices_of_mentioned": {s: {"price": q.price, "change_today_pct": round((q.price / q.prev_close - 1) * 100, 2)}
                                for s, q in quotes.items() if s in mentioned},
        "spy_change_today_pct": round(spy_move * 100, 2),
        "big_company_moves_today": {s: {"name": BIG_STOCKS[s], "vs_spy_today_pct": round(vs_spy[s] * 100, 2)}
                                    for s in big_moves},
        "holdings": {s: round(w, 3) for s, w in weights.items()},
        "todays_market_view": last.get("market_view"),
        "buys_left_today": NEWS_MAX_BUYS_PER_DAY - used,
    }
    data, error = hunt_news(cfg, context)
    record = {"date": today, "time": datetime.now(ET).strftime("%H:%M"), "source": "news", "dry_run": dry_run,
              "orders": [], "notes": []}
    if data is None:
        log.error("News Hunter failed: %s", error)
        return 1
    record["plain_summary"] = data.get("summary", "")
    record["ideas"] = data.get("ideas", [])

    size = NEWS_BUY_SIZE * equity
    stocks_total = sum(w for s, w in weights.items() if s in BIG_STOCKS or (s not in CORE_ETFS and broker.vet(s)[2] in ("big", "small")))
    small_total = sum(w for s, w in weights.items() if s not in CORE_ETFS and s not in BIG_STOCKS and broker.vet(s)[2] == "small")
    buys: list[Order] = []
    for idea in data.get("ideas", [])[: NEWS_MAX_BUYS_PER_DAY - used]:
        sym = idea["symbol"].upper().strip()
        ok, lev, kind, why = broker.vet(sym)
        if ok and lev:
            ok, why = False, "leveraged or inverse fund"
        if ok:
            df = market.daily_bars([sym]).get(sym)
            t = summarize(df) if df is not None and len(df) > 30 else None
            if t is None:
                ok, why = False, "no price history"
            elif kind == "small" and t["price"] < MIN_SMALL_STOCK_PRICE:
                ok, why = False, f"price ${t['price']} is too low (penny stock)"
            elif t["avg_dollar_vol_m"] < (MIN_SMALL_STOCK_DOLLAR_VOLUME_M if kind == "small" else MIN_DOLLAR_VOLUME_M):
                ok, why = False, f"too illiquid (${t['avg_dollar_vol_m']}M/day)"
        if ok and kind in ("big", "small"):
            _, notes = earnings_blackout({sym: weights.get(sym, 0) + NEWS_BUY_SIZE}, weights, {sym},
                                         {sym: idea.get("next_earnings", "")}, today, cfg.earnings_blackout_days)
            if notes:
                ok, why = False, notes[0].replace(f"didn't add to {sym}: ", "")
        cap = {"big": cfg.max_stock_weight, "small": cfg.max_small_stock_weight}.get(kind, cfg.max_position_weight)
        room = cap - weights.get(sym, 0.0)
        if kind in ("big", "small"):
            room = min(room, cfg.max_stocks_total - stocks_total)
        if kind == "small":
            room = min(room, cfg.max_small_stocks_total - small_total)
        amount = min(size, room * equity)
        if ok and amount < cfg.min_order_usd:
            ok, why = False, "already at its size limit"
        if not ok:
            record["notes"].append(f"skipped {sym}: {why}")
            continue
        buys.append(Order(sym, "buy", round(amount, 2), idea["plain_reason"]))
        if kind in ("big", "small"):
            stocks_total += amount / equity
        if kind == "small":
            small_total += amount / equity
    orders = rules.fund_buys(buys, equity=acct["equity"], cash=acct["cash"],
                             idle_value=positions.get(cfg.idle_asset, {}).get("market_value", 0.0),
                             idle_asset=cfg.idle_asset)
    if orders:
        prices = {s: q.price for s, q in quotes.items()}
        for o in orders:
            if o.symbol not in prices:
                q = market.quotes([o.symbol]).get(o.symbol)
                if q:
                    prices[o.symbol] = q.price
        if dry_run or not cfg.trading_enabled:
            record["orders"] = [{"symbol": o.symbol, "side": o.side, "notional": o.notional, "reason": o.reason,
                                 "status": "not sent"} for o in orders]
        else:
            record["orders"] = broker.execute(orders, prices, positions)
            scorecard.record(record["orders"], "news", prices, prices.get("SPY"), today, cfg.idle_asset)
        n = sum(1 for o in record["orders"] if o["side"] == "buy" and o["status"] not in ("error", "skipped", "not sent"))
        journal.write("news.json", {"buys": {today: used + n}})
        positions = broker.positions()
        _write_status(cfg, _account(cfg, broker, positions), positions, history)
    journal.append_decision(record)
    log.info("News Hunter: %s", record["plain_summary"])
    for o in record["orders"]:
        log.info("%s %s $%.2f -> %s (%s)", o["side"], o["symbol"], o["notional"], o["status"], o.get("reason", ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="trader")
    parser.add_argument("command", choices=["trade", "snapshot", "watch", "news", "market-open"])
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
    if args.command == "news":
        return news(cfg, broker, market, args.dry_run, args.force)
    return trade(cfg, broker, market, args.dry_run, args.force)


if __name__ == "__main__":
    sys.exit(main())
