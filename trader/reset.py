"""Start over with the agent's money.

    python -m trader.reset data        # empty the dashboard's history (docs/data) for a fresh start
    python -m trader.reset liquidate   # sell every position in the paper account (market must be open)
    python -m trader.reset on-open     # liquidate once, if docs/data/liquidate.json says it is pending

The agent's capital is virtual ($1,000 inside the paper account, `capital.json`). Selling everything
does not change the account's total value, so it never moves the starting point.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import journal, ledger

log = logging.getLogger("trader.reset")
ET = ZoneInfo("America/New_York")


def empty_files(now: str, capital: float, spy: float | None = None) -> dict:
    """What each history file looks like with nothing in it. capital.json is removed on purpose:
    the next daily run writes it again, with the account's value at that moment as the start."""
    return {
        "equity.json": [],
        "decisions.json": [],
        "scorecard.json": [],
        # one starting point, so the chart and the S&P 500 card both start at the capital
        "intraday.json": {"points": [[now, capital, spy]] if spy else []},
        "watch.json": {"highs": {}, "breakouts": {}},
        "news.json": {},
        "sales.json": {"updated": now, "fills_seen": 0, "summary": ledger.summary([]), "rows": []},
        "report_card.json": {"all": {"picks": 0}, "by_trade_type": {}, "by_kind": {}, "recent": [], "updated": now[:10]},
        "status.json": {"updated": now, "mode": "paper", "capital": capital, "spy_start": spy, "equity": capital,
                        "cash": capital, "performance": {"days_tracked": 0}, "positions": []},
    }


def reset_data(capital: float = 1000.0, spy: float | None = None) -> list[str]:
    """`spy` is the S&P 500's price now: the benchmark starts as if the same capital were put in SPY now."""
    now = datetime.now(ET).isoformat(timespec="minutes")
    files = empty_files(now, capital, spy)
    for name, content in files.items():
        journal.write(name, content)
    cap = journal._path("capital.json")
    if cap.exists():
        cap.unlink()
    return sorted(files) + ["capital.json (removed)"]


def liquidate(broker, wait_seconds: int = 120) -> int:
    """Cancel open orders and sell every position. Returns how many positions were left unsold."""
    broker.client.close_all_positions(cancel_orders=True)
    for _ in range(max(1, wait_seconds // 5)):
        left = broker.positions()
        if not left:
            return 0
        time.sleep(5)
    return len(broker.positions())


def on_open(broker, cfg) -> int:
    marker = journal.read("liquidate.json", {})
    if not marker.get("pending"):
        log.info("nothing pending")
        return 0
    if not cfg.paper:
        log.error("refusing to sell everything outside paper trading")
        return 1
    if not broker.market_open():
        log.info("market closed; will try at the next open")
        return 0
    left = liquidate(broker)
    now = datetime.now(ET).isoformat(timespec="minutes")
    if left:
        log.error("%d positions still unsold; will try again", left)
        return 1
    cap = journal._path("capital.json")
    if cap.exists():
        cap.unlink()   # the next daily run starts the $1,000 from the account's value right now (all cash)
    journal.write("liquidate.json", {"pending": False, "done": now})
    log.info("all positions sold")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = (argv or sys.argv[1:] or [""])[0]
    from .broker import Broker
    from .config import load_config
    cfg = load_config()
    if cmd == "data":
        args = argv or sys.argv[1:]
        if journal.read("decisions.json", []) and "--force" not in args:
            log.error("there is history already; this would erase it (add --force if you mean it)")
            return 1
        try:
            from .market import Market
            spy = round(float(Market(cfg).daily_bars(["SPY"], days=10)["SPY"]["close"].iloc[-1]), 2)
        except Exception as e:   # the benchmark falls back to the first minute the watcher records
            log.warning("SPY price unavailable (%s); the S&P line will start at the first live price", e)
            spy = None
        print("\n".join(reset_data(cfg.capital_usd or 1000.0, spy)))
        return 0
    if cmd == "liquidate":
        if not cfg.paper:
            log.error("refusing to sell everything outside paper trading")
            return 1
        return 1 if liquidate(Broker(cfg)) else 0
    if cmd == "on-open":
        return on_open(Broker(cfg), cfg)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
