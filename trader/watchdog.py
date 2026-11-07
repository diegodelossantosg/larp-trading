"""Checks that each agent actually ran today, and says which ones to start again.

    python -m trader.watchdog      # prints a JSON list such as ["trade", "news"]

The workflow `.github/workflows/watchdog.yml` runs this every 10 minutes during market hours and
starts whatever is missing. Starting twice is harmless: the daily decision refuses to run a second
time on the same day (see `trade_done`), and the watcher and News Hunter queue behind a running one.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MAX_FAILED_TRIES = 3          # failed daily decisions we retry before giving up for the day
TRADE_DUE = time(9, 50)       # the decision is planned for 9:36; this much grace before we start it
WATCH_DUE = time(9, 40)       # the watcher should be running a few minutes after the open
NEWS_DUE = time(10, 15)
NEWS_MAX_GAP_MIN = 90         # the News Hunter runs hourly; older than this = missed
ACTIVE = ("queued", "in_progress", "waiting", "pending", "requested")


def trade_done(decisions: list[dict], today: str) -> bool:
    """True when today's daily decision has been made (or can't be: halted, or failed too often)."""
    tries = 0
    for d in decisions:
        if d.get("date") != today or d.get("source") or d.get("dry_run"):
            continue
        if "market_view" in d or str(d.get("error", "")).startswith("HALTED"):
            return True
        tries += 1
    return tries >= MAX_FAILED_TRIES


def missing_agents(now: datetime, market_open: bool, decisions: list[dict],
                   watch_active: bool, news_age_min: float | None) -> list[str]:
    """Which agents should be running by now but aren't. Nothing is missing while the market is closed."""
    if not market_open:
        return []
    now = now.astimezone(ET)
    t, out = now.time(), []
    if t >= TRADE_DUE and not trade_done(decisions, now.strftime("%Y-%m-%d")):
        out.append("trade")
    if t >= WATCH_DUE and not watch_active:
        out.append("watch")
    if t >= NEWS_DUE and (news_age_min is None or news_age_min > NEWS_MAX_GAP_MIN):
        out.append("news")
    return out


def _runs(workflow: str) -> list[dict]:
    out = subprocess.run(["gh", "run", "list", "--workflow", workflow, "--limit", "10",
                          "--json", "status,createdAt"], capture_output=True, text=True, check=True).stdout
    return json.loads(out or "[]")


def watch_active(runs: list[dict]) -> bool:
    return any(r["status"] in ACTIVE for r in runs)


def news_age_min(runs: list[dict], now: datetime) -> float | None:
    """Minutes since the newest News Hunter run was started (None if there is none)."""
    ages = [(now - datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00"))).total_seconds() / 60
            for r in runs]
    return min(ages) if ages else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from . import journal
    from .broker import Broker
    from .config import load_config

    now = datetime.now(ET)
    is_open = Broker(load_config()).market_open()
    missing = missing_agents(now, is_open, journal.read("decisions.json", []),
                             watch_active(_runs("watch.yml")), news_age_min(_runs("hunter.yml"), now))
    logging.info("market open: %s; missing: %s", is_open, missing or "nothing")
    print(json.dumps(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
