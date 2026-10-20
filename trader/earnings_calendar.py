"""Which companies report earnings in the next few trading days, from Nasdaq's public earnings calendar.

The earnings rule only needs to know whether a stock reports inside its short blackout window, so we
fetch just those few days. If any day can't be fetched we return None and the caller falls back to
the date Claude gave (an unknown date then still blocks the buy).
"""

from __future__ import annotations

import json
import logging
import urllib.request

import numpy as np

log = logging.getLogger(__name__)

URL = "https://api.nasdaq.com/api/calendar/earnings?date={day}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; larp-trading/1.0; paper-trading research)",
           "Accept": "application/json", "Accept-Encoding": "identity"}


def parse(raw: bytes) -> set[str]:
    """Tickers in one day's calendar response (a day with no reports has no rows)."""
    rows = ((json.loads(raw).get("data") or {}).get("rows")) or []
    return {r["symbol"].strip().upper() for r in rows if r.get("symbol")}


def fetch_day(day: str, timeout: float = 10) -> set[str]:
    req = urllib.request.Request(URL.format(day=day), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse(resp.read())


def upcoming(today: str, days: int, fetch=fetch_day) -> dict[str, str] | None:
    """{symbol: report date} for every company reporting from today through `days` trading days ahead.

    None if the calendar couldn't be read, so nothing is wrongly treated as "not reporting".
    """
    out: dict[str, str] = {}
    for i in range(days + 1):
        day = str(np.busday_offset(today, i, roll="forward"))
        try:
            symbols = fetch(day)
        except Exception as e:  # a missing calendar must never crash a run
            log.info("earnings calendar unavailable for %s: %s", day, e)
            return None
        for s in symbols:
            out.setdefault(s, day)
    return out
