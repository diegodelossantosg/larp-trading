"""Corporate actions (splits, mergers, spin-offs, name changes...) for what we hold, from Alpaca.

Checked once a day (cached in data/corp_actions.json). The watcher uses splits to rescale a
position's high on the split day, so a 4-for-1 split isn't sold as a 75% crash (QUAL, 2026-11-05);
the News Hunter and the daily decision see every upcoming event for our holdings.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from . import journal

log = logging.getLogger(__name__)

SPLITS = {"forward_splits", "reverse_splits", "unit_splits"}


def parse(raw: dict) -> list[dict]:
    """Flatten Alpaca's {"corporate_actions": {"forward_splits": [...], ...}} into simple rows."""
    out = []
    for kind, items in (raw.get("corporate_actions") or {}).items():
        for it in items or []:
            syms = {it.get(k) for k in ("symbol", "target_symbol", "acquirer_symbol", "old_symbol", "new_symbol",
                                        "source_symbol") if it.get(k)}
            row = {"type": kind.rstrip("s"), "symbols": sorted(syms),
                   "ex_date": it.get("ex_date") or it.get("effective_date") or it.get("process_date")}
            if kind in SPLITS and it.get("old_rate") and it.get("new_rate"):
                row["ratio"] = float(it["new_rate"]) / float(it["old_rate"])   # 4.0 = 4 new shares per old
            out.append(row)
    return out


def fetch(cfg, symbols: list[str], today: str, back: int = 3, ahead: int = 30) -> list[dict]:
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.requests import CorporateActionsRequest

    client = CorporateActionsClient(cfg.alpaca_api_key, cfg.alpaca_secret_key, raw_data=True)
    d = date.fromisoformat(today)
    raw = client.get_corporate_actions(CorporateActionsRequest(
        symbols=symbols, start=d - timedelta(days=back), end=d + timedelta(days=ahead)))
    return parse(raw if isinstance(raw, dict) else {})


def for_holdings(cfg, symbols: list[str], today: str) -> list[dict]:
    """Today's list for these symbols, fetched at most once a day (and again if holdings changed)."""
    symbols = sorted(set(symbols))
    state = journal.read("corp_actions.json", {})
    if state.get("date") == today and state.get("symbols") == symbols:
        return state.get("actions", [])
    try:
        actions = fetch(cfg, symbols, today) if symbols else []
    except Exception as e:   # never block trading on this; the watcher also has a price-based split guard
        log.warning("corporate actions unavailable: %s", e)
        return state.get("actions", []) if state.get("date") == today else []
    journal.write("corp_actions.json", {"date": today, "symbols": symbols, "actions": actions})
    return actions


def splits_today(actions: list[dict], today: str) -> dict[str, float]:
    """{symbol: ratio} for splits whose ex-date is today."""
    return {s: a["ratio"] for a in actions if a.get("ratio") and a.get("ex_date") == today for s in a["symbols"]}
