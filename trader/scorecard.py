"""Who is earning: every buy, its source (Claude's daily pick or the watcher) and how it did vs SPY.

Each buy is tracked from its price at purchase until the position is sold, and compared with
what the same money would have done in SPY over the same time.
"""

from __future__ import annotations

from . import journal

FILE = "scorecard.json"


def record(results: list[dict], source: str, prices: dict[str, float], spy_price: float | None,
           date: str, idle_asset: str = "SPY") -> None:
    if not spy_price:
        return
    rows = journal.read(FILE, [])
    for r in results:
        sym = r["symbol"]
        if r["side"] != "buy" or sym == idle_asset or r.get("status") in ("error", "skipped", "not sent"):
            continue
        if sym not in prices:
            continue
        rows.append({"date": date, "source": source, "symbol": sym, "amount": r["notional"],
                     "entry": prices[sym], "spy_entry": spy_price, "last": prices[sym],
                     "spy_last": spy_price, "open": True})
    journal.write(FILE, rows)


def update(prices: dict[str, float], spy_price: float | None, held: set[str], date: str | None = None) -> None:
    """Refresh open buys with current prices; close the ones whose ETF is no longer held."""
    rows = journal.read(FILE, [])
    if not rows or not spy_price:
        return
    for row in rows:
        if not row["open"] or row["symbol"] not in prices:
            continue
        row["last"], row["spy_last"] = prices[row["symbol"]], spy_price
        if row["symbol"] not in held:
            row["open"] = False
            row["closed_date"] = date
    journal.write(FILE, rows)


def summary(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for source in ("claude", "watch"):
        mine = [r for r in rows if r["source"] == source]
        if not mine:
            continue
        rets = [r["last"] / r["entry"] - 1 for r in mine]
        excess = [r["last"] / r["entry"] - r["spy_last"] / r["spy_entry"] for r in mine]
        out[source] = {
            "buys": len(mine),
            "avg_return_pct": round(sum(rets) / len(rets) * 100, 2),
            "avg_vs_spy_pct": round(sum(excess) / len(excess) * 100, 2),
            "beat_spy": sum(1 for e in excess if e > 0),
            "dollars": round(sum(r["amount"] * ret for r, ret in zip(mine, rets)), 2),
            "dollars_vs_spy": round(sum(r["amount"] * e for r, e in zip(mine, excess)), 2),
        }
    return out
