"""Every sale the agent made, with what it earned: the data behind the dashboard's "Sold" page.

Built from Alpaca's own filled orders (exact prices and quantities), using the average-cost method:
each sale is measured against the average price paid for the shares held at that moment. A partial
sale (a "trim") is a closed trade for the part sold; the rest stays open.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .universe import NAMES


def sales(fills: list[dict], decisions: list[dict], spy: pd.Series | None = None) -> list[dict]:
    """fills: Broker.filled_orders(); decisions: decisions.json; spy: daily SPY closes (date index)."""
    by_order = {}
    for d in decisions:
        for o in d.get("orders", []):
            if o.get("order_id"):
                by_order[o["order_id"]] = d.get("source") or "claude"
    held: dict[str, dict] = {}
    out = []
    for f in fills:
        sym, qty, px = f["symbol"], f["qty"], f["price"]
        day = f["time"][:10]
        lot = held.setdefault(sym, {"qty": 0.0, "cost": 0.0, "since": day, "bought_by": by_order.get(f["id"], "claude")})
        if f["side"] == "buy":
            if lot["qty"] <= 1e-9:
                lot.update(qty=0.0, cost=0.0, since=day, bought_by=by_order.get(f["id"], "claude"))
            lot["qty"] += qty
            lot["cost"] += qty * px
            continue
        if lot["qty"] <= 1e-9:
            continue                          # a sale of shares bought before the agent started
        q = min(qty, lot["qty"])
        avg = lot["cost"] / lot["qty"]
        pnl = q * (px - avg)
        row = {
            "order_id": f["id"], "date": day, "symbol": sym, "name": NAMES.get(sym, ""),
            "bought": lot["since"], "bought_by": lot["bought_by"], "sold_by": by_order.get(f["id"], "claude"),
            "days": (datetime.fromisoformat(day) - datetime.fromisoformat(lot["since"])).days,
            "shares": round(q, 4), "avg_buy_price": round(avg, 2), "sell_price": round(px, 2),
            "cost": round(q * avg, 2), "proceeds": round(q * px, 2),
            "pnl": round(pnl, 2), "pnl_pct": round((px / avg - 1) * 100, 2),
            "full": q >= lot["qty"] - 1e-6,
        }
        if spy is not None and len(spy):
            s0 = spy[spy.index <= pd.Timestamp(lot["since"])]
            s1 = spy[spy.index <= pd.Timestamp(day)]
            if len(s0) and len(s1):
                spy_pct = (s1.iloc[-1] / s0.iloc[-1] - 1) * 100
                row["vs_spy_pct"] = round(row["pnl_pct"] - spy_pct, 2)
        lot["cost"] -= q * avg
        lot["qty"] -= q
        out.append(row)
    return sorted(out, key=lambda r: r["date"], reverse=True)


def summary(rows: list[dict]) -> dict:
    if not rows:
        return {"sales": 0}
    wins = [r for r in rows if r["pnl"] > 0]
    losses = [r for r in rows if r["pnl"] <= 0]
    return {"sales": len(rows), "won": len(wins), "lost": len(losses),
            "total_pnl": round(sum(r["pnl"] for r in rows), 2),
            "win_rate": round(len(wins) / len(rows), 2),
            "avg_win_pct": round(sum(r["pnl_pct"] for r in wins) / len(wins), 2) if wins else None,
            "avg_loss_pct": round(sum(r["pnl_pct"] for r in losses) / len(losses), 2) if losses else None,
            "best": max(rows, key=lambda r: r["pnl"])["symbol"], "worst": min(rows, key=lambda r: r["pnl"])["symbol"]}
