"""Claude's report card: how each kind of daily pick has done against the S&P 500.

Every buy from Claude's daily decision is a "pick" with the trade type Claude gave it (news catalyst,
earnings, trend, macro theme, defensive hedge, rebound, other). After the close each day the snapshot
measures each pick's return vs SPY after 5, 20 and 60 trading days (or so far, if younger), and
groups the results by trade type and by ETF vs company stock. The summary goes back into Claude's
next briefing, so it can lean towards what works and away from what doesn't, and onto the dashboard.
"""

from __future__ import annotations

import statistics

import pandas as pd

from .universe import CORE_ETFS

HORIZONS = (5, 20, 60)
TRADE_TYPES = ["news_catalyst", "earnings", "trend_momentum", "macro_theme", "defensive_hedge", "rebound", "other"]


def picks(decisions: list[dict], idle_asset: str) -> list[dict]:
    """Claude's executed buys (not the watcher's, not the News Hunter's), one per symbol per day."""
    out, seen = [], set()
    for d in decisions:
        if d.get("source") or d.get("dry_run") or not d.get("market_view"):
            continue
        types = {p["symbol"].upper(): p.get("trade_type", "untagged") for p in d.get("positions", [])}
        reasons = {p["symbol"].upper(): p.get("plain_reason") or p.get("thesis", "")[:100] for p in d.get("positions", [])}
        for o in d.get("orders", []):
            sym = o["symbol"]
            if o["side"] != "buy" or sym == idle_asset or o.get("status") in ("error", "skipped", "not sent"):
                continue
            if (sym, d["date"]) in seen:
                continue
            seen.add((sym, d["date"]))
            out.append({"date": d["date"], "symbol": sym, "trade_type": types.get(sym, "untagged"),
                        "kind": "ETF" if sym in CORE_ETFS else "stock", "reason": reasons.get(sym, "")})
    return out


def score(picks_: list[dict], closes: dict[str, pd.Series]) -> list[dict]:
    """Adds vs-SPY results after each horizon (None if not reached yet) and so far."""
    spy = closes.get("SPY")
    out = []
    for p in picks_:
        px = closes.get(p["symbol"])
        if px is None or spy is None:
            continue
        dates = [d.strftime("%Y-%m-%d") for d in px.index]
        if p["date"] not in dates:
            continue
        i = dates.index(p["date"])
        s_dates = [d.strftime("%Y-%m-%d") for d in spy.index]
        if p["date"] not in s_dates:
            continue
        j = s_dates.index(p["date"])
        row = dict(p)
        for h in HORIZONS:
            if i + h < len(px) and j + h < len(spy):
                row[f"vs_spy_{h}d_pct"] = round(((px.iloc[i + h] / px.iloc[i]) - (spy.iloc[j + h] / spy.iloc[j])) * 100, 2)
            else:
                row[f"vs_spy_{h}d_pct"] = None
        row["vs_spy_so_far_pct"] = round(((px.iloc[-1] / px.iloc[i]) - (spy.iloc[-1] / spy.iloc[j])) * 100, 2)
        row["days_held_so_far"] = len(px) - 1 - i
        out.append(row)
    return out


def summarize(scored: list[dict]) -> dict:
    def group(rows):
        g = {"picks": len(rows)}
        for key in [f"vs_spy_{h}d_pct" for h in HORIZONS] + ["vs_spy_so_far_pct"]:
            xs = [r[key] for r in rows if r.get(key) is not None]
            if xs:
                g[key.replace("_pct", "_avg_pct")] = round(statistics.mean(xs), 2)
                g[key.replace("_pct", "_beat_rate")] = round(sum(x > 0 for x in xs) / len(xs), 2)
        return g

    by_type = {t: group([r for r in scored if r["trade_type"] == t])
               for t in sorted({r["trade_type"] for r in scored})}
    by_kind = {k: group([r for r in scored if r["kind"] == k]) for k in ("ETF", "stock")}
    return {"all": group(scored), "by_trade_type": by_type, "by_kind": by_kind,
            "recent": sorted(scored, key=lambda r: r["date"], reverse=True)[:12]}


def build(decisions: list[dict], idle_asset: str, daily_bars) -> dict:
    """daily_bars(symbols, days) -> {symbol: DataFrame with a 'close' column} (Market.daily_bars)."""
    ps = picks(decisions, idle_asset)
    if not ps:
        return {"all": {"picks": 0}, "by_trade_type": {}, "by_kind": {}, "recent": []}
    first = min(p["date"] for p in ps)
    days = (pd.Timestamp.now() - pd.Timestamp(first)).days + 130
    bars = daily_bars(sorted({p["symbol"] for p in ps} | {"SPY"}), days=days)
    closes = {s: df["close"] for s, df in bars.items()}
    for s, c in closes.items():
        idx = pd.DatetimeIndex(c.index)
        closes[s] = pd.Series(c.to_numpy(), index=idx.tz_convert("America/New_York").normalize() if idx.tz else idx)
    return summarize(score(ps, closes))
