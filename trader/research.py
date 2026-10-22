"""Helpers for the research loop on research/events.csv.gz (see event_dataset).

Every rule is judged the same way: net of costs (0.1% a round trip, 0.4% when bought in the
pre-market, where prices are thinner), against SPY, year by year, and against random days of the
same companies entered the same way ("edge vs chance"). Rules are designed on `dev` (and shown on
`big`, which is biased upwards); `holdout` is looked at once, at the very end.
"""

from __future__ import annotations

import pandas as pd

COSTS = {"pre8": 0.004}
DEFAULT_COST = 0.001


def load(path: str = "research/events.csv.gz") -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def col(entry: str, hold: int, stop: bool) -> str:
    return f"{entry}_{'s' if stop else 'r'}{hold}"


def evaluate(df: pd.DataFrame, mask, entry: str, hold: int, stop: bool = True, sets=("big", "dev")) -> dict:
    """Net result of a rule per set: trades, avg, years positive, edge vs random days (same set)."""
    c = col(entry, hold, stop)
    cost = COSTS.get(entry, DEFAULT_COST)
    out = {}
    for s in sets:
        ev = df[mask & (df["set"] == s)][["year", c]].dropna()
        ctl = df[(df["kind"] == "control") & (df["set"] == s)][["year", c]].dropna()
        if len(ev) < 10:
            out[s] = {"trades": len(ev)}
            continue
        r = ev[c] - cost
        base = (ctl[c] - cost).mean() if len(ctl) else float("nan")
        by_year = r.groupby(ev["year"]).mean()
        base_year = (ctl[c] - cost).groupby(ctl["year"]).mean()
        edge_year = (by_year - base_year.reindex(by_year.index)).dropna()
        counts = ev.groupby("year").size()
        edge_year = edge_year[counts.reindex(edge_year.index) >= 5]
        out[s] = {"trades": len(r), "avg_pct": round(r.mean() * 100, 2), "win": round((r > 0).mean(), 2),
                  "random_pct": round(base * 100, 2), "edge_pct": round((r.mean() - base) * 100, 2),
                  "t": round(r.mean() / (r.std() / len(r) ** 0.5), 1) if r.std() > 0 else None,
                  "years_edge": f"{int((edge_year > 0).sum())}/{len(edge_year)}",
                  "edge_by_year": {int(y): round(v * 100, 2) for y, v in edge_year.items()}}
    return out


def table(df, rules: dict, entries=("pre8", "open", "h11", "close"), holds=(5, 10, 20), stop=True, sets=("big", "dev")):
    """rules: {name: mask}. One line per rule x entry x hold."""
    rows = []
    for name, m in rules.items():
        for e in entries:
            for h in holds:
                res = evaluate(df, m, e, h, stop, sets)
                row = {"rule": name, "entry": e, "hold": h}
                for s in sets:
                    x = res.get(s, {})
                    row[f"{s}_n"] = x.get("trades")
                    row[f"{s}_avg"] = x.get("avg_pct")
                    row[f"{s}_edge"] = x.get("edge_pct")
                    row[f"{s}_yrs"] = x.get("years_edge")
                rows.append(row)
    return pd.DataFrame(rows)
