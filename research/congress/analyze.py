"""Does buying what members of the House buy, once it is public, beat the S&P 500?

    python -m research.congress.analyze

Entry: the next trading day's open after the report was FILED (when anyone can see it). Exit: the close N trading days
later. Result: the stock's return minus SPY's over the same days, minus 0.1% costs. Controls: random days of the same
stocks in the same year (does the member's pick add anything over just owning those stocks?). Design years 2016-2020,
validation 2021 on. Also shown: the same trades from the TRADE date (what the members themselves got; we can't do that).
"""
import glob

import numpy as np
import pandas as pd

COST = 0.001
HOLDS = (5, 20, 60, 120)


def load_trades():
    t = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob("research/congress/trades_*.csv.gz"))])
    t = t.dropna(subset=["ticker", "filed", "date"])
    t["ticker"] = t.ticker.str.replace(".", "-", regex=False).str.upper()
    t["amount_low"] = pd.to_numeric(t.amount_low, errors="coerce")
    t = t[t.asset_type.isin(["ST"]) | t.asset_type.isna()]
    t["days_late"] = (pd.to_datetime(t.filed) - pd.to_datetime(t.date)).dt.days
    return t[(t.days_late >= 0) & (t.days_late <= 400)]


def returns(D, sym, i0, h):
    """(stock - SPY) from day i0's open to day i0+h-1's close, or None."""
    j, S = D["col"].get(sym), D["col"]["SPY"]
    i1 = i0 + h - 1
    if j is None or i1 >= len(D["dates"]):
        return None
    o, c, so, sc = D["o"][i0, j], D["c"][i1, j], D["o"][i0, S], D["c"][i1, S]
    if not all(np.isfinite([o, c, so, sc])) or o <= 0:
        return None
    return (c / o - 1) - (sc / so - 1) - COST


def tstat(x, groups):
    """t-stat with returns averaged per month first (trades in the same month are not independent)."""
    s = pd.Series(x).groupby(np.asarray(groups)).mean()
    return float(s.mean() / (s.std() / np.sqrt(len(s)))) if len(s) > 3 and s.std() > 0 else float("nan")


def main():
    z = np.load("research/replay/daily.npz")
    dates = pd.to_datetime(z["dates"])
    D = {"dates": dates, "col": {s: k for k, s in enumerate(z["symbols"])}, "o": z["open"].astype(float),
         "c": z["close"].astype(float)}
    t = load_trades()
    t = t[t.ticker.isin(D["col"])]
    idx = np.searchsorted(dates.values, pd.to_datetime(t.filed).values, side="right")   # next trading day after filing
    t["i_pub"] = idx
    t["i_trade"] = np.searchsorted(dates.values, pd.to_datetime(t.date).values, side="right")
    t["year"] = pd.to_datetime(t.filed).dt.year
    t["month"] = pd.to_datetime(t.filed).dt.to_period("M").astype(str)
    buys = t[t.type == "P"].drop_duplicates(["ticker", "filed", "member"])
    sells = t[t.type.str.startswith("S", na=False)].drop_duplicates(["ticker", "filed", "member"])
    # clusters: 2+ different members buying the same stock within 30 days (counted at the filing that makes it 2+)
    b = buys.sort_values("filed")
    clust = []
    for (tick), g in b.groupby("ticker"):
        f = pd.to_datetime(g.filed).values
        for k in range(len(g)):
            w = g[(pd.to_datetime(g.filed) <= f[k]) & (pd.to_datetime(g.filed) > f[k] - np.timedelta64(30, "D"))]
            clust.append(w.member.nunique() >= 2)
    b["cluster"] = clust
    rng = np.random.default_rng(0)
    groups = {"all buys": b, "buys $50k+": b[b.amount_low >= 50001], "buys $250k+": b[b.amount_low >= 250001],
              "buys reported within 7 days": b[b.days_late <= 7], "2+ members buying (30 days)": b[b.cluster],
              "sales (stock after a member sells)": sells}
    print(f"trades loaded {len(t)}, buys {len(b)}, sales {len(sells)}, years {sorted(t.year.unique())}")
    rows = []
    for name, g in groups.items():
        for h in HOLDS:
            g0 = g[g.i_pub < len(dates)]
            r = [returns(D, s, int(i), h) for s, i in zip(g0.ticker, g0.i_pub)]
            ok = np.array([x is not None for x in r], bool)
            gg = g0[ok]
            r = np.array([x for x in r if x is not None])
            # control: random day of the same stock in the same year
            ctl = []
            for s, y in zip(gg.ticker, gg.year):
                days = np.where(dates.year == y)[0]
                for _ in range(3):
                    x = returns(D, s, int(rng.choice(days)), h)
                    if x is not None:
                        ctl.append(x)
            trade_r = [returns(D, s, int(i), h) for s, i in zip(gg.ticker, gg.i_trade)]
            trade_r = np.array([x for x in trade_r if x is not None])
            des, val = gg.year <= 2020, gg.year >= 2021
            yr = pd.Series(r).groupby(gg.year.values).mean()
            rows.append({"group": name, "hold": h, "n": len(r), "avg_vs_spy_pct": r.mean() * 100,
                         "edge_vs_random_days_pct": (r.mean() - np.mean(ctl)) * 100 if ctl else np.nan,
                         "t_month": tstat(r, gg.month.values),
                         "design_2016_20_pct": r[des.values].mean() * 100 if des.any() else np.nan,
                         "valid_2021_26_pct": r[val.values].mean() * 100 if val.any() else np.nan,
                         "years_ahead": f"{(yr > 0).sum()}/{len(yr)}",
                         "from_trade_date_pct": trade_r.mean() * 100 if len(trade_r) else np.nan})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))
    out.to_csv("research/congress/results.csv", index=False)


if __name__ == "__main__":
    main()
