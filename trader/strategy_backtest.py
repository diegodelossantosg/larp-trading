"""Compare simple systematic ETF strategies with buying and holding SPY, on daily data since 2016.

    python -m trader.strategy_backtest

Strategies (all long-only, using dividend-adjusted closes, signals at a close and positions held
from the next day, 0.05% cost on every change of position):

- spy_hold:          buy and hold SPY (the benchmark)
- spy_trend:         SPY when SPY is above its 200-day average, otherwise T-bills (BIL)
- sso_trend:         2x S&P 500 (SSO) when SPY is above its 200-day average, otherwise BIL
- sso_hold:          buy and hold SSO (to show what the trend filter adds)
- sector_momentum:   each month, the 3 sector ETFs with the best 12-1 month return, equal weight
- sector_mom_filter: same, but a sector only counts if its 12-1 month return beats BIL; else BIL
- dual_momentum:     each month, if SPY's 12-month return beats BIL: hold the stronger of SPY and
                     EFA (world ex-US); otherwise hold bonds (AGG)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("strategies")

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
OTHERS = ["SPY", "SSO", "BIL", "EFA", "AGG"]
COST = 0.0005


def run_weights(rets: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Daily strategy returns: weights decided at day t's close earn day t+1's returns, minus costs."""
    w = weights.reindex(rets.index).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * rets.fillna(0.0)).sum(axis=1)
    turnover = w.diff().fillna(w).abs().sum(axis=1).shift(1).fillna(0.0)
    return gross - COST * turnover


def stats(r: pd.Series, bench: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    dd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(252)
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    byear = (1 + bench).groupby(bench.index.year).prod() - 1
    return {
        "final_value_of_1000": round(float(1000 * eq.iloc[-1]), 0),
        "total_return_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(float(eq.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "max_drawdown_pct": round(float(dd) * 100, 1),
        "volatility_pct": round(float(vol) * 100, 1),
        "sharpe": round(float(r.mean() * 252 / vol), 2) if vol > 0 else None,
        "years_beating_spy": f"{int((yearly > byear + 1e-9).sum())}/{len(yearly)}",
        "yearly_pct": {str(y): round(float(v) * 100, 1) for y, v in yearly.items()},
    }


def strategies(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rets = px.pct_change()
    idx = px.index
    one = lambda col: pd.DataFrame({col: 1.0}, index=idx)  # noqa: E731
    above = px["SPY"] > px["SPY"].rolling(200).mean()

    def trend(asset):
        w = pd.DataFrame(0.0, index=idx, columns=[asset, "BIL"])
        w.loc[above, asset] = 1.0
        w.loc[~above, "BIL"] = 1.0
        w.loc[px["SPY"].rolling(200).mean().isna()] = np.nan
        return w

    month_ends = px.groupby([idx.year, idx.month]).tail(1).index
    mom = px.shift(21) / px.shift(252) - 1  # 12-1 month momentum
    ret12 = px / px.shift(252) - 1

    def sector(filtered: bool):
        w = pd.DataFrame(np.nan, index=idx, columns=SECTORS + ["BIL"])
        for d in month_ends:
            m = mom.loc[d, SECTORS].dropna()
            if len(m) < 5:
                continue
            row = pd.Series(0.0, index=w.columns)
            for s in m.sort_values(ascending=False).index[:3]:
                if filtered and not m[s] > mom.loc[d, "BIL"]:
                    row["BIL"] += 1 / 3
                else:
                    row[s] += 1 / 3
            w.loc[d] = row
        return w

    def dual():
        w = pd.DataFrame(np.nan, index=idx, columns=["SPY", "EFA", "AGG"])
        for d in month_ends:
            r = ret12.loc[d]
            if r[["SPY", "EFA", "BIL"]].isna().any():
                continue
            row = pd.Series(0.0, index=w.columns)
            if r["SPY"] > r["BIL"]:
                row["SPY" if r["SPY"] >= r["EFA"] else "EFA"] = 1.0
            else:
                row["AGG"] = 1.0
            w.loc[d] = row
        return w

    return {
        "spy_hold": one("SPY"), "spy_trend": trend("SPY"), "sso_trend": trend("SSO"), "sso_hold": one("SSO"),
        "sector_momentum": sector(False), "sector_mom_filter": sector(True), "dual_momentum": dual(),
    }


def evaluate(px: pd.DataFrame) -> dict:
    rets = px.pct_change()
    ws = strategies(px)
    series = {name: run_weights(rets, w) for name, w in ws.items()}
    # Compare everything over the same period: once every strategy has a full signal history.
    start = max(w.dropna(how="all").index[0] for w in ws.values())
    series = {k: v.loc[v.index > start] for k, v in series.items()}
    bench = series["spy_hold"]
    covid = (bench.index >= pd.Timestamp("2020-02-19")) & (bench.index <= pd.Timestamp("2020-04-30"))
    since21 = bench.index >= pd.Timestamp("2021-01-01")
    return {"from": str(bench.index[0].date()), "to": str(bench.index[-1].date()),
            "results": {k: stats(v, bench) for k, v in series.items()},
            # What-if views: the COVID crash and rebound (19 Feb - 30 Apr 2020) removed, and 2021 onwards only.
            "without_covid": {k: stats(v[~covid], bench[~covid]) for k, v in series.items()},
            "since_2021": {k: stats(v[since21], bench[since21]) for k, v in series.items()}}


def fetch() -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    df = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=SECTORS + OTHERS, timeframe=TimeFrame.Day,
        start=datetime(2015, 1, 1, tzinfo=timezone.utc), adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
    px = df["close"].unstack("symbol")
    px.index = pd.DatetimeIndex([t.date() for t in px.index])
    return px.sort_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = evaluate(fetch())
    result["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open("docs/data/strategy_backtest.json", "w") as f:
        json.dump(result, f, indent=1)
    for view in ("results", "without_covid", "since_2021"):
        print(view)
        print(json.dumps({k: {m: v[m] for m in v if m != "yearly_pct"} for k, v in result[view].items()}, indent=1))
    print("from", result["from"], "to", result["to"])


if __name__ == "__main__":
    main()
