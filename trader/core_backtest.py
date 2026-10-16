"""How much money should sit in SSO between trades, and should that shrink when the market falls?

    python -m trader.core_backtest --years 9

Portfolio = a "core" (SSO, the S&P 500 x2, or plain SPY) + an active trading part. The active part
is a stand-in for our traders: every month, the 4 strongest-trending of the ~100 most-traded
companies as of 2018 (crashed and delisted ones included), each with a 20% trailing stop, 0.1%
costs per trade. Tested since 2017:
  - core size: 0%, 20%, 30%, 40%, 50%, 60% (the rest goes to the active part)
  - core rule: always on, or cut in half / to zero (into cash) while the S&P 500 is below its
    200-day average (a common "the market is falling" signal), checked daily.
Judged on yearly return, worst drop, worst calendar year, and how many years it beat the S&P 500.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("core")

COST = 0.001
STOP = 0.20
N_PICKS = 4


def active_sleeve(closes: pd.DataFrame, lows: pd.DataFrame, highs: pd.DataFrame) -> pd.Series:
    """Daily return of the monthly trend-picks part, fully invested, with trailing stops."""
    rets = closes.pct_change()
    score = (closes / closes.shift(126) - 1) / (rets.rolling(126).std() * np.sqrt(252))
    score = score.where(closes > closes.rolling(200).mean())
    month_end = set(closes.groupby(closes.index.to_period("M")).tail(1).index)
    out, held, peak = [], {}, {}
    for t, day in enumerate(closes.index):
        r = 0.0
        if held and t > 0:
            for s in list(held):
                lo, hi, c, prev = lows.at[day, s], highs.at[day, s], closes.at[day, s], closes.iloc[t - 1][s]
                if np.isnan(c) or np.isnan(prev):   # stopped trading (e.g. delisted): treat as sold at last price
                    held.pop(s)
                    continue
                stop_px = peak[s] * (1 - STOP)
                if not np.isnan(lo) and lo <= stop_px:
                    r += (min(stop_px, prev) / prev - 1 - COST) / N_PICKS   # sold at the stop price
                    held.pop(s)
                    continue
                r += (c / prev - 1) / N_PICKS
                peak[s] = max(peak[s], hi if not np.isnan(hi) else c)
        if day in month_end and t >= 200:
            picks = list(score.loc[day].dropna().sort_values(ascending=False).index[:N_PICKS])
            changed = len(set(picks) ^ set(held))
            r -= COST * changed / N_PICKS
            held = {s: True for s in picks}
            peak = {s: closes.at[day, s] for s in picks}
        out.append(r)
    return pd.Series(out, index=closes.index)


def portfolio(core_r: pd.Series, act_r: pd.Series, spy: pd.Series, weight: float, rule: str) -> pd.Series:
    below = (spy < spy.rolling(200).mean()).shift(1, fill_value=False)
    w = pd.Series(weight, index=core_r.index)
    if rule == "halve when market falls":
        w = w.where(~below, weight / 2)
    elif rule == "zero when market falls":
        w = w.where(~below, 0.0)
    switch_cost = w.diff().abs().fillna(0) * COST
    return w * core_r + (1 - weight) * act_r - switch_cost   # money taken out of the core waits in cash


def stats(r: pd.Series, spy_r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    yearly = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1)
    spy_y = spy_r.groupby(spy_r.index.year).apply(lambda g: (1 + g).prod() - 1)
    return {"yearly_pct": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 1),
            "worst_drop_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
            "worst_year_pct": round(float(yearly.min()) * 100, 1),
            "years_beating_spy": f"{int((yearly > spy_y).sum())}/{len(yearly)}",
            "growth_of_1000": round(1000 * float(eq.iloc[-1])),
            "by_year_pct": {str(y): round(float(v) * 100, 1) for y, v in yearly.items()}}


def run(daily: dict[str, pd.DataFrame], companies: list[str]) -> dict:
    closes = pd.DataFrame({s: daily[s]["close"] for s in companies if s in daily}).sort_index()
    idx = daily["SPY"].index
    closes = closes.reindex(idx)
    lows = pd.DataFrame({s: daily[s]["low"] for s in closes}).reindex(idx)
    highs = pd.DataFrame({s: daily[s]["high"] for s in closes}).reindex(idx)
    act = active_sleeve(closes, lows, highs)
    spy = daily["SPY"]["close"]
    spy_r, sso_r = spy.pct_change().fillna(0), daily["SSO"]["close"].reindex(idx).pct_change().fillna(0)
    start = idx[260]                          # after the warm-up for trends and averages
    rows = []
    for core, w, rule in itertools.product(("SSO", "SPY"), (0.0, 0.2, 0.3, 0.4, 0.5, 0.6),
                                           ("always", "halve when market falls", "zero when market falls")):
        if w == 0 and (rule != "always" or core == "SPY"):   # no core: only one version
            continue
        r = portfolio(sso_r if core == "SSO" else spy_r, act, spy, w, rule)
        rows.append({"core": core if w else "none", "core_pct": int(w * 100), "rule": rule,
                     **stats(r.loc[start:], spy_r.loc[start:])})
    rows.sort(key=lambda x: -x["yearly_pct"])
    return {"start": str(start.date()), "end": str(idx[-1].date()), "companies": len(closes.columns),
            "just_spy": stats(spy_r.loc[start:], spy_r.loc[start:]),
            "just_sso": stats(sso_r.loc[start:], spy_r.loc[start:]),
            "active_part_alone": stats(act.loc[start:], spy_r.loc[start:]),
            "all_combinations": rows}


def main() -> None:
    from .audit_backtest import universe_2018
    from .exit_backtest import fetch_daily_ohlc

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=9)
    ap.add_argument("--out", default="docs/data/core_backtest.json")
    args = ap.parse_args()
    companies = universe_2018()
    daily = fetch_daily_ohlc(sorted(set(companies) | {"SPY", "SSO"}), args.years)
    report = run(daily, companies)
    report["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "all_combinations"}, indent=1)[:4000])
    for r in report["all_combinations"]:
        print({k: v for k, v in r.items() if k != "by_year_pct"})


if __name__ == "__main__":
    main()
