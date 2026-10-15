"""Pairs trading ("don't trade stocks, trade their relationship"), tested honestly.

    python -m trader.pairs_backtest --years 10

Twelve pairs of near-twin companies (Coke/Pepsi, Visa/Mastercard, Exxon/Chevron, ...). Three ways:

A. The video's rule: every R days, buy whichever of the two fell behind over the last L days and
   short the one that ran ahead (half the money each). Also its mirror (buy the leader, short the
   laggard), to see which side really wins.
B. The classic version: watch the price ratio; when it is unusually stretched (z-score beyond a
   threshold over the last W days) buy the cheap one and short the rich one, close when it's back.
C. Buy-only (what a small account can do): hold only the laggard, compared with holding both.

Costs: 0.05% per trade per side, and 0.5% a year to borrow shares for shorting. All 12 pairs are
traded together with equal money. Settings are picked on 2016-2020 and judged on 2021-2026.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("pairs")

PAIRS = [("KO", "PEP"), ("V", "MA"), ("XOM", "CVX"), ("HD", "LOW"), ("JPM", "BAC"), ("MCD", "YUM"),
         ("PG", "CL"), ("UPS", "FDX"), ("WMT", "TGT"), ("LMT", "NOC"), ("MRK", "PFE"), ("GS", "MS")]
COST = 0.0005          # per unit of weight traded
BORROW = 0.005 / 252   # daily cost per unit of weight shorted
SPLIT = "2021-01-01"


def _run(ra: np.ndarray, rb: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> np.ndarray:
    """Daily net returns for weights decided at each day's close (applied to the next day)."""
    wa_prev, wb_prev = np.r_[0, wa[:-1]], np.r_[0, wb[:-1]]
    gross = wa_prev * ra + wb_prev * rb
    turnover = np.abs(wa - wa_prev) + np.abs(wb - wb_prev)
    borrow = (np.clip(-wa_prev, 0, None) + np.clip(-wb_prev, 0, None)) * BORROW
    return gross - turnover * COST - borrow


def relative(pa: pd.Series, pb: pd.Series, lookback: int, every: int, mode: str):
    """mode: 'video' (long laggard, short leader), 'mirror' (long leader, short laggard), 'long_only'."""
    rel = (pa / pa.shift(lookback)) / (pb / pb.shift(lookback))
    a_behind = (rel < 1).to_numpy()
    valid = rel.notna().to_numpy()
    n = len(pa)
    wa, wb = np.zeros(n), np.zeros(n)
    cur = (0.0, 0.0)
    for t in range(n):
        if valid[t] and t % every == 0:
            if mode == "video":
                cur = (0.5, -0.5) if a_behind[t] else (-0.5, 0.5)
            elif mode == "mirror":
                cur = (-0.5, 0.5) if a_behind[t] else (0.5, -0.5)
            else:
                cur = (1.0, 0.0) if a_behind[t] else (0.0, 1.0)
        wa[t], wb[t] = cur
    return wa, wb


def zscore(pa: pd.Series, pb: pd.Series, window: int, entry: float, exit_: float = 0.5, stop: float = 4.0):
    r = np.log(pa / pb)
    z = ((r - r.rolling(window).mean()) / r.rolling(window).std()).to_numpy()
    n = len(pa)
    wa, wb = np.zeros(n), np.zeros(n)
    pos = 0  # +1: long a / short b, -1: the opposite
    for t in range(n):
        if np.isnan(z[t]):
            pos = 0
        elif pos == 0:
            pos = -1 if z[t] > entry else (1 if z[t] < -entry else 0)
        elif abs(z[t]) < exit_ or abs(z[t]) > stop:
            pos = 0
        wa[t], wb[t] = 0.5 * pos, -0.5 * pos
    return wa, wb


def _stats(r: pd.Series) -> dict:
    if len(r) < 20:
        return {}
    years = len(r) / 252
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(252)
    return {"yearly_pct": round(((eq.iloc[-1]) ** (1 / years) - 1) * 100, 2),
            "sharpe": round(r.mean() * 252 / vol, 2) if vol > 0 else None,
            "worst_drop_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
            "growth_of_1000": round(1000 * float(eq.iloc[-1]), 0)}


def evaluate(closes: pd.DataFrame, spy: pd.Series) -> dict:
    pairs = [(a, b) for a, b in PAIRS if a in closes and b in closes]
    rets = closes.pct_change().fillna(0.0)

    def portfolio(make, benchmark=False) -> pd.Series:
        cols = []
        for a, b in pairs:
            wa, wb = make(closes[a], closes[b])
            r = _run(rets[a].to_numpy(), rets[b].to_numpy(), wa, wb)
            if benchmark:  # buy-only is judged against simply holding both names
                r = r - 0.5 * (rets[a].to_numpy() + rets[b].to_numpy())
            cols.append(pd.Series(r, index=closes.index))
        return pd.concat(cols, axis=1).mean(axis=1).iloc[130:]

    configs = []
    for mode, lb, ev in itertools.product(("video", "mirror", "long_only"), (1, 5, 20, 60, 120), (1, 5, 21)):
        configs.append((f"{mode}: lookback {lb}d, rebalance every {ev}d",
                        lambda pa, pb, lb=lb, ev=ev, mode=mode: relative(pa, pb, lb, ev, mode), mode == "long_only"))
    for w, e in itertools.product((20, 60, 120), (1.5, 2.0, 2.5)):
        configs.append((f"classic z-score: window {w}d, enter at {e}", lambda pa, pb, w=w, e=e: zscore(pa, pb, w, e), False))

    rows = []
    for label, make, bench in configs:
        r = portfolio(make, bench)
        train, test = r[r.index < SPLIT], r[r.index >= SPLIT]
        by_year = {str(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}
        rows.append({"rule": label, "vs": "holding both names" if bench else "cash (market-neutral)",
                     "train_2016_2020": _stats(train), "test_2021_on": _stats(test), "all_years": _stats(r),
                     "by_year_pct": by_year})
    rank = lambda x: x["train_2016_2020"].get("sharpe") or -9
    neutral = sorted([x for x in rows if not x["rule"].startswith("long_only")], key=rank, reverse=True)
    long_only = sorted([x for x in rows if x["rule"].startswith("long_only")], key=rank, reverse=True)
    video_daily = next(x for x in rows if x["rule"] == "video: lookback 1d, rebalance every 1d")
    spy_r = spy.pct_change().fillna(0).reindex(closes.index).fillna(0).iloc[130:]
    return {"pairs": [f"{a}/{b}" for a, b in pairs], "costs": "0.05% per trade per side, 0.5%/yr borrow",
            "spy_buy_and_hold": {"train_2016_2020": _stats(spy_r[spy_r.index < SPLIT]), "test_2021_on": _stats(spy_r[spy_r.index >= SPLIT])},
            "video_rule_daily": video_daily,
            "best_market_neutral_by_2016_2020": neutral[:6],
            "worst_market_neutral": neutral[-3:],
            "best_buy_only_by_2016_2020": long_only[:4],
            "rules_tested": len(rows)}


def main() -> None:
    from .exit_backtest import fetch_daily_ohlc

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--out", default="docs/data/pairs_backtest.json")
    args = ap.parse_args()
    symbols = sorted({s for p in PAIRS for s in p} | {"SPY"})
    daily = fetch_daily_ohlc(symbols, args.years)
    closes = pd.DataFrame({s: d["close"] for s, d in daily.items()}).sort_index().ffill()
    closes = closes[closes.index >= closes.index[0] + pd.Timedelta(days=300)]  # drop the warm-up extra
    report = evaluate(closes.drop(columns="SPY"), closes["SPY"])
    report["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    report["first_day"], report["last_day"] = str(closes.index[0].date()), str(closes.index[-1].date())
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
