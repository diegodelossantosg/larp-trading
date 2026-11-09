"""Do candlestick patterns COMBINED with a market condition beat the S&P 500?

    python -m trader.candle_combos

Bullish candlestick patterns (and "any of them", and "two or more the same day") are crossed with conditions
a day trader would add: heavy volume, a deep 5-day fall, near the 52-week low, RSI oversold, below/above the
50-day average, a bull or bear market (SPY vs its 200-day average)... Each trade: buy the next day's open,
leave at the close N days after the signal (N = 5, 10, 20), 0.1% cost. The result of a trade is the stock's
return MINUS SPY's over the same days (so "beating the S&P 500"), and also the plain return.

Two controls, because with hundreds of combinations some look good by luck:
  1. Honest split: combinations are ranked using the DESIGN years (2016-2020) only, and judged on the later
     VALIDATION years (2021 on), which played no part in the choice.
  2. The same condition WITHOUT any candle (every stock-day that meets it) tells whether the candle adds
     anything, or the condition alone does the work ("edge vs condition alone").
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import numpy as np
import pandas as pd

from .candle_backtest import BULL, detect
from .candle_years import COST, fetch_bars, load_members

log = logging.getLogger("candle_combos")

HOLDS = (5, 10, 20)
DESIGN_YEARS = range(2016, 2021)
FILTERS = ["none", "high volume", "fell 5%+ in 5 days", "fell 10%+ in 5 days", "near 52-week low", "RSI under 30",
           "below 50-day avg", "above 50-day avg", "market bull (SPY>200d)", "market bear (SPY<200d)",
           "high volume + fell 5%+", "high volume + near 52-week low", "RSI under 30 + near 52-week low",
           "below 50-day avg + high volume", "fell 10%+ + market bull"]
MIN_TRADES_YEAR = 15


def _rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    s = pd.Series(c)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (100 - 100 / (1 + up / dn)).to_numpy()


def filter_masks(o, h, l, c, v, spy_bull: np.ndarray) -> dict[str, np.ndarray]:
    s = pd.Series
    sma50 = s(c).rolling(50).mean().to_numpy()
    vavg = s(v).rolling(20).mean().shift(1).to_numpy()
    low252 = s(l).rolling(252, min_periods=120).min().to_numpy()
    c5 = np.full(len(c), np.nan)
    c5[5:] = c[:-5]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret5 = c / c5 - 1
        hv = v > 1.5 * vavg
        f5, f10 = ret5 <= -0.05, ret5 <= -0.10
        low = c <= 1.10 * low252
        rsi = _rsi(c) < 30
        dn50, up50 = c < sma50, c > sma50
    bull = spy_bull.astype(bool)
    bear = ~bull
    m = {"none": np.ones(len(c), bool), "high volume": hv, "fell 5%+ in 5 days": f5, "fell 10%+ in 5 days": f10,
         "near 52-week low": low, "RSI under 30": rsi, "below 50-day avg": dn50, "above 50-day avg": up50,
         "market bull (SPY>200d)": bull, "market bear (SPY<200d)": bear,
         "high volume + fell 5%+": hv & f5, "high volume + near 52-week low": hv & low,
         "RSI under 30 + near 52-week low": rsi & low, "below 50-day avg + high volume": dn50 & hv,
         "fell 10%+ + market bull": f10 & bull}
    return {k: np.nan_to_num(x, nan=0).astype(bool) for k, x in m.items()}


def excess_after(o, c, so, sc, hold: int) -> tuple[np.ndarray, np.ndarray]:
    """(stock return, stock return minus SPY's) buying the next open and selling `hold` days after the signal."""
    n = len(c)
    entry, exit_ = np.full(n, np.nan), np.full(n, np.nan)
    sentry, sexit = np.full(n, np.nan), np.full(n, np.nan)
    entry[: n - 1], sentry[: n - 1] = o[1:], so[1:]
    if hold < n:
        exit_[: n - hold], sexit[: n - hold] = c[hold:], sc[hold:]
    with np.errstate(invalid="ignore", divide="ignore"):
        r, sr = exit_ / entry - 1, sexit / sentry - 1
    return r, r - sr


def analyze(bars: dict[str, pd.DataFrame], spy: pd.DataFrame, member: dict[int, set[str]], years: list[int]) -> dict:
    sig: dict[tuple, list] = {}          # (pattern, filter, year, hold) -> [(excess, plain)]
    ctl: dict[tuple, list] = {}          # (filter, year, hold) -> [sum excess, sum plain, count]
    spy = spy.dropna(subset=["open", "close"])
    spy_sma = spy["close"].rolling(200).mean()
    spy_bull_s = (spy["close"] > spy_sma)
    patterns = BULL + ["any bullish pattern", "2+ patterns the same day"]
    for sym, df in bars.items():
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        if len(df) < 300 or sym == "SPY":
            continue
        o, h, l, c, v = (df[k].to_numpy("float64") for k in ("open", "high", "low", "close", "volume"))
        so = spy["open"].reindex(df.index).to_numpy("float64")
        sc = spy["close"].reindex(df.index).to_numpy("float64")
        bull_mk = spy_bull_s.reindex(df.index).fillna(False).to_numpy()
        yrs = df.index.year.to_numpy()
        det = detect(o, h, l, c)
        pm = {p: det[p][0] for p in BULL}
        pm["any bullish pattern"] = np.any([det[p][0] for p in BULL], axis=0)
        pm["2+ patterns the same day"] = np.sum([det[p][0] for p in BULL], axis=0) >= 2
        fm = filter_masks(o, h, l, c, v, bull_mk)
        res = {hd: excess_after(o, c, so, sc, hd) for hd in HOLDS}
        for y in years:
            if sym not in member.get(y, ()):
                continue
            iny = yrs == y
            if not iny.any():
                continue
            for hd in HOLDS:
                r, ex = res[hd]
                ok = iny & np.isfinite(ex)
                if not ok.any():
                    continue
                for fname, fmask in fm.items():
                    base = ok & fmask
                    if base.any():
                        a = ctl.setdefault((fname, y, hd), [0.0, 0.0, 0])
                        a[0] += float(ex[base].sum()); a[1] += float(r[base].sum()); a[2] += int(base.sum())
                    for pat in patterns:
                        m = base & pm[pat]
                        if m.any():
                            sig.setdefault((pat, fname, y, hd), []).extend(zip(ex[m].tolist(), r[m].tolist()))
    return summarize(sig, ctl, years, patterns)


def _t(a: np.ndarray) -> float:
    return float(a.mean() / (a.std() / math.sqrt(len(a)))) if len(a) > 2 and a.std() > 0 else 0.0


def summarize(sig, ctl, years, patterns) -> dict:
    design = [y for y in years if y in DESIGN_YEARS]
    valid = [y for y in years if y not in DESIGN_YEARS]
    rows = []
    for pat in patterns:
        for fname in FILTERS:
            for hd in HOLDS:
                def collect(ys):
                    ex, pl, per = [], [], {}
                    edge = []
                    for y in ys:
                        xs = sig.get((pat, fname, y, hd), [])
                        if not xs:
                            continue
                        e = np.array([a for a, _ in xs]) - COST
                        p = np.array([b for _, b in xs]) - COST
                        c = ctl.get((fname, y, hd))
                        cb = (c[0] / c[2] - COST) if c and c[2] else float("nan")
                        ex.append(e); pl.append(p)
                        per[y] = (float(e.mean() * 100), len(e), float((e - cb).mean() * 100) if not math.isnan(cb) else None)
                        if not math.isnan(cb):
                            edge.append(e - cb)
                    if not ex:
                        return None
                    e, p = np.concatenate(ex), np.concatenate(pl)
                    ed = np.concatenate(edge) if edge else np.array([0.0])
                    ys_ok = [v for v in per.values() if v[1] >= MIN_TRADES_YEAR]
                    return {"n": int(len(e)), "excess_pct": float(e.mean() * 100), "plain_pct": float(p.mean() * 100),
                            "win_vs_spy": float((e > 0).mean()), "edge_vs_condition_pct": float(ed.mean() * 100),
                            "t_excess": _t(e), "t_edge": _t(ed), "years_judged": len(ys_ok),
                            "years_beating_spy": sum(1 for v in ys_ok if v[0] > 0),
                            "years_beating_condition": sum(1 for v in ys_ok if v[2] is not None and v[2] > 0)}
                d, v = collect(design), collect(valid)
                if d and v:
                    rows.append({"pattern": pat, "condition": fname, "hold_days": hd, "design": d, "validation": v})
    judged = [r for r in rows if r["design"]["n"] >= 150 and r["validation"]["n"] >= 100]
    top = sorted(judged, key=lambda r: -r["design"]["excess_pct"])[:15]
    passing = [r for r in judged if r["validation"]["excess_pct"] > 0 and r["validation"]["t_excess"] > 3
               and r["validation"]["edge_vs_condition_pct"] > 0 and r["validation"]["t_edge"] > 2
               and r["validation"]["years_beating_spy"] >= 0.7 * max(r["validation"]["years_judged"], 1)]
    cond_only = []
    for fname in FILTERS:
        for hd in HOLDS:
            tot = {"ex": 0.0, "pl": 0.0, "n": 0}
            for y in valid:
                c = ctl.get((fname, y, hd))
                if c:
                    tot["ex"] += c[0]; tot["pl"] += c[1]; tot["n"] += c[2]
            if tot["n"]:
                cond_only.append({"condition": fname, "hold_days": hd, "n": tot["n"],
                                  "excess_pct": (tot["ex"] / tot["n"] - COST) * 100, "plain_pct": (tot["pl"] / tot["n"] - COST) * 100})
    return {"cost_round_trip_pct": COST * 100, "design_years": list(design), "validation_years": list(valid),
            "combinations_judged": len(judged), "top_by_design": top, "passing_validation": passing,
            "condition_alone_validation": cond_only,
            "all": [{k: r[k] for k in ("pattern", "condition", "hold_days")} | {"design_excess_pct": r["design"]["excess_pct"],
                    "validation_excess_pct": r["validation"]["excess_pct"], "validation_n": r["validation"]["n"],
                    "validation_edge_pct": r["validation"]["edge_vs_condition_pct"]} for r in judged]}


def main() -> None:
    from datetime import datetime, timedelta, timezone

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data/candle_combos.json")
    ap.add_argument("--max-symbols", type=int, default=0)
    args = ap.parse_args()
    member, years = load_members()
    symbols = sorted({s for v in member.values() for s in v})
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    bars = fetch_bars(symbols + ["SPY"], pd.Timestamp("2015-01-01", tz="UTC"), end)
    spy = bars.pop("SPY")
    log.info("downloaded %d symbols", len(bars))
    res = analyze(bars, spy, member, years)
    res["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M+00:00")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    log.info("combinations judged: %d; passing validation: %d", res["combinations_judged"], len(res["passing_validation"]))
    for r in res["top_by_design"]:
        log.info("%-26s %-34s %2dd design %+.2f%% -> validation %+.2f%% (edge %+.2f%%, n=%d)", r["pattern"], r["condition"],
                 r["hold_days"], r["design"]["excess_pct"], r["validation"]["excess_pct"],
                 r["validation"]["edge_vs_condition_pct"], r["validation"]["n"])


if __name__ == "__main__":
    main()
