"""Which intraday buy signals for ETFs would have worked best? An audit of the Guard's rules.

    python -m trader.signal_backtest --months 24

Every 15-minute bar of every ETF over two years gets 15 yes/no signals (the Guard's current six plus
nine new candidates). A "rule" is a set of signals plus how many must agree. When a rule fires
(after the first half hour, SPY not down more than 0.5% today, first time that ETF fires that day,
at most 4 buys a day), we buy at the next bar's open and hold up to 1, 3 or 5 trading days with a
4%, 6% or 8% trailing stop. Each trade is compared with SPY over the same time, after 0.1% costs.

Against fooling ourselves: rules are ranked on the FIRST year only; the best ones are then checked on
the SECOND year, which played no part in choosing them. Only that second-year number counts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import statistics
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .universe import CORE_ETFS, LEVERAGED

log = logging.getLogger("signal_backtest")

COST = 0.001
MAX_BUYS_PER_DAY = 4
HOLDS = (1, 3, 5)
STOPS = (0.04, 0.06, 0.08)

SIGNALS = {
    # The Guard's current six.
    "uptrend": "price above its 20-day average, which is above the 50-day",
    "healthy_rise": "up 1% (or 0.75x its usual daily swing) to 6% today",
    "above_vwap": "above today's average traded price",
    "bullish_candle": "hammer or bullish engulfing on the last 15-minute bar",
    "volume_surge": "last 15 minutes traded 1.5x today's average",
    "high_20d": "at a new 20-day high",
    # New candidates.
    "rel_strength": "beat the S&P 500 over the last 3 months",
    "opening_breakout": "above the high of the first 30 minutes",
    "pullback": "uptrend, dipped to its 20-day average yesterday, rising today",
    "not_overheated": "daily RSI below 70",
    "oversold_uptrend": "daily RSI below 45 while above its 50-day average",
    "calm_market": "S&P 500 above its 50-day average and its 20-day swings below 20% a year",
    "sector_rotation": "beat the S&P 500 over the last 5 days",
    "gap_holds": "opened at least 0.3% higher and is still above the open",
    "volume_trend": "5-day average volume 20% above the 20-day average",
}
NAMES = list(SIGNALS)
CURRENT = NAMES[:6]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _daily_features(daily: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """Per day, what was known before that day's open."""
    c = daily["close"]
    tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - c.shift()).abs(), (daily["low"] - c.shift()).abs()],
                   axis=1).max(axis=1)
    spy_c = spy["close"].reindex(c.index).ffill()
    f = pd.DataFrame({
        "prev_close": c,
        "sma20": c.rolling(20).mean(),
        "sma50": c.rolling(50).mean(),
        "high20": c.rolling(20).max(),
        "atr_pct": tr.rolling(14).mean() / c * 100,
        "rs63": c.pct_change(63) - spy_c.pct_change(63),
        "rs5": c.pct_change(5) - spy_c.pct_change(5),
        "rsi": _rsi(c),
        "low": daily["low"],
        "vol_trend": daily["volume"].rolling(5).mean() / daily["volume"].rolling(20).mean(),
        "spy_ok": (spy_c > spy_c.rolling(50).mean()) & (spy_c.pct_change().rolling(20).std() * np.sqrt(252) < 0.20),
    })
    return f.shift(1)  # only yesterday's information is known at today's open


def build(intraday: dict[str, pd.DataFrame], daily: dict[str, pd.DataFrame]):
    """Signal table (one row per ETF per 15-minute bar) plus what is needed to price exits."""
    spy_i, spy_d = intraday["SPY"], daily["SPY"]
    spy_open = {(ts.date(), ts.time()): o for ts, o in zip(spy_i.index, spy_i["open"])}
    spy_prev = spy_d["close"].shift(1)
    spy_prev = {ts.date(): v for ts, v in spy_prev.items()}
    rows, meta = [], []
    for sym in sorted(intraday):
        if sym in LEVERAGED or sym not in daily or sym not in CORE_ETFS or sym == "SPY":
            continue
        feats = _daily_features(daily[sym], spy_d)
        fdates = {ts.date(): i for i, ts in enumerate(feats.index)}
        df = intraday[sym]
        for day, b in df.groupby(df.index.date):
            if day not in fdates or len(b) < 6:
                continue
            f = feats.iloc[fdates[day]].to_dict()
            if any(pd.isna(f[x]) for x in ("sma50", "rs63", "rsi", "atr_pct", "vol_trend")):
                continue
            o, h, l, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
            vw = b["vwap"].to_numpy(float)
            cumv = np.cumsum(v)
            vwap = np.cumsum(vw * v) / np.where(cumv > 0, cumv, 1)
            avgv = cumv / np.arange(1, len(v) + 1)
            body = np.abs(c - o)
            lower = np.minimum(o, c) - l
            upper = h - np.maximum(o, c)
            hammer = (lower >= 2 * body) & (upper <= body + 1e-9) & (h > l)
            engulf = np.zeros(len(c), bool)
            engulf[1:] = (c[:-1] < o[:-1]) & (c[1:] > o[1:]) & (c[1:] >= o[:-1]) & (o[1:] <= c[:-1])
            or_high = max(h[0], h[1])
            prev = f["prev_close"]
            for k in range(1, len(b) - 1):        # bar k has closed; buy at bar k+1's open
                ts = b.index[k]
                spy_now = spy_open.get((day, b.index[k + 1].time()))
                if spy_now is None or day not in spy_prev or spy_now / spy_prev[day] - 1 < -0.005:
                    continue
                price, chg = c[k], c[k] / prev - 1
                rows.append((
                    price > f["sma20"] > f["sma50"],
                    max(0.01, 0.75 * f["atr_pct"] / 100) <= chg <= 0.06,
                    price > vwap[k],
                    bool(hammer[k] or engulf[k]),
                    v[k] >= 1.5 * avgv[k],
                    price >= f["high20"],
                    f["rs63"] > 0,
                    k >= 2 and price > or_high,
                    f["sma20"] > f["sma50"] and f["low"] <= f["sma20"] * 1.01 and price > prev,
                    f["rsi"] < 70,
                    f["rsi"] < 45 and price > f["sma50"],
                    bool(f["spy_ok"]),
                    f["rs5"] > 0,
                    o[0] >= prev * 1.003 and price > o[0],
                    f["vol_trend"] > 1.2,
                ))
                meta.append((sym, day, k, ts))
        log.info("signals: %s done (%d rows so far)", sym, len(rows))
    sig = np.array(rows, dtype=bool)
    meta = pd.DataFrame(meta, columns=["sym", "day", "k", "ts"])
    return sig, meta


class Exits:
    """Result vs SPY of buying at the bar after row i, for each hold/stop, computed once and cached."""

    def __init__(self, intraday, daily, meta):
        self.meta = meta[["sym", "day", "k"]].to_numpy()
        self.bars = {}
        for s, df in intraday.items():
            for d, g in df.groupby(df.index.date):
                self.bars[(s, d)] = (g.index, *(g[c].to_numpy(float) for c in ("open", "high", "low", "close")))
        self.daily = {s: {c: df[c].to_numpy(float) for c in ("open", "high", "low", "close")} for s, df in daily.items()}
        self.dates = {s: [ts.date() for ts in df.index] for s, df in daily.items()}
        self.pos = {s: {d: i for i, d in enumerate(ds)} for s, ds in self.dates.items()}
        self.cache: dict[tuple, float | None] = {}

    def result(self, i: int, hold: int, stop: float) -> float | None:
        key = (i, hold, stop)
        if key not in self.cache:
            self.cache[key] = self._result(i, hold, stop)
        return self.cache[key]

    def _result(self, i, hold, stop):
        sym, day, k = self.meta[i]
        idx, o, h, l, c = self.bars[(sym, day)]
        spy = self.bars.get(("SPY", day))
        if spy is None or idx[k + 1] not in spy[0]:
            return None
        entry = o[k + 1]
        spy_entry = spy[1][spy[0].get_loc(idx[k + 1])]
        high, out, exit_day = entry, None, day
        for j in range(k + 1, len(c)):            # rest of the entry day, bar by bar
            if l[j] <= high * (1 - stop):
                out = min(high * (1 - stop), o[j])
                break
            high = max(high, h[j])
        d0 = self.pos[sym].get(day)
        if out is None:
            if d0 is None:
                return None
            dd, dates = self.daily[sym], self.dates[sym]
            for n in range(1, hold + 1):          # following days, using daily high/low
                if d0 + n >= len(dates):
                    return None
                exit_day = dates[d0 + n]
                if dd["low"][d0 + n] <= high * (1 - stop):
                    out = min(high * (1 - stop), dd["open"][d0 + n])
                    break
                high = max(high, dd["high"][d0 + n])
                if n == hold:
                    out = dd["close"][d0 + n]
        sp = self.pos["SPY"].get(exit_day)
        if out is None or sp is None:
            return None
        return float(out / entry - self.daily["SPY"]["close"][sp] / spy_entry - COST)


def trades(rule_mask: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Row ids actually bought: first firing per ETF per day, at most 4 a day, earliest first."""
    m = meta[rule_mask]
    if m.empty:
        return np.array([], int)
    first = m.groupby(["sym", "day"], sort=False).head(1)
    first = first.sort_values(["day", "k", "sym"])
    return first.groupby("day").head(MAX_BUYS_PER_DAY).index.to_numpy()


def score(ids, exits: Exits, hold: int, stop: float) -> dict:
    xs = [r for r in (exits.result(i, hold, stop) for i in ids) if r is not None]
    if len(xs) < 3:
        return {"trades": len(xs)}
    sd = statistics.stdev(xs)
    return {"trades": len(xs), "avg_vs_spy_pct": round(statistics.mean(xs) * 100, 3),
            "beat_spy_rate": round(sum(x > 0 for x in xs) / len(xs), 3),
            "t_stat": round(statistics.mean(xs) / (sd / len(xs) ** 0.5), 2) if sd else None,
            "total_vs_spy_pct": round(sum(xs) * 100, 1)}


def candidate_rules() -> list[tuple[tuple[str, ...], int]]:
    rules = [(tuple(CURRENT), m) for m in (4, 5, 6)]
    for size in (1, 2, 3):
        rules += [(combo, size) for combo in itertools.combinations(NAMES, size)]
    # The current six plus one new filter that must also hold.
    rules += [(tuple(CURRENT) + (extra,), 6) for extra in NAMES[6:]]
    return rules


def _rule_fire(sig, combo, need, col):
    if len(combo) == 7:                       # the current six (5 of 6) plus one extra filter
        return (sig[:, [col[n] for n in CURRENT]].sum(1) >= 5) & sig[:, col[combo[6]]], f"current 5-of-6 + {combo[6]}"
    fire = sig[:, [col[n] for n in combo]].sum(1) >= need
    if combo == tuple(CURRENT):
        return fire, f"current {need}-of-6"
    return fire, " + ".join(combo) + ("" if need == len(combo) else f" ({need} of {len(combo)})")


def run(sig: np.ndarray, meta: pd.DataFrame, exits: Exits, min_train_trades: int = 60, shortlist: int = 40,
        top: int = 15) -> dict:
    days = sorted(meta["day"].unique())
    split = days[len(days) // 2]
    train_mask = (meta["day"] < split).to_numpy()
    col = {n: i for i, n in enumerate(NAMES)}
    # Stage 1: every rule with the Guard's current exit (hold up to 3 days, 6% stop), first year only.
    stage1 = []
    for combo, need in candidate_rules():
        fire, label = _rule_fire(sig, combo, need, col)
        ids = trades(fire & train_mask, meta)
        if len(ids) < min_train_trades:
            continue
        sc = score(ids, exits, 3, 0.06)
        if sc.get("t_stat") is not None:
            stage1.append((sc["t_stat"], label, fire))
    stage1.sort(key=lambda x: -x[0])
    log.info("stage 1: %d rules scored", len(stage1))
    keep = stage1[:shortlist] + [x for x in stage1[shortlist:] if x[1] == "current 5-of-6"]
    # Stage 2: the shortlist with every hold/stop, still first year only.
    results = []
    for _, label, fire in keep:
        ids = trades(fire & train_mask, meta)
        for hold in HOLDS:
            for stop in STOPS:
                tr = score(ids, exits, hold, stop)
                if tr.get("t_stat") is not None:
                    results.append({"rule": label, "hold_days": hold, "stop_pct": int(stop * 100), "train": tr,
                                    "_fire": fire})
    results.sort(key=lambda r: -r["train"]["t_stat"])
    picked, seen = [], set()
    for r in results:                         # best distinct rules by first-year evidence
        if r["rule"] not in seen:
            seen.add(r["rule"])
            picked.append(r)
        if len(picked) == top:
            break
    current = [r for r in results if r["rule"] == "current 5-of-6" and r["hold_days"] == 3 and r["stop_pct"] == 6]
    for r in picked + current:                # Stage 3: the untouched second year decides
        r["test"] = score(trades(r["_fire"] & ~train_mask, meta), exits, r["hold_days"], r["stop_pct"] / 100)
    clean = lambda r: {k: v for k, v in r.items() if k != "_fire"}
    return {"first_year": f"{days[0]} to {split}", "second_year": f"{split} to {days[-1]}",
            "rules_tried": len(stage1), "signals": SIGNALS,
            "current_rule": [clean(r) for r in current], "best_by_first_year": [clean(r) for r in picked]}


def main() -> None:
    from . import backtest

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--out", default="docs/data/signal_backtest.json")
    args = ap.parse_args()
    symbols = sorted(set(CORE_ETFS) - LEVERAGED)
    intraday, daily = backtest.fetch(args.months, symbols)
    for s, df in daily.items():   # day-level index for daily bars
        df.index = pd.DatetimeIndex([ts.tz_convert("America/New_York").normalize().tz_localize(None) for ts in df.index])
    for s, df in intraday.items():
        df.index = df.index.tz_convert("America/New_York")
    sig, meta = build(intraday, daily)
    log.info("signal table: %d rows", len(meta))
    report = run(sig, meta, Exits(intraday, daily, meta))
    report["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "signals"}, indent=1, default=str))


if __name__ == "__main__":
    main()
