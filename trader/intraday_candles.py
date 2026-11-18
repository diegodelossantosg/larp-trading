"""Honest re-test of same-day candlestick trading, with realistic per-symbol costs.

    python -m trader.intraday_candles --years 5

What changed vs trader/candle_backtest.py (which found nothing):
- costs per symbol instead of a flat 0.1%: 0.02% per side for SPY/QQQ, 0.03% for IWM/DIA,
  0.05% for other ETFs and big stocks (round trip = twice that);
- 1-hour (and 30-minute) candles really get tested: the old same-session check needed 7 candles
  of the same day, and a day only has 7 one-hour candles, so 1 hour tested nothing;
- control = random entries at the SAME time of day on other days of the same symbol, same side,
  same stop size, same exit (so time-of-day effects don't look like a pattern edge);
- design period (first 60% of the days) picks the rules; the later 40% validates them;
- luck check: the validation is re-run on 500 random halves of the symbols; a rule only counts
  if it wins (after costs AND vs the control) in at least 75% of them.

Every trade enters at the next candle's open and is always closed by the 16:00 close. Exits:
"2:1 target" (stop past the pattern, target twice the risk), "hold 1 hour", "hold to the close".
Bearish patterns are traded short. "$1,000 pot": 5 slots of $200, at most 5 trades a day (the
first ones of the day), compared with holding $1,000 of SPY over the same validation days.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .candle_backtest import BEAR, COST, ETFS, _sh, contexts, detect, intraday_trades, resample

log = logging.getLogger("intraday_candles")

TIMEFRAMES = {"5 min": 5, "15 min": 15, "30 min": 30, "1 hour": 60}
COST_PER_SIDE = {"SPY": 0.0002, "QQQ": 0.0002, "IWM": 0.0003, "DIA": 0.0003}
DEFAULT_COST = 0.0005
DESIGN_SHARE = 0.6
CONTROLS = 3
POT, SLOTS = 1000.0, 5
LUCK_DRAWS, LUCK_PASS = 500, 0.75


def cost_rt(sym: str) -> float:
    return 2 * COST_PER_SIDE.get(sym, DEFAULT_COST)


def gross_trades(o, h, l, c, day_last, idx, stop, short, hold_bars) -> dict[str, np.ndarray]:
    """Gross returns (no costs) for each exit rule; reuses the tested candle_backtest simulator."""
    return {k: v + COST for k, v in intraday_trades(o, h, l, c, day_last, idx, stop, short, hold_bars).items()}


def same_time_controls(day: np.ndarray, slot: np.ndarray, idx: np.ndarray, k: int, rng) -> np.ndarray:
    """For each entry bar, k random bars at the same slot on other days of the same symbol (-1 if none)."""
    nd, ns = int(day.max()) + 1, int(slot.max()) + 1
    grid = np.full((nd, ns), -1, dtype=np.int64)
    grid[day, slot] = np.arange(len(day))
    out = np.full((len(idx), k), -1, dtype=np.int64)
    for j in range(k):
        d = rng.integers(0, nd, len(idx))
        d = np.where(d == day[idx], (d + 1) % nd, d)
        out[:, j] = grid[d, slot[idx]]
    return out


def symbol_trades(df: pd.DataFrame, minutes: int, rng) -> dict:
    """{(pattern, context, exit): dict of arrays date/time/gross/ctrl_gross} for one symbol & timeframe."""
    ny = df.index.tz_convert("America/New_York")
    dates = np.array(ny.date)
    day = pd.factorize(dates)[0]
    n = len(df)
    day_last = pd.Series(np.arange(n)).groupby(day).transform("max").to_numpy()
    slot = ((ny.hour * 60 + ny.minute - 570) // minutes).to_numpy()
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    hold_bars = max(1, 60 // minutes)
    can_enter = np.arange(n) < day_last
    out = {}
    ctx = {True: contexts(c, v, True), False: contexts(c, v, False)}
    for name, (mask, nb) in detect(o, h, l, c).items():
        same_session = _sh(day.astype(float), nb - 1) == day      # the pattern's own candles: one session
        idx = np.flatnonzero(mask & same_session & can_enter)
        if not len(idx):
            continue
        short = name in BEAR
        lows = np.min([_sh(l, k)[idx] for k in range(nb)], axis=0)
        highs = np.max([_sh(h, k)[idx] for k in range(nb)], axis=0)
        stop = highs if short else lows
        rets = gross_trades(o, h, l, c, day_last, idx, stop, short, hold_bars)
        # controls: same slot, other day, same relative stop distance from that bar's close
        frac = stop / c[idx]
        ctl = same_time_controls(day, slot, idx, CONTROLS, rng)
        ctl_rets = defaultdict(list)
        for j in range(CONTROLS):
            ci = ctl[:, j]
            okc = (ci >= 0) & can_enter[np.maximum(ci, 0)]
            cidx = np.where(okc, ci, idx)          # placeholder where no control exists (masked below)
            r = gross_trades(o, h, l, c, day_last, cidx, c[cidx] * frac, short, hold_bars)
            for ex, x in r.items():
                ctl_rets[ex].append(np.where(okc, x, np.nan))
        ctl_mean = {ex: np.nanmean(np.vstack(xs), axis=0) if len(xs) else None for ex, xs in ctl_rets.items()}
        minute_of_day = slot[idx] * minutes
        for cname, cm in ctx[not short].items():
            sel = cm[idx]
            for ex, r in rets.items():
                keep = sel & ~np.isnan(r) & ~np.isnan(ctl_mean[ex])
                if keep.any():
                    out[(name, cname, ex)] = {"date": dates[idx][keep], "minute": minute_of_day[keep],
                                              "gross": r[keep], "ctrl": ctl_mean[ex][keep]}
    return out


def _stats(net: np.ndarray, edge: np.ndarray) -> dict:
    n = len(net)
    if n < 2:
        return {"trades": int(n)}
    sd = net.std(ddof=1)
    return {"trades": int(n), "avg_net_pct": round(float(net.mean()) * 100, 4),
            "edge_vs_control_pct": round(float(edge.mean()) * 100, 4),
            "win_rate": round(float((net > 0).mean()), 3),
            "t_net": round(float(net.mean() / (sd / n ** 0.5)), 2) if sd > 0 else None}


def pot_path(dates, minutes, net, days: list) -> float:
    """$1,000 in 5 slots; each day the first 5 trades (by time) each use one slot."""
    df = pd.DataFrame({"d": dates, "m": minutes, "r": net}).sort_values(["d", "m"])
    daily = df.groupby("d")["r"].apply(lambda x: x.iloc[:SLOTS].sum() / SLOTS)
    pot = POT
    for d in days:
        if d in daily.index:
            pot *= 1 + daily[d]
    return round(pot, 2)


def evaluate(trades: dict, spy_close: pd.Series | None, rng) -> dict:
    """trades: {(timeframe, pattern, context, exit): {sym: arrays}} -> report."""
    all_days = sorted({d for per_sym in trades.values() for t in per_sym.values() for d in t["date"]})
    if not all_days:
        return {"rules_tested": 0, "passing": [], "picked_in_design": []}
    split = all_days[int(len(all_days) * DESIGN_SHARE)]
    val_days = [d for d in all_days if d >= split]
    rows = []
    for key, per_sym in trades.items():
        syms = sorted(per_sym)
        cat = {f: np.concatenate([per_sym[s][f] for s in syms]) for f in ("date", "minute", "gross", "ctrl")}
        sym_id = np.concatenate([np.full(len(per_sym[s]["gross"]), i) for i, s in enumerate(syms)])
        cost = np.concatenate([np.full(len(per_sym[s]["gross"]), cost_rt(s)) for s in syms])
        net, edge = cat["gross"] - cost, cat["gross"] - cat["ctrl"]
        des = cat["date"] < split
        d = _stats(net[des], edge[des])
        row = {"timeframe": key[0], "pattern": key[1], "side": "sell (short)" if key[1] in BEAR else "buy",
               "context": key[2], "exit": key[3], "design": d}
        picked = (d.get("trades", 0) >= 200 and d["avg_net_pct"] > 0 and d["edge_vs_control_pct"] > 0
                  and (d["t_net"] or 0) > 2)
        if picked:
            val = ~des
            v = _stats(net[val], edge[val])
            v["avg_net_pct_double_costs"] = round(float((net[val] - cost[val]).mean()) * 100, 4) if val.any() else None
            years = pd.Series(net[val]).groupby(pd.to_datetime(pd.Series(cat["date"][val])).dt.year.to_numpy()).mean()
            v["years_positive"] = f"{int((years > 0).sum())}/{len(years)}"
            # luck check: random halves of the symbols
            wins = 0
            ids = np.unique(sym_id[val])
            for _ in range(LUCK_DRAWS if len(ids) >= 4 else 0):
                half = rng.choice(ids, max(2, len(ids) // 2), replace=False)
                m = val & np.isin(sym_id, half)
                if m.sum() >= 20 and net[m].mean() > 0 and edge[m].mean() > 0:
                    wins += 1
            v["luck_share_winning"] = round(wins / LUCK_DRAWS, 3) if len(ids) >= 4 else None
            v["pot_end_on_1000"] = pot_path(cat["date"][val], cat["minute"][val], net[val], val_days)
            row["validation"] = v
            row["passes"] = bool(v.get("trades", 0) >= 100 and v["avg_net_pct"] > 0 and v["edge_vs_control_pct"] > 0
                                 and (v["luck_share_winning"] or 0) >= LUCK_PASS)
        rows.append(row)
    picked = [r for r in rows if "validation" in r]
    picked.sort(key=lambda r: -r["validation"].get("avg_net_pct", -99))
    best_design = sorted(rows, key=lambda r: -r["design"].get("avg_net_pct", -99))[:10]
    spy_hold = None
    if spy_close is not None and len(val_days) > 1:
        s = spy_close[spy_close.index.isin(val_days)]
        if len(s) > 1:
            spy_hold = round(POT * float(s.iloc[-1] / s.iloc[0]), 2)
    return {"period": f"{all_days[0]} to {all_days[-1]}", "validation_starts": str(split),
            "validation_days": len(val_days), "rules_tested": len(rows),
            "picked_in_design": picked, "passing": [r for r in picked if r["passes"]],
            "best_in_design": best_design, "spy_hold_validation_on_1000": spy_hold}


def run(bars: dict[str, pd.DataFrame], seed: int = 11, timeframes: dict[str, int] = TIMEFRAMES, base: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    trades: dict = defaultdict(dict)
    for tf, minutes in timeframes.items():
        for sym, raw in bars.items():
            df = resample(raw, minutes, base)
            if len(df) < 200:
                continue
            for (p, cx, ex), t in symbol_trades(df, minutes, rng).items():
                trades[(tf, p, cx, ex)][sym] = t
            log.info("%s %s done", tf, sym)
    spy = None
    if "SPY" in bars:
        s = bars["SPY"]
        spy = s.groupby(np.array(s.index.tz_convert("America/New_York").date))["close"].last()
    report = evaluate(trades, spy, rng)
    report.update({"symbols": sorted(bars), "costs_round_trip_pct": {"SPY/QQQ": 0.04, "IWM/DIA": 0.06, "others": 0.1},
                   "timeframes": list(timeframes)})
    return report


def main() -> None:
    from .candle_backtest import fetch_intraday
    from .universe import BIG_STOCKS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--out", default="docs/data/intraday_candles.json")
    args = ap.parse_args()
    bars = fetch_intraday(sorted(set(ETFS) | set(BIG_STOCKS)), args.years)
    report = run(bars)
    report["generated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("period", "validation_starts", "rules_tested", "passing",
                                             "spy_hold_validation_on_1000")}, indent=1, default=str))
    print(json.dumps(report["picked_in_design"][:15], indent=1, default=str))


if __name__ == "__main__":
    main()
