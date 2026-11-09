"""Which candlestick patterns usually make money? Every year on its own, daily candles, many stocks.

    python -m trader.candle_years

The 16 classic patterns (8 bullish, 8 bearish, the same definitions as candle_backtest) are looked for on the
daily candles of that year's ~500 most-traded US stocks (research/lab.npz decides who, delisted ones included;
the prices, with highs, are downloaded here). Each calendar year is its own sample, 2016 to now, so a pattern has
to work year after year, not just once.

Trading rule (the textbook one, no look-ahead): after the pattern's last candle closes, buy (bullish) or sell
short (bearish) at the NEXT day's open, and leave at the close of day N after the signal (N = 1, 3, 5, 10), or
with the "2:1" rule: stop just beyond the pattern's extreme, target twice the risk, otherwise out after 10 days.
Every trade pays 0.1% in costs. Shorting is shown without borrowing costs, which only makes it look better.

"Works" is judged on the money made (not versus the S&P 500), but a pattern must also beat doing the same trade
on ordinary days in the same situation (a stock that has just fallen / risen for 5 days, no candle needed),
because in a rising market almost anything wins.
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import numpy as np
import pandas as pd

from .candle_backtest import BEAR, BULL, detect

log = logging.getLogger("candle_years")

COST = 0.001
HOLDS = (1, 3, 5, 10)
MIN_TRADES_YEAR = 20          # a year counts only with at least this many trades
MIN_YEARS = 6                 # ... and a pattern needs this many such years to be judged
STOP_MAX_DAYS = 10
RNG_SEED = 7


def _sh(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[: len(x) - k]
    return out


def returns_after(o: np.ndarray, c: np.ndarray, hold: int) -> np.ndarray:
    """Long return of buying at the next open and selling at the close `hold` days after the signal day."""
    n = len(c)
    entry = np.full(n, np.nan)
    entry[: n - 1] = o[1:]
    exit_ = np.full(n, np.nan)
    if hold < n:
        exit_[: n - hold] = c[hold:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return exit_ / entry - 1


def stop_target(o, h, l, c, t: int, long: bool, stop_price: float):
    """2:1 trade from signal day t. Returns the gross return, or None if there is no valid trade."""
    n = len(c)
    if t + 1 >= n or not np.isfinite(o[t + 1]):
        return None
    entry = o[t + 1]
    risk = (entry - stop_price) if long else (stop_price - entry)
    if not np.isfinite(risk) or risk <= 0 or risk > 0.25 * entry:
        return None
    target = entry + 2 * risk if long else entry - 2 * risk
    last = min(n - 1, t + STOP_MAX_DAYS)
    if last <= t:
        return None
    for d in range(t + 1, last + 1):
        lo, hi = l[d], h[d]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if long:
            if o[d] <= stop_price and d > t + 1:
                return o[d] / entry - 1
            if lo <= stop_price:                       # the stop is checked first when both are touched
                return stop_price / entry - 1
            if hi >= target:
                return target / entry - 1
        else:
            if o[d] >= stop_price and d > t + 1:
                return entry / o[d] - 1
            if hi >= stop_price:
                return entry / stop_price - 1
            if lo <= target:
                return entry / target - 1
    cl = c[last]
    return (cl / entry - 1) if long else (entry / cl - 1)


def analyze(bars: dict[str, pd.DataFrame], member: dict[int, set[str]], years: list[int],
            seed: int = RNG_SEED) -> dict:
    """bars: symbol -> daily DataFrame with open/high/low/close/volume. member: year -> set of symbols."""
    rng = np.random.default_rng(seed)
    names = BULL + BEAR
    # per (pattern, year, hold): list of gross LONG returns of the signal trades
    sig: dict[tuple, list] = {}
    stp: dict[tuple, list] = {}
    base = {}                       # (direction, year, hold, kind) -> [sum, count]; kind: any / ctx
    stp_pool: dict[tuple, list] = {}  # (direction, year) -> [(sym, t, stop)] sampled ordinary-day trades
    arrays = {}
    for sym, df in bars.items():
        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) < 60:
            continue
        o, h, l, c = (df[k].to_numpy("float64") for k in ("open", "high", "low", "close"))
        arrays[sym] = (df.index, o, h, l, c)
        yrs = df.index.year.to_numpy()
        masks = detect(o, h, l, c)
        rets = {hd: returns_after(o, c, hd) for hd in HOLDS}
        fell = _sh(c, 1) < _sh(c, 6)
        rose = _sh(c, 1) > _sh(c, 6)
        for y in years:
            if sym not in member.get(y, ()):
                continue
            iny = yrs == y
            if not iny.any():
                continue
            for hd in HOLDS:
                r = rets[hd]
                ok = iny & np.isfinite(r)
                for direction, ctx in (("long", fell), ("short", rose)):
                    kk = (direction, y, hd)
                    b = base.setdefault(kk + ("any",), [0.0, 0])
                    b[0] += float(r[ok].sum()); b[1] += int(ok.sum())
                    cx = ok & ctx
                    b = base.setdefault(kk + ("ctx",), [0.0, 0])
                    b[0] += float(r[cx].sum()); b[1] += int(cx.sum())
                for pat in names:
                    m = masks[pat][0] & ok
                    if m.any():
                        sig.setdefault((pat, y, hd), []).extend(r[m].tolist())
            # the 2:1 stop/target variant
            for pat in names:
                mk, nb = masks[pat]
                bullish = pat in BULL
                for t in np.flatnonzero(mk & iny):
                    lo = np.nanmin(l[max(0, t - nb + 1): t + 1]) if bullish else None
                    hi = np.nanmax(h[max(0, t - nb + 1): t + 1]) if not bullish else None
                    g = stop_target(o, h, l, c, int(t), bullish, lo if bullish else hi)
                    if g is not None:
                        stp.setdefault((pat, y), []).append(g)
            # ordinary days in the same situation, for the 2:1 comparison (a sample, same stop rule: day's extreme)
            for direction, ctx in (("long", fell), ("short", rose)):
                cand = np.flatnonzero(ctx & iny)
                if len(cand):
                    pick = rng.choice(cand, size=min(len(cand), 6), replace=False)
                    for t in pick:
                        stop = l[t] if direction == "long" else h[t]
                        g = stop_target(o, h, l, c, int(t), direction == "long", stop)
                        if g is not None:
                            stp_pool.setdefault((direction, y), []).append(g)
    return summarize(sig, stp, base, stp_pool, years)


def _stats(xs: list[float]) -> dict:
    a = np.asarray(xs, float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "avg": float(a.mean() * 100), "median": float(np.median(a) * 100),
            "win": float((a > 0).mean()), "std": float(a.std() * 100)}


def summarize(sig, stp, base, stp_pool, years) -> dict:
    out = {"cost_round_trip_pct": COST * 100, "years": years, "patterns": {}}
    for pat in BULL + BEAR:
        bullish = pat in BULL
        direction = "long" if bullish else "short"
        sign = 1.0 if bullish else -1.0
        res = {"direction": direction, "holds": {}, "stop_target": {}}
        for hd in HOLDS:
            per_year, pooled_net, pooled_edge = {}, [], []
            for y in years:
                gross = sig.get((pat, y, hd), [])
                if not gross:
                    continue
                net = [sign * g - COST for g in gross]
                bsum, bn = base.get((direction, y, hd, "ctx"), [0.0, 0])
                ball, bnn = base.get((direction, y, hd, "any"), [0.0, 0])
                b_ctx = (sign * bsum / bn - COST) if bn else float("nan")
                b_any = (sign * ball / bnn - COST) if bnn else float("nan")
                st = _stats(net)
                st.update(edge_ctx=st["avg"] - b_ctx * 100, base_ctx=b_ctx * 100, base_any=b_any * 100)
                per_year[str(y)] = st
                pooled_net += net
                pooled_edge += [x - b_ctx for x in net] if not math.isnan(b_ctx) else []
            ys = [v for v in per_year.values() if v["n"] >= MIN_TRADES_YEAR]
            tot = _stats(pooled_net)
            if pooled_edge:
                e = np.asarray(pooled_edge)
                tot["edge_ctx"] = float(e.mean() * 100)
                tot["t_edge"] = float(e.mean() / (e.std() / math.sqrt(len(e)))) if e.std() > 0 else 0.0
            tot["years_judged"] = len(ys)
            tot["years_profit"] = sum(1 for v in ys if v["avg"] > 0)
            tot["years_edge"] = sum(1 for v in ys if v["edge_ctx"] > 0)
            res["holds"][str(hd)] = {"total": tot, "by_year": per_year}
        # 2:1 stop/target
        per_year, pooled = {}, []
        for y in years:
            g = stp.get((pat, y), [])
            if not g:
                continue
            net = [sign * 0 + x - COST for x in g]          # stop_target already returns the trade's own-direction return
            per_year[str(y)] = _stats(net)
            pooled += net
        ys = [v for v in per_year.values() if v["n"] >= MIN_TRADES_YEAR]
        tot = _stats(pooled)
        tot["years_judged"] = len(ys)
        tot["years_profit"] = sum(1 for v in ys if v["avg"] > 0)
        poolr = []
        for y in years:
            poolr += [x - COST for x in stp_pool.get((direction, y), [])]
        tot["ordinary_days_avg"] = float(np.mean(poolr) * 100) if poolr else None
        res["stop_target"] = {"total": tot, "by_year": per_year}
        out["patterns"][pat] = res
    out["verdict"] = verdicts(out)
    return out


def verdicts(out: dict) -> dict:
    """One line per pattern: works / mixed / does not work, from its best holding period and the 2:1 rule."""
    v = {}
    for pat, res in out["patterns"].items():
        best = None
        for hd, h in res["holds"].items():
            t = h["total"]
            if t.get("n", 0) < 100 or "t_edge" not in t:
                continue
            score = t["avg"]
            if best is None or score > best[1]["avg"]:
                best = (hd, t)
        if best is None:
            v[pat] = {"verdict": "not enough trades"}
            continue
        hd, t = best
        yj = max(t["years_judged"], 1)
        works = (t["avg"] > 0 and t["years_profit"] >= 0.7 * yj and t["years_edge"] >= 0.8 * yj
                 and t["t_edge"] > 3)      # strict on purpose: 16 patterns x 4 holds, some would pass by chance
        mixed = t["avg"] > 0 and (t["years_profit"] >= 0.55 * yj) and not works
        v[pat] = {"verdict": "works" if works else "mixed" if mixed else "does not work", "best_hold_days": int(hd),
                  "avg_net_pct": round(t["avg"], 3), "win_rate": round(t["win"], 3), "trades": t["n"],
                  "years_profit": f"{t['years_profit']}/{yj}", "years_beating_ordinary_days": f"{t['years_edge']}/{yj}",
                  "edge_vs_ordinary_days_pct": round(t["edge_ctx"], 3), "t_edge": round(t["t_edge"], 2)}
    return v


# ---------- data ----------

def fetch_bars(symbols: list[str], start, end, batch: int = 100) -> dict[str, pd.DataFrame]:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from .config import load_config

    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    out = {}
    for k in range(0, len(symbols), batch):
        try:
            df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols[k:k + batch], timeframe=TimeFrame.Day,
                                                        start=start, end=end, adjustment=Adjustment.ALL,
                                                        feed=DataFeed.SIP)).df
        except Exception as e:
            log.warning("batch %d failed: %s", k, e)
            continue
        if df.empty:
            continue
        df = df[["open", "high", "low", "close", "volume"]].astype("float64")
        for sym, d in df.groupby(level="symbol"):
            d = d.droplevel("symbol")
            d.index = d.index.tz_convert("America/New_York").normalize().tz_localize(None)
            out[sym] = d
        log.info("bars: %d/%d symbols", min(k + batch, len(symbols)), len(symbols))
    return out


def load_members(path: str = "research/lab.npz") -> tuple[dict[int, set[str]], list[int]]:
    z = np.load(path, allow_pickle=False)
    syms, years, member = list(z["symbols"]), [int(y) for y in z["years"]], z["member"]
    etfs = set(z["etfs"])
    return {y: {s for s, m in zip(syms, member[i]) if m and s not in etfs} for i, y in enumerate(years)}, years


def main() -> None:
    from datetime import datetime, timedelta, timezone

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data/candle_years.json")
    ap.add_argument("--max-symbols", type=int, default=0, help="for a quick trial")
    args = ap.parse_args()
    member, years = load_members()
    symbols = sorted({s for v in member.values() for s in v})
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
        member = {y: {s for s in v if s in set(symbols)} for y, v in member.items()}
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    bars = fetch_bars(symbols, pd.Timestamp("2015-06-01", tz="UTC"), end)
    log.info("downloaded %d symbols", len(bars))
    res = analyze(bars, member, years)
    res["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M+00:00")
    res["symbols"] = len(bars)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for pat, v in res["verdict"].items():
        log.info("%-22s %s", pat, v)


if __name__ == "__main__":
    main()
