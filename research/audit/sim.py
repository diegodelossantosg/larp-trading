"""Portfolio audit simulator: idle asset + 'ideas' sleeve (Claude proxy, with the watcher's stops) + news sleeve.

Everything runs on research/lab.npz (daily bars 2016-2026, groups A design / B validation / C final) and
research/events.csv.gz (news events). Costs: 0.1% per side. See REPORT.md for what it can and cannot say.
Run from the repo root:  PYTHONPATH=. python research/audit/<script>.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader.lab import Lab, timing

COST = 0.001
LEVERAGED = {"SSO", "QLD", "UWM", "DDM"}
START = 260          # first trading day with enough history


def idle_returns(L: Lab) -> pd.Series:
    """SSO while SPY is above its 200-day average, SPY otherwise (the live rule)."""
    return timing(L, "SPY", "SSO", "2x when above 200d avg")


def spy_returns(L: Lab) -> pd.Series:
    return L.series("SPY").pct_change().fillna(0.0)


def swing_pct(L: Lab, t: int, cols: np.ndarray) -> np.ndarray:
    """Stand-in for 'normal daily swing' (ATR): 1.6 x the 20-day mean absolute daily move."""
    r = np.abs(np.nan_to_num(L.ret[t - 19:t + 1, cols]))
    return 1.6 * r.mean(axis=0)


def sleeve(L: Lab, idle: np.ndarray, *, group: str | None, kind: str, n: int = 10, seeds: int = 20,
           cadence: int = 21, stop: tuple | None = None, seed0: int = 0):
    """Random 'ideas' (zero skill): every `cadence` trading days pick n names at random, equal weight.

    kind: 'stocks' (the group's companies, as of each year) or 'etfs' (non-leveraged ETFs).
    stop: None, ('fixed', 0.06) or ('atr', 4.0)  [atr stop clamped to 8%-20%, like the watcher].
    A stopped (or delisted) slot sells and its money goes to the idle asset until the next rebalance.
    Returns (daily return averaged over seeds, share of the sleeve in ideas, number of stop-outs per year).
    """
    T, S = len(L.dates), seeds
    c = L.close.astype("float64")
    out = np.zeros(T)
    inv = np.zeros(T)
    stops_hit = 0
    rng = np.random.default_rng(1000 + seed0)
    etf_cols = np.array([L.col[s] for s in L.etfs if s not in LEVERAGED and s != "SPY" and s != "SHY"])
    state = None
    for t in range(START, T):
        if state is not None:
            val, inidea, peak, lvl_pct = state
            r_idle = idle[t]
            r = L.ret[t, cols_held]                                  # (S, n)
            r = np.where(np.isnan(r), 0.0, r)
            nan_close = np.isnan(c[t, cols_held])
            before = val.sum(axis=1)
            if stop is not None:
                lvl = peak * (1 - lvl_pct)
                prev = c[t - 1, cols_held]
                hit = inidea & (L.low[t, cols_held] <= lvl) & ~np.isnan(prev)
                r_hit = np.minimum(lvl, L.open[t, cols_held]) / prev - 1
                r = np.where(hit, r_hit, r)
            val = np.where(inidea, val * (1 + r), val * (1 + r_idle))
            if stop is not None:
                val = np.where(hit, val * (1 - 2 * COST), val)        # sell the stock, buy the idle asset
                stops_hit += int(hit.sum())
                inidea = inidea & ~hit
                peak = np.where(inidea, np.fmax(peak, c[t, cols_held]), peak)
            dead = inidea & nan_close
            inidea = inidea & ~dead
            val = np.where(dead, val * (1 - 2 * COST), val)
            out[t] = float(np.mean(val.sum(axis=1) / before - 1))
            inv[t] = float(inidea.mean())
            state = (val, inidea, peak, lvl_pct)
        if (t - START) % cadence == 0:
            if kind == "stocks":
                u = L.universe(t, group) & ~np.isnan(c[t]) & (L.volume[t] > 0)
                pool = np.flatnonzero(u & ~np.isin(np.arange(len(L.symbols)), list(map(L.col.get, L.etfs))))
            else:
                pool = etf_cols[~np.isnan(c[t, etf_cols])]
            if len(pool) < n:
                continue
            picks = np.stack([rng.choice(pool, n, replace=False) for _ in range(S)])
            cols_held = picks
            if state is None:
                val = np.full((S, n), 1.0 / n)
            else:
                val = state[0]
                tot = val.sum(axis=1, keepdims=True)
                val = np.full((S, n), 1.0) * tot / n
                out[t] = (1 + out[t]) * (1 - 2 * COST) - 1            # full turnover on the rebalance
            inidea = np.ones((S, n), bool)
            peak = c[t, picks].copy()
            if stop is None:
                lvl_pct = np.zeros((S, n))
            elif stop[0] == "fixed":
                lvl_pct = np.full((S, n), stop[1])
            else:
                sw = swing_pct(L, t, picks.ravel()).reshape(S, n)
                lvl_pct = np.clip(stop[1] * sw, 0.08, 0.20)
            state = (val, inidea, peak, lvl_pct)
    years = (T - START) / 252
    return pd.Series(out, index=L.dates), pd.Series(inv, index=L.dates), stops_hit / S / years


def combine(w: float, ideas: pd.Series, idle: pd.Series, alpha_yr: float = 0.0, invested: pd.Series | None = None,
            ) -> pd.Series:
    """Daily return of: w in the ideas sleeve (plus an assumed yearly skill `alpha_yr` on what is
    actually invested in ideas) and the rest in the idle asset."""
    a = ideas + (alpha_yr / 252) * (invested if invested is not None else 1.0)
    return w * a + (1 - w) * idle


def stats(r: pd.Series, bench: pd.Series, start: str = "2017-01-01", end: str | None = None) -> dict:
    r, bench = r.loc[start:end], bench.loc[start:end]
    eq, beq = (1 + r).cumprod(), (1 + bench).cumprod()
    yrs = len(r) / 252
    cagr, bcagr = eq.iloc[-1] ** (1 / yrs) - 1, beq.iloc[-1] ** (1 / yrs) - 1
    ry = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1)
    by = bench.groupby(bench.index.year).apply(lambda g: (1 + g).prod() - 1)
    return {"cagr": cagr * 100, "spy_cagr": bcagr * 100, "excess": (cagr - bcagr) * 100,
            "mdd": float((eq / eq.cummax() - 1).min()) * 100, "spy_mdd": float((beq / beq.cummax() - 1).min()) * 100,
            "yrs_beat": f"{int((ry > by).sum())}/{len(ry)}", "worst_year_excess": float((ry - by).min()) * 100}
