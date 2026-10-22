"""Strategy lab: test many ideas on research/lab.npz (see lab_data) without fooling ourselves.

Stock-picking ideas are run on each symbol group separately: A (design), B (validation) and
C (final check). A rule only counts if it beats the S&P 500 AND an equal-weight basket of the same
group's universe, net of costs, in A, then B, then (once) C, in most years.

Timing ideas for the idle asset (SSO) are judged on three markets: S&P 500 (SPY/SSO), Nasdaq 100
(QQQ/QLD) and small caps (IWM/UWM); a rule designed on one must also help the others.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

COST = 0.001   # per side


class Lab:
    def __init__(self, path: str = "research/lab.npz"):
        z = np.load(path, allow_pickle=False)
        self.dates = pd.DatetimeIndex(z["dates"])
        self.symbols = list(z["symbols"])
        self.col = {s: j for j, s in enumerate(self.symbols)}
        self.groups = np.array(z["groups"])
        self.years = list(z["years"])
        self.member = z["member"]
        self.open, self.close, self.low, self.volume = z["open"], z["close"], z["low"], z["volume"]
        c = self.close.astype("float64")
        prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
        self.ret = c / prev - 1
        self.etfs = set(z["etfs"])

    def series(self, sym: str) -> pd.Series:
        return pd.Series(self.close[:, self.col[sym]].astype("float64"), index=self.dates)

    def universe(self, day_i: int, group: str | None) -> np.ndarray:
        y = self.dates[day_i].year
        if y not in self.years:
            return np.zeros(len(self.symbols), bool)
        m = self.member[self.years.index(y)].copy()
        if group:
            m &= self.groups == group
        return m


# ---------- stock-picking scores (higher = buy); all use data up to and including day t ----------

def _r(c, t, a, b=0):
    with np.errstate(invalid="ignore", divide="ignore"):
        return c[t - b] / c[t - a] - 1


def score(lab: Lab, name: str, t: int) -> np.ndarray:
    c = lab.close.astype("float64")
    if t < 260:
        return np.full(c.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vol60 = np.nanstd(lab.ret[t - 59:t + 1], axis=0) * np.sqrt(252)
        sma200 = np.nanmean(c[t - 199:t + 1], axis=0)
        hi252 = np.nanmax(c[t - 251:t + 1], axis=0)
        if name == "mom_12_1":
            return _r(c, t, 252, 21)
        if name == "mom_6_1":
            return _r(c, t, 126, 21)
        if name == "mom_3":
            return _r(c, t, 63)
        if name == "mom_12_1_risk_adj":
            return _r(c, t, 252, 21) / vol60
        if name == "near_52w_high":
            return c[t] / hi252
        if name == "low_vol":
            return -vol60
        if name == "reversal_1m":
            return -_r(c, t, 21)
        if name == "dip_in_uptrend":
            s = -_r(c, t, 10)
            return np.where(c[t] > sma200, s, np.nan)
        if name == "mom_and_low_vol":
            a, b = _r(c, t, 252, 21), -vol60
            return pd.Series(a).rank(pct=True).to_numpy() + pd.Series(b).rank(pct=True).to_numpy()
        if name == "trend_up_low_vol":
            return np.where(c[t] > sma200, -vol60, np.nan)
    raise KeyError(name)


SCORES = ["mom_12_1", "mom_6_1", "mom_3", "mom_12_1_risk_adj", "near_52w_high", "low_vol", "reversal_1m",
          "dip_in_uptrend", "mom_and_low_vol", "trend_up_low_vol"]


def month_ends(lab: Lab) -> list[int]:
    m = lab.dates.to_period("M")
    return [i for i in range(len(m) - 1) if m[i] != m[i + 1]]


def pick_backtest(lab: Lab, score_name: str, group: str, n: int = 10, stop: float | None = None,
                  every: int = 1) -> pd.Series:
    """Daily returns of: every `every` month-ends buy the top-n by score in that group's universe,
    equal weight, hold until the next rebalance. Optional trailing stop on daily lows; a stopped or
    delisted name turns into cash until the next rebalance. Costs on everything traded."""
    reb = set(month_ends(lab)[::every])
    c = lab.close.astype("float64")
    out = np.zeros(len(lab.dates))
    held = np.array([], int)
    val = np.array([])        # value of each holding (portfolio starts each rebalance at 1.0)
    cash, peak = 0.0, np.array([])
    for t in range(1, len(lab.dates)):
        if len(held):
            before = val.sum() + cash
            r = lab.ret[t, held]
            r = np.where(np.isnan(r) | (val == 0), 0.0, r)
            if stop is not None:
                prev = c[t - 1, held]
                lvl = peak * (1 - stop)
                hit = (val > 0) & (lab.low[t, held] <= lvl) & ~np.isnan(prev)
                r = np.where(hit, np.minimum(lvl, lab.open[t, held]) / prev - 1, r)
                val = val * (1 + r)
                cash += float((val[hit] * (1 - COST)).sum())
                val = np.where(hit, 0.0, val)
                peak = np.fmax(peak, c[t, held])
            else:
                val = val * (1 + r)
            dead = np.isnan(c[t, held]) & (val > 0)
            cash += float(val[dead].sum())
            val = np.where(dead, 0.0, val)
            out[t] = (val.sum() + cash) / before - 1
        if t in reb:
            s = score(lab, score_name, t)
            u = lab.universe(t, group) & ~np.isnan(s) & ~np.isnan(c[t])
            idx = np.flatnonzero(u)
            if len(idx) >= n:
                top = idx[np.argsort(-s[idx])[:n]]
                total = val.sum() + cash if len(held) else 1.0
                cur = {j: v / total for j, v in zip(held.tolist(), val.tolist())}
                new = {int(j): 1.0 / n for j in top}
                turnover = sum(abs(new.get(j, 0) - cur.get(j, 0)) for j in set(new) | set(cur))
                out[t] = (1 + out[t]) * (1 - COST * turnover) - 1
                held = np.array(sorted(new))
                val, cash = np.full(n, 1.0 / n), 0.0
                peak = c[t, held].copy()
    return pd.Series(out, index=lab.dates)


def equal_weight(lab: Lab, group: str) -> pd.Series:
    """The whole group universe, equal weight, rebalanced monthly (what 'picking' must beat)."""
    out = np.zeros(len(lab.dates))
    members = None
    ends = set(month_ends(lab))
    for t in range(1, len(lab.dates)):
        if members is not None:
            r = lab.ret[t, members]
            out[t] = np.nanmean(np.where(np.isnan(r), 0.0, r))
        if t in ends:
            m = lab.universe(t, group) & ~np.isnan(lab.close[t])
            members = np.flatnonzero(m) if m.any() else None
    return pd.Series(out, index=lab.dates)


def stats(r: pd.Series, bench: pd.Series, start: str = "2017-01-01") -> dict:
    r, bench = r.loc[start:], bench.loc[start:]
    eq = (1 + r).cumprod()
    years = len(r) / 252
    yr = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1)
    by = bench.groupby(bench.index.year).apply(lambda g: (1 + g).prod() - 1)
    beq = (1 + bench).cumprod()
    return {"yearly_pct": round((eq.iloc[-1] ** (1 / years) - 1) * 100, 1),
            "bench_yearly_pct": round((beq.iloc[-1] ** (1 / years) - 1) * 100, 1),
            "worst_drop_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1),
            "years_beating": f"{int((yr > by).sum())}/{len(yr)}",
            "by_year_excess_pct": {int(y): round(float(yr[y] - by[y]) * 100, 1) for y in yr.index}}


# ---------- timing rules for the idle asset: hold the 2x fund (lev) or the plain one (base) ----------

TIMING = ["always 2x", "2x when above 200d avg", "2x above 200d, else cash", "2x when calm (20d vol < 20%)",
          "2x above 200d and calm", "2x on turn of month (else 1x)", "2x when above 50d avg", "always 1x"]


def timing(lab: Lab, base: str, lev: str, rule: str) -> pd.Series:
    """Daily returns; decisions use yesterday's close, switching costs COST each time."""
    b = lab.series(base)
    rb, rl = b.pct_change().fillna(0), lab.series(lev).pct_change().fillna(0)
    above200 = (b > b.rolling(200).mean()).shift(1, fill_value=False)
    above50 = (b > b.rolling(50).mean()).shift(1, fill_value=False)
    calm = (rb.rolling(20).std() * np.sqrt(252) < 0.20).shift(1, fill_value=False)
    m = b.index.to_period("M")
    pos_in_month = pd.Series(range(len(b)), index=b.index).groupby(m).cumcount()
    left_in_month = pd.Series(range(len(b)), index=b.index).groupby(m).cumcount(ascending=False)
    tom = (pos_in_month <= 2) | (left_in_month == 0)
    lev_on = {"always 2x": pd.Series(True, index=b.index), "2x when above 200d avg": above200,
              "2x above 200d, else cash": above200, "2x when calm (20d vol < 20%)": calm,
              "2x above 200d and calm": above200 & calm, "2x on turn of month (else 1x)": tom,
              "2x when above 50d avg": above50, "always 1x": pd.Series(False, index=b.index)}[rule]
    other = pd.Series(0.0, index=b.index) if rule == "2x above 200d, else cash" else rb
    r = np.where(lev_on, rl, other)
    switches = lev_on.astype(int).diff().abs().fillna(0)
    return pd.Series(r, index=b.index) - switches * COST * 2
