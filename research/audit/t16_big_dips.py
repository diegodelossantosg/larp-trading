"""T16: an extra agent that buys BIG DIPS (no news needed), on top of SPY. Does it beat the S&P 500?

Run from the repo root:  python -m research.audit.t16_big_dips

Dip rules (signal at the close of day t, buy the next open, sell at the close `hold` days after the signal):
  1-day fall of 5 / 7 / 10 % ; 1-day fall of 5 / 7 % MORE than SPY ; 5-day fall of 10 / 15 / 20 %.
Universe: each year's ~500 most-traded stocks (research/lab.npz, delisted included), price >= $5, 20-day average
dollar volume >= $20M. Result of a trade = stock return minus SPY return over the same days, minus 0.1% costs
(the rest of the money sits in SPY, so this is exactly what the extra agent adds). Portfolio: 10% per buy, at most
2 buys a day (biggest fall first), at most 30% in dips, no stop. Design years 2016-2020, validation 2021 on.
"""
import itertools, numpy as np, pandas as pd
from trader.lab import Lab

COST, SIZE, CAP, PER_DAY = 0.001, 0.10, 0.30, 2
L = Lab()
o, c, v = L.open.astype("float64"), L.close.astype("float64"), L.volume.astype("float64")
n_days, n_sym = c.shape
spy = L.col["SPY"]
yrs = np.array([d.year for d in L.dates])
ret1 = np.full_like(c, np.nan); ret1[1:] = c[1:] / c[:-1] - 1
ret5 = np.full_like(c, np.nan); ret5[5:] = c[5:] / c[:-5] - 1
spy1 = ret1[:, [spy]]
dvol = pd.DataFrame(c * v).rolling(20).mean().to_numpy()
member = np.zeros_like(c, bool)
for i, y in enumerate(L.years):
    member[yrs == y] = L.member[i][None, :]
ok = member & (c >= 5) & (dvol >= 20e6)
ok[:, [spy]] = False
RULES = {"1-day fall 5%+": (ret1 <= -0.05), "1-day fall 7%+": (ret1 <= -0.07), "1-day fall 10%+": (ret1 <= -0.10),
         "1-day fall 5%+ more than SPY": (ret1 - spy1 <= -0.05), "1-day fall 7%+ more than SPY": (ret1 - spy1 <= -0.07),
         "5-day fall 10%+": (ret5 <= -0.10), "5-day fall 15%+": (ret5 <= -0.15), "5-day fall 20%+": (ret5 <= -0.20)}
DEPTH = {"1-day fall 5%+": ret1, "1-day fall 7%+": ret1, "1-day fall 10%+": ret1, "1-day fall 5%+ more than SPY": ret1 - spy1,
         "1-day fall 7%+ more than SPY": ret1 - spy1, "5-day fall 10%+": ret5, "5-day fall 15%+": ret5, "5-day fall 20%+": ret5}


def excess(hold):
    ex = np.full_like(c, np.nan)
    sret = np.full(n_days, np.nan)
    sret[: n_days - hold] = c[hold:, spy] / o[1: n_days - hold + 1, spy] - 1
    r = np.full_like(c, np.nan)
    r[: n_days - hold] = c[hold:] / o[1: n_days - hold + 1] - 1
    return r - sret[:, None]


rows = []
for hold in (5, 10, 20):
    ex = excess(hold)
    allx = {p: np.nanmean(ex[ok & (yrs >= a)[:, None] & (yrs <= b)[:, None]]) - COST for p, (a, b) in
            {"design": (2016, 2020), "valid": (2021, 2026)}.items()}
    for name, mask in RULES.items():
        sig = ok & mask & np.isfinite(ex)
        res = {}
        for per, (a, b) in {"design": (2016, 2020), "valid": (2021, 2026)}.items():
            sel = sig & ((yrs >= a) & (yrs <= b))[:, None]
            x = ex[sel] - COST
            # portfolio: biggest fall first, 2 a day, 30% cap
            open_pos, pnl, by_year, trades = [], 0.0, {}, 0
            for t in np.flatnonzero(((yrs >= a) & (yrs <= b))):
                open_pos = [p for p in open_pos if p > t]
                cols = np.flatnonzero(sig[t])
                if not len(cols):
                    continue
                cols = cols[np.argsort(DEPTH[name][t, cols])][:PER_DAY]
                for j in cols:
                    if (len(open_pos) + 1) * SIZE > CAP + 1e-9:
                        break
                    open_pos.append(t + hold)
                    g = SIZE * (ex[t, j] - COST)
                    pnl += g; trades += 1; by_year[yrs[t]] = by_year.get(yrs[t], 0.0) + g
            nyears = b - a + 1 if per == "design" else (yrs.max() - 2021 + 1)
            res[per] = dict(n=len(x), avg=x.mean() * 100 if len(x) else np.nan,
                            t=(x.mean() / (x.std() / len(x) ** .5)) if len(x) > 2 and x.std() > 0 else 0.0,
                            trades=trades, alpha_yr=pnl / nyears * 100, years_pos=sum(1 for g in by_year.values() if g > 0),
                            years=len(by_year), market=allx[per] * 100)
        rows.append(dict(rule=name, hold=hold, **{f"{p}_{k}": v for p, d in res.items() for k, v in d.items()}))
out = pd.DataFrame(rows)
out.to_csv("research/audit/out_t16.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
show = out[["rule", "hold", "design_n", "design_avg", "valid_n", "valid_avg", "valid_t", "valid_alpha_yr", "valid_years_pos", "valid_years", "valid_market"]].round(2)
print(show.to_string(index=False))
