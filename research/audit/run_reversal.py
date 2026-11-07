"""Does buying a big one-day drop beat the S&P 500 over 10 years (2017-2026)? Design on group A, check on B, final C.

Pure price rule, no news needed: a stock falls `drop` more than SPY in one day, trades >= `dv` $M/day, and we hold `h` days.
Result of a trade = stock return minus SPY return over the hold, minus 0.1%. 'edge' subtracts the average of random days of the
same group (luck). Portfolio: 2 buys/day max, 7.5% each, at most 30% invested. Delisted stocks keep their last price.
"""
import itertools, time, numpy as np, pandas as pd
from trader.lab import Lab

L = Lab(); T, N = len(L.dates), len(L.symbols)
c = pd.DataFrame(L.close.astype("float64")).ffill().to_numpy()          # last price kept after a delisting
raw = L.close.astype("float64")
spy = c[:, L.col["SPY"]]
r1 = np.vstack([np.full((1, N), np.nan), raw[1:] / raw[:-1] - 1])
spy_r1 = np.r_[np.nan, spy[1:] / spy[:-1] - 1]
ex1 = r1 - spy_r1[:, None]                                              # one-day move relative to SPY
dv20 = pd.DataFrame(raw * L.volume.astype("float64")).rolling(20, min_periods=10).mean().to_numpy() / 1e6
sma50 = pd.DataFrame(raw).rolling(50, min_periods=40).mean().to_numpy()
sma200 = pd.DataFrame(raw).rolling(200, min_periods=150).mean().to_numpy()
spy_up = spy > pd.Series(spy).rolling(200, min_periods=150).mean().to_numpy()
ret20 = np.vstack([np.full((20, N), np.nan), raw[20:] / raw[:-20] - 1])
atr = pd.DataFrame(np.abs(r1)).rolling(14, min_periods=10).mean().to_numpy() * 1.6 * 100      # ~ATR% stand-in
isetf = np.isin(np.arange(N), [L.col[s] for s in L.etfs])
START, COST, SIZE, PER_DAY, CAP = 260, 0.001, 0.075, 2, 0.30

def fwd(h):
    out = np.full((T, N), np.nan)
    out[:T - h] = c[h:] / c[:T - h] - 1 - (spy[h:] / spy[:T - h] - 1)[:, None]
    return out

def run(group, drop, dv, h, regime, trend, fw):
    cand_days = []
    for t in range(START, T - h):
        m = L.universe(t, group) & ~isetf & ~np.isnan(raw[t]) & (dv20[t] >= dv) & (ex1[t] <= -drop)
        if regime == "spy_up": m &= bool(spy_up[t])
        if trend == "weak": m &= (raw[t] < sma50[t])
        elif trend == "weak_wild": m &= (raw[t] < sma50[t]) & (atr[t] >= 3.0)
        elif trend == "wild": m &= (atr[t] >= 3.0)
        elif trend == "falling20": m &= (ret20[t] < -0.05)
        idx = np.flatnonzero(m)
        if len(idx): cand_days.append((t, idx[np.argsort(ex1[t, idx])]))
    open_pos, net, gains, ys = [], [], 0.0, {}
    for t, idx in cand_days:
        open_pos = [p for p in open_pos if p[0] > t]
        exposure, held, took = len(open_pos) * SIZE, {p[1] for p in open_pos}, 0
        for j in idx:
            if took >= PER_DAY or exposure + SIZE > CAP + 1e-9: break
            r = fw[t, j]
            if j in held or np.isnan(r): continue
            open_pos.append((t + h, j)); held.add(j); exposure += SIZE; took += 1
            net.append(r - COST); gains += SIZE * (r - COST); y = L.dates[t + h].year; ys[y] = ys.get(y, 0) + SIZE * (r - COST)
    return np.array(net), gains, ys

rows, t0 = [], time.time()
FW = {h: fwd(h) for h in (5, 10, 20)}
for g in "ABC":
    for h in (5, 10, 20):
        pool = np.concatenate([FW[h][t, L.universe(t, g) & ~isetf & ~np.isnan(raw[t])] for t in range(START, T - h, 5)])
        base = np.nanmean(pool) - COST                                         # random days of the same group
        for drop, dv, regime, trend in itertools.product((.02, .03, .04, .06), (100, 300, 1000), ("any", "spy_up"), ("any", "weak", "falling20", "wild", "weak_wild")):
            net, gains, ys = run(g, drop, dv, h, regime, trend, FW[h])
            yrs = (T - START) / 252
            n = len(net)
            rows.append({"group": g, "drop": drop, "dv": dv, "hold": h, "regime": regime, "trend": trend, "trades": n, "trades_yr": round(n / yrs, 1),
                         "avg_net_pct": round(net.mean() * 100, 2) if n else np.nan, "base_pct": round(base * 100, 2),
                         "edge_pct": round((net.mean() - base) * 100, 2) if n else np.nan,
                         "t": round((net.mean() - base) / (net.std() / n ** .5), 1) if n > 10 and net.std() > 0 else np.nan,
                         "alpha_yr": round(gains / yrs * 100, 2), "years_pos": sum(v > 0 for v in ys.values()), "years": len(ys)})
        print(g, h, round(time.time() - t0), "s", flush=True)
pd.DataFrame(rows).to_csv("research/audit/out_reversal.csv", index=False)
