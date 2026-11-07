"""T6: capitulation rebound in depth: edge by horizon, stops, sizing/caps, bigger universe, the opposite direction (big up days), ETFs."""
import itertools, numpy as np, pandas as pd
src = open("research/audit/t5_capitulation_portfolio.py").read().split("\nrows = []\nvariants")[0]
exec(src)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
yrs = (T - START) / 252
def signals(group, drop, vmin, dv, sign=-1):
    out = []
    for t in range(START, T - 31):
        u = L.universe(t, group) if group else (L.member[L.years.index(L.dates[t].year)] if L.dates[t].year in L.years else np.zeros(N, bool))
        m = u & ~isetf & ~np.isnan(raw[t]) & (dv20[t] >= dv) & (vr[t] >= vmin) & ((ex1[t] <= -drop) if sign < 0 else (ex1[t] >= drop))
        for j in np.flatnonzero(m): out.append((t, j))
    return out
# 1) horizon curve (all signals, pooled over A,B,C; entry at the close of the signal day), edge vs random days
curve = {}
for g in "ABC":
    sig = signals(g, .02, 2.0, 1000)
    for h in (1, 2, 3, 5, 7, 10, 15, 20, 30):
        fw = fwd(h)
        vals = np.array([fw[t, j] for t, j in sig if t + h < T]); vals = vals[~np.isnan(vals)]
        pool = np.concatenate([fw[t, L.universe(t, g) & ~isetf & ~np.isnan(raw[t])] for t in range(START, T - 31, 5)])
        curve.setdefault(h, []).append((vals.mean() - np.nanmean(pool), len(vals)))
print("EDGE vs random days after a capitulation signal, by days held (average of A,B,C, before costs):")
print(pd.DataFrame({h: [round(np.mean([x[0] for x in v]) * 100, 2)] for h, v in curve.items()}, index=["edge_%"]).to_string())
# 2) stops on the trades (daily lows) and the sizing/caps, in the full-portfolio frame (A,B,C average of excess vs SPY)
def series_stop(group, stop, h=10, size=SIZE, per_day=PER_DAY, cap=CAP, drop=.02, vmin=2.0, dv=1000, cost=COST):
    out = np.zeros(T); n = 0; open_pos = []
    for t in range(START, T - h - 1):
        m = L.universe(t, group) & ~isetf & ~np.isnan(raw[t]) & (dv20[t] >= dv) & (vr[t] >= vmin) & (ex1[t] <= -drop)
        idx = np.flatnonzero(m)
        if not len(idx): continue
        idx = idx[np.argsort(ex1[t, idx])]
        open_pos = [p for p in open_pos if p[0] > t]; exposure = len(open_pos) * size; held = {p[1] for p in open_pos}; took = 0
        for j in idx:
            if took >= per_day or exposure + size > cap + 1e-9: break
            if j in held: continue
            entry = c[t, j]; peak = entry; exit_px, exit_t = c[t + h, j], t + h
            for k in range(1, h + 1):
                if stop and L.low[t + k, j] <= peak * (1 - stop):
                    exit_px, exit_t = min(peak * (1 - stop), L.open[t + k, j]), t + k; break
                peak = max(peak, c[t + k, j])
            r = exit_px / entry - 1 - (spy[exit_t] / spy[t] - 1)
            if np.isnan(r): continue
            open_pos.append((t + h, j)); held.add(j); exposure += size; took += 1; n += 1
            out[exit_t] += size * (r - cost)
    return pd.Series(out, index=L.dates), n
res = []
configs = {"no stop (centre)": dict(stop=None), "stop 8%": dict(stop=.08), "stop 12%": dict(stop=.12), "stop 20%": dict(stop=.20),
           "size 5%": dict(stop=None, size=.05), "size 15%": dict(stop=None, size=.15), "size 20% cap 40%": dict(stop=None, size=.20, cap=.40),
           "3 buys/day, cap 40%": dict(stop=None, per_day=3, cap=.40), "1 buy/day": dict(stop=None, per_day=1)}
for name, kw in configs.items():
    ex = []
    for g in "ABC":
        s, n = series_stop(g, **kw); st = S.stats(spy_s + s, spy_s, "2017-01-01", None); ex.append((st["excess"], st["mdd"], n / yrs))
    res.append({"variant": name, "vs_SP500_mean": round(np.mean([e[0] for e in ex]), 2), "per_group": [round(e[0], 2) for e in ex], "mdd_mean": round(np.mean([e[1] for e in ex]), 1), "trades_yr": round(np.mean([e[2] for e in ex]), 1)})
print("\nSTOPS / SIZING (portfolio vs S&P, 2017-2026):"); print(pd.DataFrame(res).to_string(index=False))
# 3) bigger universe: the union of A+B+C (about 1,000 companies)
s, n = series_stop(None, None, per_day=3, cap=.40) if False else (None, None)
ex = []
for per_day, cap in ((2, .30), (3, .40), (5, .50)):
    out = np.zeros(T); open_pos = []; n = 0
    for t in range(START, T - 11):
        if L.dates[t].year not in L.years: continue
        m = L.member[L.years.index(L.dates[t].year)] & ~isetf & ~np.isnan(raw[t]) & (dv20[t] >= 1000) & (vr[t] >= 2.0) & (ex1[t] <= -.02)
        idx = np.flatnonzero(m)
        if not len(idx): continue
        idx = idx[np.argsort(ex1[t, idx])]
        open_pos = [p for p in open_pos if p[0] > t]; exposure = len(open_pos) * SIZE; held = {p[1] for p in open_pos}; took = 0
        for j in idx:
            if took >= per_day or exposure + SIZE > cap + 1e-9: break
            r = FW[("close", 10)][t, j]
            if j in held or np.isnan(r): continue
            open_pos.append((t + 10, j)); held.add(j); exposure += SIZE; took += 1; n += 1; out[t + 10] += SIZE * (r - COST)
    st = S.stats(spy_s + pd.Series(out, index=L.dates), spy_s, "2017-01-01", None)
    ex.append({"universe": "A+B+C (~1,000 firms)", "per_day": per_day, "cap": cap, "trades_yr": round(n / yrs, 1), "vs_SP500": round(st["excess"], 2), "mdd": round(st["mdd"], 1), "spy_mdd": round(st["spy_mdd"], 1), "yrs_beat": st["yrs_beat"]})
print("\nBIGGER UNIVERSE:"); print(pd.DataFrame(ex).to_string(index=False))
# 4) opposite direction: big UP days with a volume surge in very liquid names (momentum continuation?)
up = []
for g in "ABC":
    sig = signals(g, .02, 2.0, 1000, sign=+1)
    for h in (5, 10):
        fw = fwd(h); vals = np.array([fw[t, j] for t, j in sig if t + h < T]); vals = vals[~np.isnan(vals)]
        pool = np.concatenate([fw[t, L.universe(t, g) & ~isetf & ~np.isnan(raw[t])] for t in range(START, T - 31, 5)])
        up.append({"group": g, "hold": h, "signals": len(vals), "edge_pct": round((vals.mean() - np.nanmean(pool)) * 100, 2)})
print("\nBIG UP DAYS (+2% vs SPY, volume 2x, dv>=1000) edge vs random days, before costs:"); print(pd.DataFrame(up).pivot_table(index="hold", columns="group", values="edge_pct").to_string())
