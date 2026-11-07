"""T7: leak check. Liquidity and the 50-day average now use data up to YESTERDAY only (the signal day's own volume no longer leaks into the filter)."""
import numpy as np, pandas as pd
src = open("research/audit/t5_capitulation_portfolio.py").read().split("\nrows = []\nvariants")[0]
exec(src)
dv20 = np.vstack([np.full((1, N), np.nan), dv20[:-1]])           # yesterday's 20-day dollar volume
sma50 = np.vstack([np.full((1, N), np.nan), sma50[:-1]])
rows = []
for g in "ABC":
    for name, (drop, vmin, dv) in {"centre drop2% vol2x dv1000": (.02, 2.0, 1000), "vol1.5x dv700": (.02, 1.5, 700), "drop3% vol2x dv1000": (.03, 2.0, 1000), "drop2% vol2x dv2500": (.02, 2.0, 2500)}.items():
        s, n = sleeve_series(g, drop, vmin, dv, 10, "close")
        st = S.stats(spy_s + s, spy_s, "2017-01-01", None)
        rows.append({"rule": name, "group": g, "trades_yr": round(n / ((T - START) / 252), 1), "vs_SP500": round(st["excess"], 2), "mdd": round(st["mdd"], 1), "yrs_beat": st["yrs_beat"]})
d = pd.DataFrame(rows); d.to_csv("research/audit/out_t7.csv", index=False)
print(d.pivot_table(index="rule", columns="group", values="vs_SP500", sort=False).round(2).assign(mean=lambda x: x.mean(axis=1).round(2)).to_string())
# union universe (S&P-500-like, ~500 most-traded firms per year, 3 groups together)
for per_day, cap in ((2, .30), (3, .40)):
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
    print(f"UNION ~500 firms/yr, {per_day}/day cap {cap:.0%}: trades/yr {n/((T-START)/252):.0f}, vs S&P {st['excess']:+.2f} pts/yr, worst drop {st['mdd']:.1f}% (S&P {st['spy_mdd']:.1f}%), years beating {st['yrs_beat']}, worst year vs S&P {st['worst_year_excess']:.1f}")
