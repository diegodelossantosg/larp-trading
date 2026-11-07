"""T5: SPY + the capitulation-rebound sleeve as a full portfolio over 2017-2026, per group (each group = a separate ~330-company universe)."""
import itertools, numpy as np, pandas as pd
src = open("research/audit/t4_capitulation.py").read().split("\n# 1) neighbourhood")[0]
exec(src)
import research.audit.sim as S
spy_s = S.spy_returns(L)
def sleeve_series(group, drop, vmin, dv, h, entry, size=SIZE, cost=COST, per_day=PER_DAY, cap=CAP):
    fw = FW[(entry, h)]
    out = np.zeros(T); n = 0
    cand = []
    for t in range(START, T - h - 1):
        m = L.universe(t, group) & ~isetf & ~np.isnan(raw[t]) & (dv20[t] >= dv) & (vr[t] >= vmin) & (ex1[t] <= -drop)
        idx = np.flatnonzero(m)
        if len(idx): cand.append((t, idx[np.argsort(ex1[t, idx])]))
    open_pos = []
    for t, idx in cand:
        open_pos = [p for p in open_pos if p[0] > t]; exposure = len(open_pos) * size; held = {p[1] for p in open_pos}; took = 0
        for j in idx:
            if took >= per_day or exposure + size > cap + 1e-9: break
            r = fw[t, j]
            if j in held or np.isnan(r): continue
            open_pos.append((t + h, j)); held.add(j); exposure += size; took += 1; n += 1
            out[t + h] += size * (r - cost)
    return pd.Series(out, index=L.dates), n
rows = []
variants = {"centre: drop2% vol2x dv1000 hold10 close": (.02, 2.0, 1000, 10, "close", SIZE, COST),
            "same, size 15%": (.02, 2.0, 1000, 10, "close", .15, COST),
            "same, costs 0.3%": (.02, 2.0, 1000, 10, "close", SIZE, .003),
            "same, costs 0.6%": (.02, 2.0, 1000, 10, "close", SIZE, .006),
            "next-open entry": (.02, 2.0, 1000, 10, "next_open", SIZE, COST),
            "looser: vol1.5x dv700": (.02, 1.5, 700, 10, "close", SIZE, COST),
            "hold 5": (.02, 2.0, 1000, 5, "close", SIZE, COST)}
for name, (drop, vmin, dv, h, entry, size, cost) in variants.items():
    for g in "ABC":
        s, n = sleeve_series(g, drop, vmin, dv, h, entry, size, cost)
        p = spy_s + s
        st = S.stats(p, spy_s, "2017-01-01", None)
        rows.append({"variant": name, "group": g, "trades_yr": round(n / ((T - START) / 252), 1), "vs_SP500": round(st["excess"], 2), "peor_caida": round(st["mdd"], 1),
                     "spy_mdd": round(st["spy_mdd"], 1), "yrs_beat": st["yrs_beat"], "worst_year_vs_spy": round(st["worst_year_excess"], 1)})
d = pd.DataFrame(rows); d.to_csv("research/audit/out_t5.csv", index=False)
pd.set_option("display.width", 200)
print(d.pivot_table(index="variant", columns="group", values="vs_SP500", sort=False).round(2).assign(mean=lambda x: x.mean(axis=1).round(2)).to_string())
print("\nyears beating the S&P and worst drop (centre rule):")
print(d[d.variant.str.startswith("centre")][["group", "trades_yr", "vs_SP500", "peor_caida", "spy_mdd", "yrs_beat", "worst_year_vs_spy"]].to_string(index=False))
