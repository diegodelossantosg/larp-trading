"""Portfolio with idle money in SPY + News Hunter variants (filters + sizes), on dev / holdout separately, together, and with the edge halved."""
import numpy as np, pandas as pd
import research.audit.sim as _sim
_sim.idle_returns = lambda L: _sim.spy_returns(L)
src = open("research/audit/run_combined.py").read().split("NEWS = {")[0]
exec(src)
F = {"none": lambda d: d.move_1h.notna(), "wild": lambda d: d.atr_pct >= 3, "below50&wild": lambda d: (d.above_sma50 == 0) & (d.atr_pct >= 3),
     "below50": lambda d: d.above_sma50 == 0}
def news_f(sets, flt, size, hold=10, dv=300, per_day=2, st="r"):
    d = R.news[R.news["set"].isin(sets) & (R.news.move_1h <= -0.02) & (R.news.dollar_vol_m >= dv)]
    d = d[F[flt](d)].sort_values(["i", "move_1h"])
    col, out, open_pos, n = f"hour_after_{st}{hold}", pd.Series(0.0, index=L.dates), [], 0
    for i, g in d.groupby("i", sort=True):
        open_pos = [p for p in open_pos if p[0] > i]
        exposure, held, took = sum(p[1] for p in open_pos), {p[2] for p in open_pos}, 0
        for sym, r in zip(g.symbol.to_numpy(), g[col].to_numpy()):
            if took >= per_day or exposure + size > R.CAP + 1e-9: break
            if sym in held or np.isnan(r) or i + hold >= len(L.dates): continue
            open_pos.append((i + hold, size, sym)); held.add(sym); exposure += size; took += 1; n += 1
            out.iloc[i + hold] += size * (r - R.COST_RT)
    return out, n
CAND = [("ACTUAL: sin filtro, 7,5%, stop", "none", .075, 10, 300, 2, "s"), ("sin filtro, 7,5%, sin stop", "none", .075, 10, 300, 2, "r"),
        ("acciones movidas (ATR>=3%), 7,5%", "wild", .075, 10, 300, 2, "r"), ("acciones movidas, 10%", "wild", .10, 10, 300, 2, "r"),
        ("movidas y bajo su media 50d, 10%", "below50&wild", .10, 10, 300, 2, "r"), ("movidas y bajo su media 50d, 15%", "below50&wild", .15, 10, 300, 2, "r"),
        ("bajo su media de 50d, 10%", "below50", .10, 10, 300, 2, "r"), ("movidas, 10%, 3 compras/dia", "wild", .10, 10, 300, 3, "r")]
rows = []
for name, flt, size, h, dv, pd_, st in CAND:
    for label, sets, hc in (("todas", ("dev", "holdout"), 1.0), ("solo dev", ("dev",), 1.0), ("solo holdout", ("holdout",), 1.0), ("ventaja a la mitad", ("dev", "holdout"), 0.5)):
        ns, n = news_f(sets, flt, size, h, dv, pd_, st)
        s = stats(idle + ns * hc, spy, START_W, END_W)
        rows.append({"variante": name, "prueba": label, "vs_SP500": round(s["excess"], 1), "peor_caida": round(s["mdd"], 1), "compras": n})
out = pd.DataFrame(rows); out.to_csv("research/audit/out_final2.csv", index=False)
pd.set_option("display.width", 230)
print(out.pivot_table(index="variante", columns="prueba", values="vs_SP500", sort=False).round(1).to_string())
print(out[out.prueba == "todas"][["variante", "peor_caida", "compras"]].to_string(index=False))
