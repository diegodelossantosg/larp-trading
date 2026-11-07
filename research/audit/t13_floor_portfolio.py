"""T13: portfolio (idle in SPY) with volatile stocks only, liquidity floor 300 / 600 / 1000 M$ and sizes, dev and holdout separately + edge halved."""
import numpy as np, pandas as pd
import research.audit.sim as _sim
_sim.idle_returns = lambda L: _sim.spy_returns(L)
src = open("research/audit/run_final2.py").read().split("CAND = [")[0]
exec(src)
rows = []
for dv in (300, 600, 1000):
    for size in (.10, .15, .20):
        for per_day in (2, 3):
            for label, sets, hc in (("todas", ("dev", "holdout"), 1.0), ("solo dev", ("dev",), 1.0), ("solo holdout", ("holdout",), 1.0), ("ventaja a la mitad", ("dev", "holdout"), 0.5)):
                ns, n = news_f(sets, "wild", size, 10, dv, per_day, "r")
                s = stats(idle + ns * hc, spy, START_W, END_W)
                rows.append({"dv": dv, "size": size, "per_day": per_day, "prueba": label, "vs_SP500": round(s["excess"], 1), "mdd": round(s["mdd"], 1), "compras": n})
d = pd.DataFrame(rows); d.to_csv("research/audit/out_t13.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
p = d.pivot_table(index=["dv", "size", "per_day"], columns="prueba", values="vs_SP500", sort=False).round(1)
p["compras(todas)"] = d[d.prueba == "todas"].set_index(["dv", "size", "per_day"]).compras
p["peor_caida(todas)"] = d[d.prueba == "todas"].set_index(["dv", "size", "per_day"]).mdd
print(p[["todas", "solo dev", "solo holdout", "ventaja a la mitad", "compras(todas)", "peor_caida(todas)"]].to_string())
