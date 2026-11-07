"""One easy table: today's method vs the alternatives, same data (2023-07 to 2026-10, the window with news)."""
import numpy as np, pandas as pd
import research.audit.sim as _sim
_sim.idle_returns = lambda L: _sim.spy_returns(L)      # idle money in plain SPY (no 2x fund): isolates real skill
src = open("research/audit/run_combined.py").read().split("NEWS = {")[0]
exec(src)
def ideas(stop, group_avg=True):
    rs, invs = [], []
    for g in "ABC":
        r, inv, _ = sleeve(L, idle_np, group=g, kind="stocks", n=10, cadence=21, stop=stop); rs.append(r); invs.append(inv)
    return sum(rs) / 3, sum(invs) / 3
S = {"atr4": ("atr", 4.0), "f6": ("fixed", .06), "f20": ("fixed", .20), "none": None}
IDEAS = {k: ideas(v) for k, v in S.items()}
N = {"today": news_series(300, .02, 2, .05, 10, "s")[0], "n75": news_series(300, .02, 2, .075, 10, "r")[0],
     "n10": news_series(300, .02, 2, .10, 10, "r")[0], "more": news_series(20, .02, 2, .05, 10, "r")[0],
     "off": None, "buys5": news_series(300, .02, 5, .05, 10, "s")[0]}
METHODS = [  # name, ideas weight, stop key, news key
 ("1. HOY: 50% ideas, stops actuales, noticias 5%", .5, "atr4", "today"),
 ("2. Solo el stop de ETFs al 6% (como hoy)", .5, "f6", "today"),
 ("3. Stop ancho 20%", .5, "f20", "today"),
 ("4. Sin stops", .5, "none", "today"),
 ("5. 30% en ideas", .3, "atr4", "today"),
 ("6. 80% en ideas", .8, "atr4", "today"),
 ("7. 0% en ideas (todo SPY + noticias)", 0.0, "atr4", "today"),
 ("8. Noticias: tamaño 7,5%", .5, "atr4", "n75"),
 ("9. Noticias: tamaño 10%", .5, "atr4", "n10"),
 ("10. Noticias: más empresas (suelo 20 M$)", .5, "atr4", "more"),
 ("11. Noticias: 5 compras/día", .5, "atr4", "buys5"),
 ("12. Sin News Hunter", .5, "atr4", "off"),
 ("13. RECOMENDADO: 30% ideas + stop 20% + noticias 7,5%", .3, "f20", "n75"),
]
rows = []
for name, w, sk, nk in METHODS:
    for skill in (0.0, 0.05):
        r, inv = IDEAS[sk]
        p = combine(w, r, idle, skill, inv)
        if N[nk] is not None: p = p + N[nk]
        s = stats(p, spy, START_W, END_W)
        rows.append({"metodo": name, "habilidad_claude": skill, "gana_al_ano": round(s["cagr"], 1), "vs_SP500": round(s["excess"], 1), "peor_caida": round(s["mdd"], 1), "anos_ganando": s["yrs_beat"]})
pd.DataFrame(rows).to_csv("research/audit/out_table_spy.csv", index=False)
s0 = stats(spy, spy, START_W, END_W); print("S&P 500:", round(s0["cagr"], 1), round(s0["mdd"], 1))
