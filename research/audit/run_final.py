"""Final robustness check with idle money in plain SPY. Candidates x (news groups separately, news ventaja halved, Claude skill)."""
import numpy as np, pandas as pd
import research.audit.sim as _sim
_sim.idle_returns = lambda L: _sim.spy_returns(L)
src = open("research/audit/run_combined.py").read().split("NEWS = {")[0]
exec(src)

def news_by(sets, size, hold=10, dv=300, drop=.02, per_day=2, st="s"):
    sub = R.news[R.news["set"].isin(sets) & (R.news.move_1h <= -drop) & (R.news.dollar_vol_m >= dv)].sort_values(["i", "move_1h"])
    col, out, open_pos, n = f"hour_after_{st}{hold}", pd.Series(0.0, index=L.dates), [], 0
    for i, g in sub.groupby("i", sort=True):
        open_pos = [p for p in open_pos if p[0] > i]
        exposure, held, took = sum(p[1] for p in open_pos), {p[2] for p in open_pos}, 0
        for sym, r in zip(g.symbol.to_numpy(), g[col].to_numpy()):
            if took >= per_day or exposure + size > R.CAP + 1e-9: break
            if sym in held or np.isnan(r) or i + hold >= len(L.dates): continue
            open_pos.append((i + hold, size, sym)); held.add(sym); exposure += size; took += 1; n += 1
            out.iloc[i + hold] += size * (r - R.COST_RT)
    return out, n

r_i, inv_i = [], []
for g in "ABC":
    r, inv, _ = sleeve(L, idle_np, group=g, kind="stocks", n=10, cadence=21, stop=None); r_i.append(r); inv_i.append(inv)
ideas, inv = sum(r_i) / 3, sum(inv_i) / 3
CAND = [("S&P 500 puro (nada)", 0.0, 0.0), ("HOY: 50% ideas + noticias 5%", .5, .05), ("A: 0% ideas + noticias 5%", 0.0, .05),
        ("B: 0% ideas + noticias 7,5%", 0.0, .075), ("C: 10% ideas + noticias 7,5%", .1, .075), ("D: 30% ideas + noticias 7,5%", .3, .075)]
rows = []
for name, w, sz in CAND:
    for label, sets, haircut in (("todas las noticias", ("dev", "holdout"), 1.0), ("solo grupo dev", ("dev",), 1.0),
                                 ("solo grupo holdout", ("holdout",), 1.0), ("noticias con la ventaja a la mitad", ("dev", "holdout"), 0.5)):
        for skill in (0.0, 0.05):
            p = combine(w, ideas, idle, skill, inv)
            if sz:
                ns, n = news_by(sets, sz); p = p + ns * haircut
            s = stats(p, spy, START_W, END_W)
            rows.append({"candidato": name, "prueba": label, "habilidad_claude": skill, "vs_SP500": round(s["excess"], 1), "peor_caida": round(s["mdd"], 1)})
pd.DataFrame(rows).to_csv("research/audit/out_final.csv", index=False)

# decade view of the ideas share with idle=SPY (no news data before 2023): what the ideas cost or earn over 2017-2026
dec = []
for w in (0, .1, .2, .3, .5):
    for skill in (0.0, 0.05):
        s = stats(combine(w, ideas, idle, skill, inv), spy, "2017-01-01", None)
        dec.append({"w": w, "habilidad": skill, "vs_SP500": round(s["excess"], 1), "peor_caida": round(s["mdd"], 1), "anos_ganando": s["yrs_beat"]})
pd.DataFrame(dec).to_csv("research/audit/out_final_decade.csv", index=False)
