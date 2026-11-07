"""Ideas sleeve (Claude proxy + watcher stops): every combination, on groups A / B / C. Writes out_ideas.csv."""
import itertools, time, numpy as np, pandas as pd
from trader.lab import Lab
from research.audit.sim import *

L = Lab(); idle = idle_returns(L); spy = spy_returns(L); idle_np = idle.to_numpy()
STOPS = [None] + [("fixed", x) for x in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20)] + [("atr", k) for k in (2, 3, 4, 6)]
NS, CADS, WS, ALPHAS = (5, 10, 20), (5, 21, 63), (0.0, 0.1, 0.3, 0.5, 0.8, 1.0), (0.0, 0.02, 0.05, 0.10)
sname = lambda s: "none" if s is None else (f"fixed{int(s[1]*100)}" if s[0] == "fixed" else f"atr{s[1]:g}x")
rows = []
t0 = time.time()
for kind, group in [("stocks", "A"), ("stocks", "B"), ("stocks", "C"), ("etfs", None)]:
    for n, cad, stop in itertools.product(NS, CADS, STOPS):
        if kind == "etfs" and n == 20: continue
        r, inv, hits = sleeve(L, idle_np, group=group, kind=kind, n=n, cadence=cad, stop=stop)
        for w, a in itertools.product(WS, ALPHAS):
            p = combine(w, r, idle, a, inv)
            s = stats(p, spy)
            rows.append({"kind": kind, "group": group or "-", "n": n, "cadence": cad, "stop": sname(stop), "w": w, "alpha": a,
                         "stops_per_year": round(hits, 1), "invested": round(float(inv.loc["2017":].mean()), 2), **{k: s[k] for k in ("cagr", "excess", "mdd", "yrs_beat", "worst_year_excess")}})
    print(kind, group, round(time.time() - t0), "s", flush=True)
pd.DataFrame(rows).to_csv("research/audit/out_ideas.csv", index=False)
print(len(rows), "rows")
