"""T8: strict protocol on the capitulation rule (leak-free). Pick the best settings on group A only, then judge them on B and C (never used to choose)."""
import itertools, numpy as np, pandas as pd
src = open("research/audit/t7_clean.py").read().split("rows = []\nfor g in")[0]
exec(src)
FW[("close", 15)] = fwd(15)
BASE = {(g, h): np.nanmean(np.concatenate([FW[("close", h)][t, L.universe(t, g) & ~isetf & ~np.isnan(raw[t])] for t in range(START, T - h - 2, 5)])) - COST for g in "ABC" for h in (5, 10, 15)}
yrs = (T - START) / 252
rows = []
for drop, vmin, dv, h in itertools.product((.015, .02, .03, .04, .05), (1.0, 1.5, 2.0, 3.0), (300, 500, 1000, 2500), (5, 10, 15)):
    for g in "ABC":
        s, n = sleeve_series(g, drop, vmin, dv, h, "close")
        st = S.stats(spy_s + s, spy_s, "2017-01-01", None)
        rows.append({"drop": drop, "vol": vmin, "dv": dv, "hold": h, "group": g, "trades_yr": round(n / yrs, 1), "vs_SP500": round(st["excess"], 2)})
d = pd.DataFrame(rows); d.to_csv("research/audit/out_t8.csv", index=False)
w = d.pivot_table(index=["drop", "vol", "dv", "hold"], columns="group", values=["vs_SP500", "trades_yr"])
ex = w["vs_SP500"]; tr = w["trades_yr"]
ok = tr["A"] >= 12
print("configs:", len(w), "| rank correlation of the excess: A->B %.2f, A->C %.2f, B->C %.2f" % (ex.A.rank().corr(ex.B.rank()), ex.A.rank().corr(ex.C.rank()), ex.B.rank().corr(ex.C.rank())))
top = ex[ok].sort_values("A", ascending=False).head(10)
print("\nTop 10 chosen on A only (>=12 trades/yr):"); print(top.round(2).to_string())
print("\nthese 10 on the unseen groups: mean B %.2f, mean C %.2f | all configs: mean B %.2f, mean C %.2f" % (top.B.mean(), top.C.mean(), ex.B[ok].mean(), ex.C[ok].mean()))
best = ex[ok].sort_values("A", ascending=False).index[0]
print("the single best on A:", best, "-> A %.2f, B %.2f, C %.2f" % tuple(ex.loc[best, ["A", "B", "C"]]))
print("share of configs with excess>0:  A %.2f  B %.2f  C %.2f  | all three: %.2f" % ((ex.A > 0).mean(), (ex.B > 0).mean(), (ex.C > 0).mean(), ((ex > 0).all(axis=1)).mean()))
