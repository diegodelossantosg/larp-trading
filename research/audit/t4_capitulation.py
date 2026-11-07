"""T4: the 'capitulation rebound' (drop vs SPY + volume surge + very liquid, hold 10d): neighbourhood of settings, year by year, pooled t."""
import itertools, numpy as np, pandas as pd
src = open("research/audit/t3_volume_gap.py").read().split("\nrows, t0 = [], time.time()\n")[0]
exec(src)
def trades(group, drop, vmin, dv, h, entry, trend="any"):
    fw = FW[(entry, h)]
    cand = []
    for t in range(START, T - h - 1):
        m = L.universe(t, group) & ~isetf & ~np.isnan(raw[t]) & (dv20[t] >= dv) & (vr[t] >= vmin) & (ex1[t] <= -drop)
        if trend == "weak": m &= raw[t] < sma50[t]
        idx = np.flatnonzero(m)
        if len(idx): cand.append((t, idx[np.argsort(ex1[t, idx])]))
    open_pos, out = [], []
    for t, idx in cand:
        open_pos = [p for p in open_pos if p[0] > t]; exposure = len(open_pos) * SIZE; held = {p[1] for p in open_pos}; took = 0
        for j in idx:
            if took >= PER_DAY or exposure + SIZE > CAP + 1e-9: break
            r = fw[t, j]
            if j in held or np.isnan(r): continue
            open_pos.append((t + h, j)); held.add(j); exposure += SIZE; took += 1; out.append((L.dates[t + h].year, r - COST))
    return pd.DataFrame(out, columns=["year", "net"])
def base_of(group, h, entry):
    pool = np.concatenate([FW[(entry, h)][t, L.universe(t, group) & ~isetf & ~np.isnan(raw[t])] for t in range(START, T - h - 1, 5)])
    return np.nanmean(pool) - COST
BASE = {(g, h, e): base_of(g, h, e) for g in "ABC" for h in (5, 10) for e in ("close", "next_open")}
yrs = (T - START) / 252
# 1) neighbourhood: edge pooled over the 3 groups
rows = []
for drop, vmin, dv, entry in itertools.product((.015, .02, .025, .03, .04), (1.5, 2.0, 2.5, 3.0), (500, 700, 1000, 1500, 2500), ("close", "next_open")):
    parts = []
    for g in "ABC":
        t = trades(g, drop, vmin, dv, 10, entry); t["edge"] = t.net - BASE[(g, 10, entry)]; t["g"] = g; parts.append(t)
    p = pd.concat(parts)
    ge = p.groupby("g").edge.mean()
    rows.append({"drop": drop, "vol_surge": vmin, "dv": dv, "entry": entry, "trades_yr_per_group": round(len(p) / 3 / yrs, 1), "edge_pct": round(p.edge.mean() * 100, 2),
                 "t": round(p.edge.mean() / (p.edge.std() / len(p) ** .5), 1) if len(p) > 20 else np.nan, "A": round(ge.get("A", np.nan) * 100, 2), "B": round(ge.get("B", np.nan) * 100, 2), "C": round(ge.get("C", np.nan) * 100, 2)})
nb = pd.DataFrame(rows); nb.to_csv("research/audit/out_t4_neighbourhood.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print("neighbourhood: share of settings with edge>0 pooled:", round((nb.edge_pct > 0).mean(), 2), "| all three groups >0:", round(((nb.A > 0) & (nb.B > 0) & (nb.C > 0)).mean(), 2), "| median edge %.2f" % nb.edge_pct.median())
print(nb.pivot_table(index=["dv"], columns=["vol_surge"], values="edge_pct").round(2).to_string(), "\n(drop/entry averaged)")
print(nb.pivot_table(index=["drop"], columns=["entry"], values="edge_pct").round(2).to_string())
# 2) the pre-chosen centre: drop 2%, vol 2x, dv 1000, hold 10, enter at the close: year by year
parts = []
for g in "ABC":
    t = trades(g, .02, 2.0, 1000, 10, "close"); t["edge"] = t.net - BASE[(g, 10, "close")]; t["g"] = g; parts.append(t)
p = pd.concat(parts)
print("\ncentre rule: trades", len(p), "| avg net %.2f%% | edge %.2f%% | t=%.1f" % (p.net.mean() * 100, p.edge.mean() * 100, p.edge.mean() / (p.edge.std() / len(p) ** .5)))
print(p.groupby("year").agg(trades=("edge", "size"), edge_pct=("edge", lambda x: round(x.mean() * 100, 2))).T.to_string())
print("years with positive edge:", int((p.groupby("year").edge.mean() > 0).sum()), "of", p.year.nunique())
print("without 2020:", round(p[p.year != 2020].edge.mean() * 100, 2), "% edge")
