"""T3 (10 years, groups A/B/C, price+volume only): do capitulation drops (volume surge) or opening gaps rebound? Edge vs random days of the group."""
import itertools, time, numpy as np, pandas as pd
src = open("research/audit/run_reversal.py").read().split("rows, t0 = [], time.time()")[0]
exec(src)
vol = L.volume.astype("float64")
vr = vol / pd.DataFrame(vol).rolling(20, min_periods=10).mean().shift(1).to_numpy()        # today's volume vs the prior 20-day average
op = L.open.astype("float64")
spy_op = op[:, L.col["SPY"]]
gap = op / np.vstack([np.full((1, N), np.nan), raw[:-1]]) - 1 - (spy_op / np.r_[np.nan, spy[:-1]] - 1)[:, None]   # opening gap vs SPY
def fwd_open(h):          # buy at the NEXT day's open, sell at the close h days later
    out = np.full((T, N), np.nan)
    out[:T - h - 1] = c[h + 1:T] / op[1:T - h] - 1 - (spy[h + 1:T] / spy_op[1:T - h] - 1)[:, None]
    return out
FW = {("close", h): fwd(h) for h in (5, 10)} | {("next_open", h): fwd_open(h) for h in (5, 10)}
def run3(group, trig, drop, vmin, dv, trend, h, entry):
    fw = FW[(entry, h)]
    cand = []
    for t in range(START, T - h - 1):
        base = L.universe(t, group) & ~isetf & ~np.isnan(raw[t]) & (dv20[t] >= dv) & (vr[t] >= vmin)
        m = base & ((ex1[t] <= -drop) if trig == "drop" else (gap[t] <= -drop))
        if trend == "weak": m &= raw[t] < sma50[t]
        idx = np.flatnonzero(m)
        if len(idx): cand.append((t, idx[np.argsort(ex1[t, idx])]))
    open_pos, net = [], []
    for t, idx in cand:
        open_pos = [p for p in open_pos if p[0] > t]; exposure = len(open_pos) * SIZE; held = {p[1] for p in open_pos}; took = 0
        for j in idx:
            if took >= PER_DAY or exposure + SIZE > CAP + 1e-9: break
            r = fw[t, j]
            if j in held or np.isnan(r): continue
            open_pos.append((t + h, j)); held.add(j); exposure += SIZE; took += 1; net.append(r - COST)
    return np.array(net)
rows, t0 = [], time.time()
for g in "ABC":
    for h, entry in itertools.product((5, 10), ("close", "next_open")):
        pool = np.concatenate([FW[(entry, h)][t, L.universe(t, g) & ~isetf & ~np.isnan(raw[t])] for t in range(START, T - h - 1, 5)])
        base = np.nanmean(pool) - COST
        for trig, drop, vmin, dv, trend in itertools.product(("drop", "gap"), (.02, .03, .05), (1.0, 2.0, 3.0), (300, 1000), ("any", "weak")):
            net = run3(g, trig, drop, vmin, dv, trend, h, entry); n = len(net)
            rows.append({"group": g, "trigger": trig, "drop": drop, "vol_surge_min": vmin, "dv": dv, "trend": trend, "hold": h, "entry": entry, "trades_yr": round(n / ((T - START) / 252), 1),
                         "avg_net_pct": round(net.mean() * 100, 2) if n else np.nan, "edge_pct": round((net.mean() - base) * 100, 2) if n else np.nan,
                         "t": round((net.mean() - base) / (net.std() / n ** .5), 1) if n > 10 and net.std() > 0 else np.nan})
    print(g, round(time.time() - t0), "s", flush=True)
d = pd.DataFrame(rows); d.to_csv("research/audit/out_t3.csv", index=False)
k = ["trigger", "drop", "vol_surge_min", "dv", "trend", "hold", "entry"]
w = d.pivot_table(index=k, columns="group", values="edge_pct")
pos = (w > 0).all(axis=1)
print("\nconfigs positive in A, B and C:", int(pos.sum()), "of", len(w))
for col in ("trigger", "vol_surge_min", "trend", "entry", "hold", "dv", "drop"):
    print("\n", col); print(pd.concat([d.groupby(col).edge_pct.mean().round(2).rename("mean_edge"), (w > 0).all(axis=1).groupby(level=col).mean().round(2).rename("share_all3_positive")], axis=1))
print("\nbest 12 by the WEAKEST group's edge (n>=40 trades/yr*10y):")
w2 = d.pivot_table(index=k, columns="group", values=["edge_pct", "trades_yr"]); w2["worst"] = w2["edge_pct"].min(axis=1)
print(w2.sort_values("worst", ascending=False).head(12).round(2).to_string())
