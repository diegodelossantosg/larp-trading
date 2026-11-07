"""T9: (a) unleveraged timing of the S&P 500 (can any rule beat buy-and-hold SPY in RETURN?)  (b) stock-picking factors with the strict protocol (pick on A, judge on B and C)."""
import numpy as np, pandas as pd
from trader.lab import Lab, stats as lstats, pick_backtest, equal_weight, SCORES
L = Lab(); pd.set_option("display.width", 220)
spy_p = L.series("SPY"); spy = spy_p.pct_change().fillna(0)
ief = L.series("IEF").pct_change().fillna(0); shy = L.series("SHY").pct_change().fillna(0)
COST = 0.001
def switch_cost(on): return on.astype(int).diff().abs().fillna(0) * COST * 2
rows = []
def add(name, r): s = lstats(r, spy); rows.append({"rule": name, "yearly_%": s["yearly_pct"], "S&P_%": s["bench_yearly_pct"], "worst_drop_%": s["worst_drop_pct"], "years_beating": s["years_beating"]})
add("SPY buy and hold", spy)
for n in (50, 100, 200):
    on = (spy_p > spy_p.rolling(n).mean()).shift(1, fill_value=False)
    for alt, nm in ((0 * spy, "cash"), (ief, "IEF bonds"), (shy, "SHY short bonds")):
        add(f"SPY above {n}d avg, else {nm}", pd.Series(np.where(on, spy, alt), index=spy.index) - switch_cost(on))
vol = spy.rolling(20).std() * np.sqrt(252)
for thr in (0.15, 0.20, 0.25, 0.30):
    on = (vol < thr).shift(1, fill_value=False)
    add(f"SPY when 20d vol < {int(thr*100)}%, else IEF", pd.Series(np.where(on, spy, ief), index=spy.index) - switch_cost(on))
# calendar: only the first/last days of the month in SPY, else cash (a pure timing idea)
m = spy.index.to_period("M"); pos = pd.Series(range(len(spy)), index=spy.index).groupby(m).cumcount(); left = pd.Series(range(len(spy)), index=spy.index).groupby(m).cumcount(ascending=False)
tom = (pos <= 2) | (left <= 1)
add("SPY only around the turn of the month", pd.Series(np.where(tom, spy, 0.0), index=spy.index) - switch_cost(tom))
# after a drop day of -2% in SPY: hold 5 extra? (always invested anyway) -> skip. Overnight only:
c, o = L.series("SPY"), pd.Series(L.open[:, L.col["SPY"]].astype("float64"), index=L.dates)
add("SPY overnight only (close->open)", (o / c.shift(1) - 1).fillna(0) - 0.0002)
add("SPY intraday only (open->close)", (c / o - 1).fillna(0) - 0.0002)
t = pd.DataFrame(rows); t.to_csv("research/audit/out_t9_timing.csv", index=False)
print("(a) TIMING THE S&P 500 WITHOUT LEVERAGE, 2017-2026:"); print(t.to_string(index=False))
# (b) factor picking with the protocol
out = []
for sc in SCORES:
    for n in (10, 20):
        for g in "ABC":
            r = pick_backtest(L, sc, g, n=n); s = lstats(r, spy); ew = lstats(equal_weight(L, g), spy)
            out.append({"score": sc, "n": n, "group": g, "yearly": s["yearly_pct"], "spy": s["bench_yearly_pct"], "ew_group": ew["yearly_pct"], "worst_drop": s["worst_drop_pct"]})
d = pd.DataFrame(out); d["vs_spy"] = d.yearly - d.spy; d["vs_ew"] = d.yearly - d.ew_group
w = d.pivot_table(index=["score", "n"], columns="group", values="vs_spy")
print("\n(b) STOCK-PICKING FACTORS, excess return vs the S&P per group (2017-2026):"); print(w.round(1).to_string())
top = w.sort_values("A", ascending=False).head(5)
print("\nthe 5 best on A (design) -> B and C:"); print(top.round(1).to_string())
print("mean of those on B %.1f, C %.1f | share of factors beating the S&P: A %.2f B %.2f C %.2f" % (top.B.mean(), top.C.mean(), (w.A > 0).mean(), (w.B > 0).mean(), (w.C > 0).mean()))
d.to_csv("research/audit/out_t9_factors.csv", index=False)
