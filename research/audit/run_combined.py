"""Everything crossed in one portfolio: idle asset (SSO/SPY trend) + ideas sleeve (zero-skill proxy + assumed skill, with the
watcher's stop) + News Hunter (all company sets together = what the live agent trades). Window 2023-07..2026-10 (news data)."""
import itertools, numpy as np, pandas as pd
from trader.lab import Lab
from research.audit.sim import *
from research.audit import run_news_lib as R

L = R.L; idle = idle_returns(L); spy = spy_returns(L); idle_np = idle.to_numpy()
START_W, END_W = "2023-07-01", None

IDLE_CUM = (1 + idle).cumprod().to_numpy(); SPY_CUM = (1 + spy).cumprod().to_numpy()

def news_series(dv, drop, per_day, size, h, st):
    """Daily extra return of the News Hunter on all three company sets together (credited on the exit day)."""
    out = pd.Series(0.0, index=L.dates)
    nn = 0
    for s in ("dev", "big", "holdout"):
        pass
    sub = R.news[R.news["set"].isin(["dev", "holdout"]) & (R.news.move_1h <= -drop) & (R.news.dollar_vol_m >= dv)].sort_values(["i", "move_1h"])
    col = f"hour_after_{st}{h}"
    base = np.nanmean([R.base[(s, dv, h, st)] for s in ("dev", "holdout")])
    open_pos = []
    for i, g in sub.groupby("i", sort=True):
        open_pos = [p for p in open_pos if p[0] > i]
        exposure, held, took = sum(p[1] for p in open_pos), {p[2] for p in open_pos}, 0
        for sym, r in zip(g.symbol.to_numpy(), g[col].to_numpy()):
            if took >= per_day or exposure + size > R.CAP + 1e-9: break
            if sym in held or np.isnan(r) or i + h >= len(L.dates): continue
            open_pos.append((i + h, size, sym)); held.add(sym); exposure += size; took += 1; nn += 1
            # the money comes out of the idle asset (SSO or SPY), so also pay what the idle asset earned beyond SPY meanwhile
            idle_extra = (IDLE_CUM[i + h] / IDLE_CUM[i]) - (SPY_CUM[i + h] / SPY_CUM[i])
            out.iloc[i + h] += size * (r - R.COST_RT - idle_extra)
    return out, nn

NEWS = {"off": None, "current (5%)": (300, .02, 2, .05, 10, "s"), "10% no stop": (300, .02, 2, .10, 10, "r"),
        "15% no stop": (300, .02, 2, .15, 10, "r"), "more firms 20M": (20, .02, 2, .05, 10, "r"), "5 buys/day": (300, .02, 5, .05, 10, "s")}
nser = {k: (None if v is None else news_series(*v)) for k, v in NEWS.items()}
for k, v in nser.items():
    if v is not None: print("news", k, "trades:", v[1], "alpha/yr in window %:", round(v[0].loc[START_W:].sum() / ((len(v[0].loc[START_W:])) / 252) * 100, 2))

STOPS = {"no stop": None, "fixed 6% (now ETFs)": ("fixed", .06), "atr 4x (now stocks)": ("atr", 4.0), "fixed 10%": ("fixed", .10), "fixed 20%": ("fixed", .20)}
sl = {}
for sn, st in STOPS.items():
    rs, invs = [], []
    for g in "ABC":
        r, inv, _ = sleeve(L, idle_np, group=g, kind="stocks", n=10, cadence=21, stop=st); rs.append(r); invs.append(inv)
    sl[sn] = (sum(rs) / 3, sum(invs) / 3)
rows = []
for sn, w, a, nk in itertools.product(STOPS, (0, .1, .3, .5, .8), (0.0, .02, .05, .10), NEWS):
    r, inv = sl[sn]
    p = combine(w, r, idle, a, inv)
    if nser[nk] is not None: p = p + nser[nk][0]
    s = stats(p, spy, START_W, END_W)
    rows.append({"stop": sn, "w": w, "skill_yr": a, "news": nk, **{k: s[k] for k in ("cagr", "spy_cagr", "excess", "mdd", "spy_mdd", "yrs_beat", "worst_year_excess")}})
pd.DataFrame(rows).to_csv("research/audit/out_combined.csv", index=False)
print(len(rows), "rows")
