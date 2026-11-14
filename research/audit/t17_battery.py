"""T17: a battery of new ideas, all judged the same strict way. Run from the repo root: python -m research.audit.t17_battery

Daily ideas (each year's ~500 most-traded stocks, price >= $5, 20-day avg dollar volume >= $20M; result = stock minus
SPY over the same days, minus 0.1% costs; design 2016-2020, validation 2021 on):
  gap down / gap up at the open, 52-week-high breakout, heavy-volume up day, tight-range breakout.
SPY calendar effects: turn of the month, day of the week (plain SPY days, no costs).
News Hunter modifiers (events_long + events_wide): does the edge of the current rule change with the market regime,
the stock's recent trend, the size of the drop, liquidity, weekday, hour or topic? design 2018-2022, validation 2023 on.
"""
import numpy as np, pandas as pd
from trader.lab import Lab

COST = 0.001
L = Lab()
o, c, v = L.open.astype("float64"), L.close.astype("float64"), L.volume.astype("float64")
n_days, n_sym = c.shape
spy = L.col["SPY"]
yrs = np.array([d.year for d in L.dates])
member = np.zeros_like(c, bool)
for i, y in enumerate(L.years):
    member[yrs == y] = L.member[i][None, :]
dvol = pd.DataFrame(c * v).rolling(20).mean().shift(1).to_numpy()
ok = member & (c >= 5) & (dvol >= 20e6); ok[:, [spy]] = False
prev_c = np.vstack([np.full((1, n_sym), np.nan), c[:-1]])
gap = o / prev_c - 1
ret1 = c / prev_c - 1
vavg = pd.DataFrame(v).rolling(20).mean().shift(1).to_numpy()
hi252 = pd.DataFrame(c).rolling(252, min_periods=200).max().shift(1).to_numpy()
rng_prev = None
PER = {"design": (2016, 2020), "valid": (2021, 2026)}


def trade_excess(entry_same_open: bool, hold: int):
    """excess return vs SPY of: buy at the signal day's open (or the next open) and sell at the close `hold` days later."""
    ex = np.full_like(c, np.nan)
    for t in range(n_days):
        e = t if entry_same_open else t + 1
        x = t + hold if entry_same_open else t + hold
        if hold == 0 and not entry_same_open:
            continue
        if e >= n_days or x >= n_days or x < e:
            continue
        r = c[x] / o[e] - 1
        s = c[x, spy] / o[e, spy] - 1
        ex[t] = r - s
    return ex


def evaluate(name, mask, ex):
    sig = ok & mask & np.isfinite(ex)
    row = {"idea": name}
    for per, (a, b) in PER.items():
        yr = (yrs >= a) & (yrs <= b)
        sel = sig & yr[:, None]
        x = ex[sel] - COST
        base = ex[ok & yr[:, None] & np.isfinite(ex)].mean() - COST
        ys = [(ex[sig & (yrs == y)[:, None]] - COST).mean() for y in range(a, b + 1) if (sig & (yrs == y)[:, None]).sum() >= 20]
        row[per] = dict(n=int(len(x)), avg=float(x.mean() * 100) if len(x) else np.nan, edge=float((x.mean() - base) * 100) if len(x) else np.nan,
                        t=float(x.mean() / (x.std() / len(x) ** .5)) if len(x) > 2 and x.std() > 0 else 0.0,
                        yp=sum(1 for g in ys if g > 0), ny=len(ys))
    return row


rows = []
for hold in (0, 1, 5, 10):
    exs = trade_excess(True, hold)
    for th in (0.03, 0.05, 0.08):
        rows.append(evaluate(f"gap DOWN {th:.0%}+ (buy the open) hold {hold}d", gap <= -th, exs) )
        rows.append(evaluate(f"gap UP {th:.0%}+ (buy the open) hold {hold}d", gap >= th, exs))
for hold in (5, 10, 20):
    exn = trade_excess(False, hold)
    brk = (c > hi252) & (v > 1.5 * vavg)
    rows.append(evaluate(f"52-week high on heavy volume, hold {hold}d", brk, exn))
    rows.append(evaluate(f"+3% day on 3x volume, hold {hold}d", (ret1 >= 0.03) & (v > 3 * vavg), exn))
    rows.append(evaluate(f"-3% day on 3x volume, hold {hold}d", (ret1 <= -0.03) & (v > 3 * vavg), exn))
print(f"{'idea':52} {'design avg':>10} {'valid avg':>9} {'valid edge':>10} {'t':>5} {'years+':>7} {'n valid':>8}")
passing = []
for r in rows:
    d, v_ = r["design"], r["valid"]
    flag = " <== passes" if (d["avg"] > 0 and v_["avg"] > 0 and v_["t"] > 3 and v_["yp"] >= 0.7 * max(v_["ny"], 1)) else ""
    if flag: passing.append(r["idea"])
    print(f"{r['idea']:52} {d['avg']:+10.2f} {v_['avg']:+9.2f} {v_['edge']:+10.2f} {v_['t']:5.1f} {v_['yp']}/{v_['ny']:>3} {v_['n']:8d}{flag}")
print("passing:", passing)

# ---- SPY calendar effects (plain SPY, close to close)
s = pd.Series(c[:, spy], index=L.dates).pct_change().dropna()
df = pd.DataFrame({"r": s}); df["dow"] = df.index.dayofweek; df["yr"] = df.index.year
dm = df.index.to_period("M"); first3 = df.groupby(dm).cumcount() < 3
last2 = df.groupby(dm).cumcount(ascending=False) < 2
df["tom"] = first3 | last2
print("\nSPY average daily return %% (all days %.3f):" % (df.r.mean() * 100))
print(" turn of month (last 2 + first 3 days): %.3f  | other days: %.3f" % (df[df.tom].r.mean() * 100, df[~df.tom].r.mean() * 100))
print(" by weekday Mon..Fri:", [round(df[df.dow == k].r.mean() * 100, 3) for k in range(5)])
for per, (a, b) in PER.items():
    d = df[(df.yr >= a) & (df.yr <= b)]
    print(f" {per}: turn of month %.3f vs other %.3f" % (d[d.tom].r.mean() * 100, d[~d.tom].r.mean() * 100))

# ---- News Hunter modifiers
fr = []
for f in ("events_long", "events_wide"):
    e = pd.read_csv(f"research/{f}.csv.gz", low_memory=False); fr.append(e)
e = pd.concat(fr); e["d"] = pd.to_datetime(e.day)
n = e[(e.kind == "news") & (e.move_1h <= -0.02) & (e.atr_pct >= 3) & (e.dollar_vol_m >= 20)].drop_duplicates(["symbol", "day"]).dropna(subset=["hour_after_r10"]).copy()
ctl = e[(e.kind == "control") & (e.atr_pct >= 3) & (e.dollar_vol_m >= 20)].drop_duplicates(["symbol", "day"])
cut = pd.Timestamp("2023-01-01")
base_d = ctl[ctl.d < cut]["h11_r10"].mean(); base_v = ctl[ctl.d >= cut]["h11_r10"].mean()
n["net"] = n.hour_after_r10 - COST
n["edge"] = np.where(n.d < cut, n.net - base_d, n.net - base_v)
n["hour"] = pd.to_datetime(n.news_time, errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.hour
n["dow"] = n.d.dt.dayofweek
sl = {"SPY above 200d": n.spy_above_sma200 == 1, "SPY below 200d": n.spy_above_sma200 == 0, "SPY fell 20d (<-3%)": n.spy_ret20_before < -0.03,
      "SPY rose 20d (>+3%)": n.spy_ret20_before > 0.03, "stock fell 20d (<-10%)": n.ret20_before < -0.10, "stock rose 20d (>+10%)": n.ret20_before > 0.10,
      "stock above 200d": n.above_sma200 == 1, "stock below 200d": n.above_sma200 == 0, "drop 2-3%": n.move_1h > -0.03,
      "drop 3-5%": (n.move_1h <= -0.03) & (n.move_1h > -0.05), "drop 5%+": n.move_1h <= -0.05, "ATR 3-4%": (n.atr_pct < 4),
      "ATR 4-6%": (n.atr_pct >= 4) & (n.atr_pct < 6), "ATR 6%+": n.atr_pct >= 6, "liquidity 20-100M": n.dollar_vol_m < 100,
      "liquidity 100-1000M": (n.dollar_vol_m >= 100) & (n.dollar_vol_m < 1000), "liquidity 1000M+": n.dollar_vol_m >= 1000,
      "Mon": n.dow == 0, "Tue": n.dow == 1, "Wed": n.dow == 2, "Thu": n.dow == 3, "Fri": n.dow == 4,
      "before 10:30": n.hour <= 9, "10:30-14:00": (n.hour >= 10) & (n.hour <= 13), "after 14:00": n.hour >= 14,
      "heavy volume (ratio>2)": n.vol_ratio > 2, "normal volume": n.vol_ratio <= 2}
for t in [c_ for c_ in n.columns if c_.startswith("topic_")]:
    sl[t] = n[t] == 1
print("\nNews Hunter rule: ALL trades design %+.2f%% (n=%d) valid %+.2f%% (n=%d) [edge vs random days]" % (
    n[n.d < cut].edge.mean() * 100, (n.d < cut).sum(), n[n.d >= cut].edge.mean() * 100, (n.d >= cut).sum()))
out = []
for k, m in sl.items():
    a, b = n[m & (n.d < cut)], n[m & (n.d >= cut)]
    if len(a) >= 40 and len(b) >= 60:
        out.append((k, len(a), a.edge.mean() * 100, len(b), b.edge.mean() * 100, b.edge.mean() / (b.edge.std() / len(b) ** .5)))
for k, na, ea, nb, eb, t in sorted(out, key=lambda x: -x[4]):
    print(f" {k:26} design {ea:+.2f}% (n={na:4d})  valid {eb:+.2f}% (n={nb:4d}) t={t:+.1f}{'  <== both positive' if ea > 0 and eb > 0 else ''}")
