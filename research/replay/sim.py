"""Hour-by-hour replay of the News Hunter, as it would have run live.

    python -m research.replay.sim                 # the current rules, every news source
    (import run() and Params to try other settings)

Every trading day from 2016:
  9:36  the daily run sells News Hunter buys that are 10+ business days old (at about the open) and puts the money
        back in SPY.
  10:00, 11:00, ... 15:00  the hunter looks at the news of the last 2 hours. A stock named there is a candidate
        if it is down 2%+ more than SPY since yesterday's close AT THAT MOMENT, and passes the live checks:
        a big company or $300M+/day traded, daily swing (ATR) 3%+, not on heavy volume so far today, no earnings
        within 3 trading days, price $5+, not already held. Biggest fall first; 10% of the money per buy; 2 buys a day
        at most; 30% of the money in stocks at most. The buy is paid with SPY sold at the same moment.
  all day  the watcher sells a stock 20% below its highest price since bought (at the stop price, or at the open
        if the stock opens below it).
Costs: 0.05% per stock trade each way (spread), 0.01% per SPY trade. Money not in stocks is always in SPY.

What a mechanical replay cannot copy: Claude reads each headline and may skip a buy (fraud, guidance cut...).
Here every candidate that passes the checks is bought, biggest fall first: the rule without Claude's judgement.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHECKS = (10, 11, 12, 13, 14, 15)
VOL_CURVE = [(0, 0.0), (30, 0.15), (60, 0.25), (120, 0.38), (180, 0.50), (240, 0.62), (300, 0.75), (360, 0.88), (390, 1.0)]


@dataclass(frozen=True)
class Params:
    drop: float = 0.02            # fall vs SPY since yesterday's close, at the moment of the check
    atr_min: float = 3.0          # daily swing, % of price
    liquid_m: float = 300.0       # $M a day traded (big companies always pass)
    max_vr: float | None = 2.0    # skip if today's volume is on pace for this many times the 20-day average
    blackout: int = 3             # no buy this many trading days or fewer before earnings
    size: float = 0.10
    per_day: int = 2
    cap: float = 0.30
    stop: float | None = None     # live since 2026-11-18: News Hunter buys have no trailing stop (sold after 10 days)
    hold: int = 10                # business days
    window_h: float = 2.0         # hours of news the hunter reads
    sources: tuple = ("sec_8k", "alpaca_events", "alpaca_12m")
    headline_limit: int | None = 50   # live: the 50 latest headlines (alpaca_12m only; the other sources are pre-filtered)
    mention_limit: int | None = 40    # live: the first 40 symbols (alphabetical) of those headlines
    hours: tuple = CHECKS
    earnings_news: bool = False   # also buy drops right after an earnings release
    stock_cost: float = 0.0005
    spy_cost: float = 0.0001
    order: str = "biggest_fall"   # or "smallest_fall", "random"
    keep: float = 1.0             # robustness check: use a random share of the news (same seed = same share)
    name_check: bool = True       # live since 2026-11: a symbol counts only if the headline names the company
    claude: str | None = None     # Claude's replay decisions (research/replay/claude_decisions.json): buy only its picks
    collect: bool = False         # record every check with candidates (to know when to ask Claude)
    take_profit: float | None = None   # sell when the price reaches this gain over the buy price
    market_up_only: bool = False  # buy only while SPY is up today (the stock fell on its own)
    spy_trend_only: bool = False  # buy only when SPY is above its 200-day average
    below_sma50_only: bool = False     # buy only stocks already below their 50-day average
    start: str = "2016-03-01"
    end: str = "2099-01-01"


class Data:
    def __init__(self, folder: str = "research/replay"):
        z = np.load(f"{folder}/daily.npz", allow_pickle=False)
        self.dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
        self.days = [d.strftime("%Y-%m-%d") for d in self.dates]
        self.di = {d: i for i, d in enumerate(self.days)}
        self.symbols = list(z["symbols"])
        self.col = {s: j for j, s in enumerate(self.symbols)}
        o, h, l, c, v = (z[k].astype("float64") for k in ("open", "high", "low", "close", "volume"))
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        pc = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
        tr = np.nanmax(np.stack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
        atr = pd.DataFrame(tr).rolling(14).mean().to_numpy() / c * 100
        dv = pd.DataFrame(c * v).rolling(20).mean().to_numpy() / 1e6
        av = pd.DataFrame(v).rolling(20).mean().to_numpy()
        # known before day t's open: values of day t-1
        sh = lambda a: np.vstack([np.full((1, a.shape[1]), np.nan), a[:-1]])
        self.atr_prev, self.dv_prev, self.av_prev = sh(atr), sh(dv), sh(av)
        self.sma50_prev = sh(pd.DataFrame(c).rolling(50).mean().to_numpy())
        self.sma200_prev = sh(pd.DataFrame(c).rolling(200).mean().to_numpy())
        self.spy = self.col["SPY"]
        from trader.universe import BIG_STOCKS
        self.big = np.array([s in BIG_STOCKS for s in self.symbols])
        hr = pd.read_csv(f"{folder}/hourly.csv.gz")
        self.hp, self.hv = {}, {}
        for r in hr.itertuples(index=False):
            key = (r.symbol, r.day)
            self.hp[key] = tuple(getattr(r, f"p{x}") for x in CHECKS)
            self.hv[key] = tuple(getattr(r, f"v{x}") for x in CHECKS)
        from trader.replay_data import news_times
        news = news_times()
        news = news[news.symbol.isin(self.col)]
        news["t"] = pd.to_datetime(news.time_et)
        self.news = news.sort_values("t").reset_index(drop=True)
        self.earn = self._earnings()
        try:   # whole-market headlines (to copy the live "50 latest headlines" limit)
            n12 = pd.read_csv(f"{folder}/news_12m.csv.gz", dtype=str).dropna(subset=["symbols"])
            n12["t"] = pd.to_datetime(n12.time_et)
            try:
                import json
                from trader.name_check import mentions_company
                self.names = json.load(open(f"{folder}/names.json"))
                n12["named"] = [mentions_company(h or "", s, self.names.get(s)) for h, s in zip(n12.headline, n12.symbols)]
            except FileNotFoundError:
                self.names = {}
            self.n12 = n12.sort_values("t").reset_index(drop=True)
        except FileNotFoundError:
            self.n12 = None

    def _earnings(self) -> dict[str, list[int]]:
        """Trading-day indexes of earnings releases per symbol (SEC item 2.02 and Alpaca earnings headlines)."""
        out: dict[str, set] = {}

        def add(sym, ts):
            ts = pd.Timestamp(ts)
            day = ts.strftime("%Y-%m-%d")
            # reaction day: same day if released before 16:00 on a trading day, else the next trading day
            k = int(np.searchsorted(self.days, day))
            if k < len(self.days) and self.days[k] == day and ts.hour >= 16:
                k += 1
            if k < len(self.days):
                out.setdefault(sym, set()).add(k)
        try:
            e = pd.read_csv(f"research/replay/edgar_8k.csv.gz", dtype=str)
            for s, t, it in zip(e.symbol, e.time_et, e["items"].fillna("")):
                if "2.02" in it:
                    add(s, t)
        except FileNotFoundError:
            pass
        from trader.replay_data import EVENT_FILES
        for f in EVENT_FILES:
            try:
                ev = pd.read_csv(f"research/{f}.csv.gz", usecols=["kind", "symbol", "news_time"], dtype=str)
            except (FileNotFoundError, ValueError):
                continue
            for s, t in zip(*ev[ev.kind == "earnings"][["symbol", "news_time"]].to_numpy().T):
                add(s, t)
        return {s: sorted(v) for s, v in out.items()}


def _frac(minutes: float) -> float:
    return float(np.interp(minutes, [a for a, _ in VOL_CURVE], [b for _, b in VOL_CURVE]))


@dataclass
class Pos:
    sym: str
    j: int
    shares: float
    entry: float
    entry_i: int
    entry_day: str
    high: float
    spy_entry: float
    why: str = ""


@dataclass
class Result:
    params: Params
    curve: pd.DataFrame
    trades: pd.DataFrame
    extra: dict = field(default_factory=dict)


def run(D: Data, P: Params = Params(), seed: int = 0) -> Result:
    rng = np.random.default_rng(seed)
    i0 = int(np.searchsorted(D.days, P.start))
    i1 = int(np.searchsorted(D.days, P.end))
    S = D.spy
    spy_sh = 1000.0 / D.c[i0 - 1, S]       # start: $1,000 in SPY at the previous close
    bench_sh = spy_sh
    pos: list[Pos] = []
    trades, curve = [], []
    news = D.news[D.news.source.isin(P.sources)]
    if P.keep < 1:
        news = news.sample(frac=P.keep, random_state=seed)
    if not P.earnings_news:
        news = news[~news.text.fillna("").str.fullmatch(r"8-K items (2\.02|9\.01|,)+")]
    by_day = {d: g for d, g in news.groupby(news.t.dt.strftime("%Y-%m-%d"))}
    n12 = D.n12
    if n12 is not None and P.name_check and "named" in n12:
        n12 = n12[n12.named]
    n12_day = {d: g for d, g in n12.groupby(n12.t.dt.strftime("%Y-%m-%d"))} if (n12 is not None and "alpaca_12m" in P.sources) else {}
    picks: dict[str, list[tuple[int, str]]] = {}
    if P.claude:
        import json
        for key, v in json.load(open(P.claude)).items():
            d, hh = key.split(" ")
            for sym in v.get("ideas", []):
                picks.setdefault(d, []).append((int(hh), sym))
    points = []
    earn = D.earn

    def sell(p: Pos, px: float, i: int, reason: str):
        nonlocal spy_sh
        val = p.shares * px * (1 - P.stock_cost)
        spy_px = D.o[i, S] if reason == "time" else D.c[i, S]
        spy_sh += val * (1 - P.spy_cost) / spy_px
        r = px / p.entry - 1
        sr = spy_px / p.spy_entry - 1
        trades.append({"symbol": p.sym, "buy_day": p.entry_day, "sell_day": D.days[i], "reason": reason,
                       "ret": r - 2 * P.stock_cost, "vs_spy": r - sr - 2 * P.stock_cost, "why": p.why})

    for i in range(i0, min(i1, len(D.days))):
        day = D.days[i]
        # 1) 9:36: time exits at about the open
        keep = []
        for p in pos:
            if P.hold and np.busday_count(p.entry_day, day) >= P.hold and np.isfinite(D.o[i, p.j]):
                sell(p, D.o[i, p.j], i, "time")
            else:
                keep.append(p)
        pos = keep
        # 2) trailing stop during the day (on yesterday's highest; then today's high counts)
        keep = []
        for p in pos:
            o, h, l = D.o[i, p.j], D.h[i, p.j], D.l[i, p.j]
            if p.entry_i < i and P.stop and np.isfinite(l):
                lvl = p.high * (1 - P.stop)
                if o <= lvl:
                    sell(p, o, i, "stop"); continue
                if l <= lvl:
                    sell(p, lvl, i, "stop"); continue
            if P.take_profit and p.entry_i < i and np.isfinite(h):
                tgt = p.entry * (1 + P.take_profit)
                if o >= tgt:
                    sell(p, o, i, "target"); continue
                if h >= tgt:
                    sell(p, tgt, i, "target"); continue
            if p.entry_i < i and np.isfinite(h):
                p.high = max(p.high, h)
            keep.append(p)
        pos = keep
        # 3) the hourly News Hunter
        bought_today = 0
        g = by_day.get(day)
        g12 = n12_day.get(day)
        if (g is not None or g12 is not None) and i > 20:
            prev_c = D.c[i - 1]
            for k, hh in enumerate(P.hours):
                if bought_today >= P.per_day:
                    break
                spy_hp = D.hp.get(("SPY", day))
                if not spy_hp or not np.isfinite(spy_hp[k]):
                    continue
                spy_px = spy_hp[k]
                spy_move = spy_px / prev_c[S] - 1
                now = pd.Timestamp(f"{day} {hh:02d}:00")
                lo = now - pd.Timedelta(hours=P.window_h)
                syms: dict[str, str] = {}
                if g is not None:
                    w = g[(g.t >= lo) & (g.t < now) & (g.source != "alpaca_12m")]
                    for s, txt in zip(w.symbol, w.text):
                        syms.setdefault(s, str(txt))
                if g12 is not None:
                    w = g12[(g12.t >= lo) & (g12.t < now)]
                    if P.headline_limit:
                        w = w.iloc[-P.headline_limit:]
                    ment = sorted({s for ss in w.symbols for s in ss.split() if s.isalpha() and len(s) <= 5})
                    if P.mention_limit:
                        ment = ment[: P.mention_limit]
                    first = {}
                    for ss, txt in zip(w.symbols, w.headline):
                        for s in ss.split():
                            first.setdefault(s, str(txt))
                    for s in ment:
                        if s in D.col:
                            syms.setdefault(s, first.get(s, ""))
                held = {p.sym for p in pos}
                eq_spy = spy_sh * spy_px
                vals = {}
                for p in pos:
                    hp = D.hp.get((p.sym, day))
                    px = hp[k] if hp and np.isfinite(hp[k]) else (D.c[i - 1, p.j] if np.isfinite(D.c[i - 1, p.j]) else p.entry)
                    vals[p.sym] = p.shares * px
                equity = eq_spy + sum(vals.values())
                cands = []
                if P.claude:     # Claude's picks so far today replace the headline list; the drop rule is Claude's call
                    syms = {sym: "Claude's pick" for h0, sym in picks.get(day, []) if h0 <= hh}
                for s, txt in syms.items():
                    j = D.col.get(s)
                    if j is None or j == S or s in held:
                        continue
                    hp = D.hp.get((s, day))
                    if not hp or not np.isfinite(hp[k]) or not np.isfinite(prev_c[j]):
                        continue
                    px = hp[k]
                    vs = px / prev_c[j] - 1 - spy_move
                    if (vs > -P.drop and not P.claude) or px < 5:
                        continue
                    if P.market_up_only and spy_move <= 0:
                        continue
                    if P.spy_trend_only and not (prev_c[S] > D.sma200_prev[i, S]):
                        continue
                    if P.below_sma50_only and not (prev_c[j] < D.sma50_prev[i, j]):
                        continue
                    if not D.big[j] and not (D.dv_prev[i, j] >= P.liquid_m):
                        continue
                    if not (D.atr_prev[i, j] >= P.atr_min):
                        continue
                    if P.max_vr is not None:
                        fr = _frac(hh * 60 - 9 * 60 - 30)
                        vol = D.hv[(s, day)][k]
                        if fr >= 0.05 and np.isfinite(vol) and D.av_prev[i, j] > 0 and vol / fr / D.av_prev[i, j] >= P.max_vr:
                            continue
                    ed = earn.get(s)
                    if not P.earnings_news and ed and (i in ed or i - 1 in ed):
                        continue          # today's drop is the earnings reaction itself
                    if P.blackout and ed:
                        nxt = ed[int(np.searchsorted(ed, i))] if np.searchsorted(ed, i) < len(ed) else None
                        if nxt is not None and nxt - i <= P.blackout:
                            continue
                    cands.append((vs, s, j, px, txt))
                if P.collect and cands:
                    points.append({"day": day, "hour": hh, "candidates": [c[1] for c in sorted(cands)]})
                if P.claude:
                    order = [sym for _, sym in picks.get(day, [])]
                    cands.sort(key=lambda c: order.index(c[1]))
                elif P.order == "random":
                    rng.shuffle(cands)
                else:
                    cands.sort(reverse=(P.order == "smallest_fall"))
                for vs, s, j, px, txt in cands:
                    if bought_today >= P.per_day:
                        break
                    invested = sum(vals.values())
                    amount = P.size * equity
                    if invested + amount > P.cap * equity + 1e-9 or amount > eq_spy:
                        break
                    fill = px * (1 + P.stock_cost)
                    spy_sh -= amount / (spy_px * (1 - P.spy_cost))
                    eq_spy = spy_sh * spy_px
                    pos.append(Pos(s, j, amount / fill, fill, i, day, fill, spy_px, why=txt[:120]))
                    vals[s] = amount
                    held.add(s)
                    bought_today += 1
        # 4) close: mark to market
        stock_val = sum(p.shares * (D.c[i, p.j] if np.isfinite(D.c[i, p.j]) else p.entry) for p in pos)
        curve.append({"day": day, "equity": spy_sh * D.c[i, S] + stock_val, "spy": bench_sh * D.c[i, S],
                      "in_stocks": stock_val})
    return Result(P, pd.DataFrame(curve), pd.DataFrame(trades), extra={"points": points})


def summary(res: Result) -> dict:
    c = res.curve
    if c.empty:
        return {}
    yrs = len(c) / 252
    a, b = c.equity.iloc[-1] / 1000, c.spy.iloc[-1] / 1000
    dd = (c.equity / c.equity.cummax() - 1).min()
    t = res.trades
    out = {"years": round(yrs, 1), "ai_total_pct": round((a - 1) * 100, 1), "spy_total_pct": round((b - 1) * 100, 1),
           "extra_per_year_pct": round(((a / b) ** (1 / yrs) - 1) * 100, 2), "worst_drop_pct": round(dd * 100, 1),
           "buys_per_year": round(len(t) / yrs, 1) if len(t) else 0}
    if len(t):
        out |= {"avg_vs_spy_pct": round(t.vs_spy.mean() * 100, 2), "win_vs_spy_pct": round((t.vs_spy > 0).mean() * 100),
                "stops_pct": round((t.reason == "stop").mean() * 100)}
    c = c.assign(y=pd.to_datetime(c.day).dt.year)
    yr = {}
    for y, g in c.groupby("y"):
        s0 = g.iloc[0]
        prev = c[c.y < y]
        base = prev.iloc[-1] if len(prev) else {"equity": 1000.0, "spy": 1000.0}
        yr[int(y)] = round(((g.equity.iloc[-1] / base["equity"]) / (g.spy.iloc[-1] / base["spy"]) - 1) * 100, 2)
    out["extra_by_year_pct"] = yr
    out["years_ahead"] = f"{sum(v > 0 for v in yr.values())}/{len(yr)}"
    return out


def variants(P: Params, **grid) -> list[Params]:
    import itertools
    keys = list(grid)
    return [dataclasses.replace(P, **dict(zip(keys, vals))) for vals in itertools.product(*grid.values())]


if __name__ == "__main__":
    import json
    D = Data()
    print(json.dumps(summary(run(D)), indent=1))
