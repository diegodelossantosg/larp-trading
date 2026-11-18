"""Why does holding ONE News Hunter stock at a time look so much better? Checks against luck.

    python -m research.replay.one_at_a_time
"""
import dataclasses

import numpy as np

from .sim import Data, Params, run, summary

ONE = dict(size=0.10, cap=0.10)


def line(name, R):
    s, t = summary(R), R.trades
    return (f"{name:44s} extra/yr {s['extra_per_year_pct']:+.2f}%  buys/yr {s['buys_per_year']:5.1f}  "
            f"avg/buy {t.vs_spy.mean() * 100:+.2f}%  years ahead {s['years_ahead']}")


def main():
    D = Data()
    P = Params()
    print(line("current (3 at a time)", run(D, P)), flush=True)
    print(line("one at a time", run(D, dataclasses.replace(P, **ONE))), flush=True)
    for name, ch in {"one at a time, hold 5": {"hold": 5}, "one at a time, hold 15": {"hold": 15},
                     "one at a time, random pick": {"order": "random"}, "one at a time, no stop": {"stop": None},
                     "one at a time, drop 3%": {"drop": 0.03}, "one at a time, swing 2%+": {"atr_min": 2.0}}.items():
        print(line(name, run(D, dataclasses.replace(P, **ONE, **ch), seed=1)), flush=True)
    # half the news, 30 different halves: the paths differ a lot more than with 80%
    d1, d3, per1, per3 = [], [], [], []
    for sd in range(30):
        r1 = run(D, dataclasses.replace(P, keep=0.5, **ONE), seed=sd)
        r3 = run(D, dataclasses.replace(P, keep=0.5), seed=sd)
        d1.append(summary(r1)["extra_per_year_pct"]); d3.append(summary(r3)["extra_per_year_pct"])
        per1.append(r1.trades.vs_spy.mean() * 100); per3.append(r3.trades.vs_spy.mean() * 100)
    d1, d3 = np.array(d1), np.array(d3)
    print(f"50% of the news, 30 runs: one at a time {d1.mean():+.2f}%/yr (avg/buy {np.mean(per1):+.2f}%), "
          f"current {d3.mean():+.2f}%/yr (avg/buy {np.mean(per3):+.2f}%), one-at-a-time ahead in {(d1 > d3).sum()}/30, "
          f"beats SPY in {(d1 > 0).sum()}/30", flush=True)
    # same trades, different picture: what do the trades that the current rule takes ON TOP of the first one earn?
    R3, R1 = run(D, P), run(D, dataclasses.replace(P, **ONE))
    k1 = set(zip(R1.trades.symbol, R1.trades.buy_day))
    extra = R3.trades[[(s, d) not in k1 for s, d in zip(R3.trades.symbol, R3.trades.buy_day)]]
    both = R3.trades[[(s, d) in k1 for s, d in zip(R3.trades.symbol, R3.trades.buy_day)]]
    print(f"current rule's buys also taken one-at-a-time: {len(both)}, avg {both.vs_spy.mean() * 100:+.2f}%; "
          f"the extra buys: {len(extra)}, avg {extra.vs_spy.mean() * 100:+.2f}%", flush=True)
    # are the extra buys the ones made while other news buys are open (crowded days, sell-offs)?
    spy = D.c[:, D.spy]
    def spy5(day):
        i = D.di[day]
        return spy[i - 1] / spy[i - 6] - 1
    for name, t in (("one-at-a-time buys", R1.trades), ("extra buys", extra)):
        print(f"{name}: SPY's move over the 5 days before the buy {np.mean([spy5(d) for d in t.buy_day]) * 100:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
