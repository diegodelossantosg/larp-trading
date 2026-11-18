"""More ideas, judged the strict way: full data AND 30 runs on random halves of the news, paired with the current
rules (no stop since 2026-11-18).

    python -m research.replay.robust_more
"""
import dataclasses, json

import numpy as np

from .sim import Data, Params, run, summary

IDEAS = {
    "take profit at +5%": {"take_profit": 0.05}, "take profit at +10%": {"take_profit": 0.10},
    "take profit at +15%": {"take_profit": 0.15},
    "only while SPY is up today": {"market_up_only": True},
    "only when SPY is above its 200-day average": {"spy_trend_only": True},
    "only stocks below their 50-day average": {"below_sma50_only": True},
    "stop 30% (instead of none)": {"stop": 0.30},
}


def main():
    D = Data()
    P = Params()
    full = summary(run(D, P))
    base = np.array([summary(run(D, dataclasses.replace(P, keep=0.5), seed=s))["extra_per_year_pct"] for s in range(30)])
    out = {"current": {"full": full, "halves_avg": float(base.mean()), "halves_beat_spy": int((base > 0).sum())}}
    print(f"current rules: full {full['extra_per_year_pct']:+.2f}%/yr ({full['years_ahead']} years), "
          f"halves {base.mean():+.2f}%/yr, beats SPY in {(base > 0).sum()}/30", flush=True)
    for name, ch in IDEAS.items():
        Q = dataclasses.replace(P, **ch)
        f = summary(run(D, Q))
        v = np.array([summary(run(D, dataclasses.replace(Q, keep=0.5), seed=s))["extra_per_year_pct"] for s in range(30)])
        d = v - base
        out[name] = {"full": f, "halves_avg": float(v.mean()), "gain": float(d.mean()), "wins": int((d > 0).sum())}
        print(f"{name:44s} full {f['extra_per_year_pct']:+.2f}%/yr  halves {v.mean():+.2f}%/yr  "
              f"vs current {d.mean():+.2f}, better in {(d > 0).sum()}/30", flush=True)
    json.dump(out, open("research/replay/robust_more.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
