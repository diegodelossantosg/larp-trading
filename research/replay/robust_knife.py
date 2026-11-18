"""Rules against buying real disasters (like PG&E in the 2018 wildfires), judged the strict way.

    python -m research.replay.robust_knife
"""
import dataclasses, json

import numpy as np

from .sim import Data, Params, run, summary

IDEAS = {
    "skip if already down 10%+ in the 5 days before": {"prior_drop": 0.10},
    "skip if already down 15%+ in the 5 days before": {"prior_drop": 0.15},
    "no re-buy for 20 days after a loss": {"cooldown": 20},
    "no re-buy for 60 days after a loss": {"cooldown": 60},
    "skip falls of more than 10% today": {"max_drop": 0.10},
    "skip falls of more than 15% today": {"max_drop": 0.15},
    "skip headlines with danger words": {"danger_words": True},
    "all three: 5-day 10%, 60-day cooldown, danger words": {"prior_drop": 0.10, "cooldown": 60, "danger_words": True},
}


def main():
    D = Data()
    P = Params()
    full = summary(run(D, P))
    base = np.array([summary(run(D, dataclasses.replace(P, keep=0.5), seed=s))["extra_per_year_pct"] for s in range(30)])
    out = {"current": {"full": full, "halves_avg": float(base.mean()), "beats_spy": int((base > 0).sum())}}
    print(f"current: full {full['extra_per_year_pct']:+.2f}%/yr, 2018 {full['extra_by_year_pct'][2018]:+.2f}, "
          f"halves {base.mean():+.2f}%/yr, beats SPY {(base > 0).sum()}/30", flush=True)
    for name, ch in IDEAS.items():
        Q = dataclasses.replace(P, **ch)
        R = run(D, Q)
        f = summary(R)
        v = np.array([summary(run(D, dataclasses.replace(Q, keep=0.5), seed=s))["extra_per_year_pct"] for s in range(30)])
        d = v - base
        pcg = R.trades[(R.trades.symbol == "PCG") & (R.trades.buy_day.str[:4] == "2018")]
        out[name] = {"full": f, "halves_avg": float(v.mean()), "gain": float(d.mean()), "wins": int((d > 0).sum()),
                     "beats_spy": int((v > 0).sum()), "pcg_2018_buys": len(pcg)}
        print(f"{name:52s} full {f['extra_per_year_pct']:+.2f}%/yr (2018 {f['extra_by_year_pct'][2018]:+.2f}, PG&E buys {len(pcg)}) "
              f"halves {v.mean():+.2f}, vs current {d.mean():+.2f}, better {(d > 0).sum()}/30, beats SPY {(v > 0).sum()}/30", flush=True)
    json.dump(out, open("research/replay/robust_knife.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
