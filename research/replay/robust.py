"""Is a better-looking alternative really better, or luck? Re-run it and the current rules on the same random 80% of
the news, 20 times, and count how often the alternative wins (paired comparison).

    python -m research.replay.robust
"""
import dataclasses, json

import numpy as np

from .sim import Data, Params, run, summary

CANDIDATES = {
    "20% per buy": {"size": 0.20}, "stop 30%": {"stop": 0.30}, "no stop": {"stop": None},
    "reads 4h of news": {"window_h": 4.0}, "smallest fall first": {"order": "smallest_fall"},
    "3 buys a day": {"per_day": 3}, "buy earnings drops too": {"earnings_news": True}, "stop 15%": {"stop": 0.15},
}
SEEDS = range(20)


def main():
    D = Data()
    base = {}
    for sd in SEEDS:
        base[sd] = summary(run(D, dataclasses.replace(Params(), keep=0.8), seed=sd))["extra_per_year_pct"]
    print("base (80% of the news):", np.round(np.mean(list(base.values())), 2), "spread", np.round(np.std(list(base.values())), 2), flush=True)
    out = {"base": base}
    for name, ch in CANDIDATES.items():
        v = {sd: summary(run(D, dataclasses.replace(Params(), keep=0.8, **ch), seed=sd))["extra_per_year_pct"] for sd in SEEDS}
        diff = np.array([v[s] - base[s] for s in SEEDS])
        out[name] = {"runs": v, "avg_gain_vs_base": round(float(diff.mean()), 2), "wins": int((diff > 0).sum())}
        print(f"{name:26s} avg vs current {diff.mean():+.2f}%/yr, better in {(diff > 0).sum()}/{len(diff)} runs", flush=True)
    json.dump(out, open("research/replay/robust.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
