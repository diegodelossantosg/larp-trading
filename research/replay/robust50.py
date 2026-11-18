"""Stricter luck check: 30 runs on different random halves of the news (paths differ much more than with 80%).

    python -m research.replay.robust50
"""
import dataclasses, json

import numpy as np

from .sim import Data, Params, run, summary

CANDIDATES = {"no stop": {"stop": None}, "stop 30%": {"stop": 0.30}, "reads 4h of news": {"window_h": 4.0},
              "stop 15%": {"stop": 0.15}, "no heavy-volume rule": {"max_vr": None}, "swing 4%+": {"atr_min": 4.0}}


def main():
    D = Data()
    base = np.array([summary(run(D, dataclasses.replace(Params(), keep=0.5), seed=s))["extra_per_year_pct"] for s in range(30)])
    out = {"base": {"avg": float(base.mean()), "beats_spy": int((base > 0).sum())}}
    print(f"current rules, 30 halves: {base.mean():+.2f}%/yr vs SPY, beats SPY in {(base > 0).sum()}/30", flush=True)
    for name, ch in CANDIDATES.items():
        v = np.array([summary(run(D, dataclasses.replace(Params(), keep=0.5, **ch), seed=s))["extra_per_year_pct"] for s in range(30)])
        d = v - base
        out[name] = {"avg": float(v.mean()), "gain": float(d.mean()), "wins": int((d > 0).sum())}
        print(f"{name:24s} {v.mean():+.2f}%/yr, vs current {d.mean():+.2f}, better in {(d > 0).sum()}/30", flush=True)
    json.dump(out, open("research/replay/robust50.json", "w"), indent=1)


if __name__ == "__main__":
    main()
