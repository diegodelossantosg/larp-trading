"""Run the News Hunter replay with the current rules and many alternatives; save research/replay/results.json.

    python -m research.replay.experiments

Each alternative changes ONE thing from the current rules. Every result is shown for the whole period and for two
halves (2016-2020 and 2021-2026): a change only counts as better if it helps in both halves.
"""

from __future__ import annotations

import dataclasses
import json
import time

from .sim import Data, Params, run, summary

BASE = Params()
HALVES = {"2016-2020": ("2016-03-01", "2021-01-01"), "2021-2026": ("2021-01-01", "2099-01-01")}

ALTERNATIVES: dict[str, dict] = {
    # what counts as a buy signal
    "drop 1.5%": {"drop": 0.015}, "drop 3%": {"drop": 0.03}, "drop 4%": {"drop": 0.04}, "drop 6%": {"drop": 0.06},
    "any swing (no ATR floor)": {"atr_min": 0.0}, "swing 2%+": {"atr_min": 2.0}, "swing 4%+": {"atr_min": 4.0},
    "swing 5%+": {"atr_min": 5.0},
    "liquid $100M+": {"liquid_m": 100.0}, "liquid $1B+": {"liquid_m": 1000.0},
    "no heavy-volume rule": {"max_vr": None}, "heavy volume 1.5x": {"max_vr": 1.5}, "heavy volume 3x": {"max_vr": 3.0},
    "no earnings blackout": {"blackout": 0}, "buy earnings drops too": {"earnings_news": True},
    # when and how it buys
    "reads 1h of news": {"window_h": 1.0}, "reads 4h of news": {"window_h": 4.0},
    "mornings only (10-11h)": {"hours": (10, 11)}, "afternoons only (14-15h)": {"hours": (14, 15)},
    "smallest fall first": {"order": "smallest_fall"}, "random pick": {"order": "random"},
    # how much
    "5% per buy": {"size": 0.05}, "15% per buy": {"size": 0.15}, "20% per buy": {"size": 0.20},
    "1 buy a day": {"per_day": 1}, "3 buys a day": {"per_day": 3}, "5 buys a day": {"per_day": 5},
    "max 20% in stocks": {"cap": 0.20}, "max 50% in stocks": {"cap": 0.50}, "max 100% in stocks": {"cap": 1.0},
    "3 buys/day, 15% each, max 60%": {"per_day": 3, "size": 0.15, "cap": 0.60},
    # when it sells
    "hold 3 days": {"hold": 3}, "hold 5 days": {"hold": 5}, "hold 15 days": {"hold": 15}, "hold 20 days": {"hold": 20},
    "hold 40 days": {"hold": 40},
    "stop 10%": {"stop": 0.10}, "stop 15%": {"stop": 0.15}, "stop 30%": {"stop": 0.30}, "no stop": {"stop": None},
}


def one(D, P) -> dict:
    out = {"all": summary(run(D, P))}
    for name, (a, b) in HALVES.items():
        out[name] = summary(run(D, dataclasses.replace(P, start=max(a, P.start), end=b)))
    return out


def main() -> None:
    t0 = time.time()
    D = Data()
    res = {"base": one(D, BASE)}
    print(f"base: {res['base']['all']}", flush=True)
    for name, ch in ALTERNATIVES.items():
        res[name] = one(D, dataclasses.replace(BASE, **ch))
        a = res[name]["all"]
        print(f"{name:34s} extra/yr {a.get('extra_per_year_pct')}%  buys/yr {a.get('buys_per_year')}  "
              f"halves {res[name]['2016-2020'].get('extra_per_year_pct')} / {res[name]['2021-2026'].get('extra_per_year_pct')}",
              flush=True)
    # each news source on its own
    res["only the saved 2016-2026 news events"] = one(D, dataclasses.replace(BASE, sources=("alpaca_events",)))
    # the full whole-market feed exists only from late April 2026: the most realistic replay, but short
    start = str(D.n12.t.min().date()) if D.n12 is not None else "2026-04-28"
    for name, src in (("full feed, last months", ("alpaca_12m",)), ("saved events, same months", ("alpaca_events",))):
        res[name] = {"all": summary(run(D, dataclasses.replace(BASE, sources=src, start=start)))}
        print(name, res[name]["all"], flush=True)
    with open("research/replay/results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
