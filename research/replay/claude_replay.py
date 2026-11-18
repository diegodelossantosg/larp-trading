"""Ask Claude, hour by hour, what the live News Hunter would have bought (July-November 2026).

    python -m research.replay.claude_replay names       # Alpaca's company names (for the wrong-tag filter)
    python -m research.replay.claude_replay ask [--max N]

Why only from July 2026: the model learned from data up to mid-2026; for anything older it may simply know what
happened next, and the test would cheat. Web search is switched off for the same reason (it would find later
articles). Claude gets the same instructions and the same kind of input as the live hunter: the latest 50 headlines
of the last 2 hours (wrong company tags removed), prices of the companies they mention, the S&P 500's move.

It is asked at every hourly check where at least one company passes the code's checks (fell 2%+ vs SPY, liquid,
volatile, no heavy volume, no earnings near), the first time that company shows up that day. Its picks are saved in
research/replay/claude_decisions.json ("YYYY-MM-DD HH" -> picks); research.replay.sim (Params.claude) then trades
only those picks with the live rules (code checks, sizes, limits, 10-day exit).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import subprocess
import tempfile

import pandas as pd

log = logging.getLogger("claude_replay")
OUT = "research/replay/claude_decisions.json"
START = "2026-07-01"


def names() -> None:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    from trader.config import load_config
    cfg = load_config()
    tc = TradingClient(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=True, raw_data=True)
    out = {}
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        for a in tc.get_all_assets(GetAssetsRequest(status=status, asset_class=AssetClass.US_EQUITY)):
            out.setdefault(a["symbol"], a.get("name") or None)
    json.dump(out, open("research/replay/names.json", "w"))
    log.info("names: %d symbols", len(out))


def replay_prompt() -> str:
    from trader.brain import NEWS_PROMPT
    return NEWS_PROMPT + """

THIS IS A REPLAY TEST of a past hour. Web search is NOT available: judge only from the headlines and prices given.
Do not use anything you may know about what happened after now_new_york (the price later, later news): decide as
if this moment were the present. Leave holdings_alerts empty."""


def ask_claude(cfg, system: str, schema: dict, prompt: str) -> tuple[dict | None, str | None]:
    """The live hunter's call, with no tools (no web search)."""
    if cfg.claude_backend == "api":
        import anthropic
        try:
            resp = anthropic.Anthropic(api_key=cfg.anthropic_api_key).messages.create(
                model=cfg.claude_model, max_tokens=8000, system=system, messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}})
        except anthropic.APIError as e:
            return None, str(e)[:200]
        text = next((b.text for b in reversed(resp.content) if b.type == "text"), "")
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            return None, "no JSON"
    cmd = ["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema), "--system-prompt", system,
           "--tools", "Read", "--permission-mode", "dontAsk", "--strict-mcp-config", "--no-session-persistence",
           "--model", cfg.claude_model, "--effort", "high"]
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd=workdir,
                                  env=os.environ.copy())
            out = json.loads(proc.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            return None, type(e).__name__
    data = out.get("structured_output")
    if out.get("is_error") or not isinstance(data, dict):
        return None, str(out.get("result"))[:200]
    return data, None


def context_for(D, day: str, hh: int, also: list[str] = ()) -> dict:
    """What the live hunter would have been given at day hh:00 (headlines, prices of mentioned companies, SPY)."""
    import numpy as np
    now = pd.Timestamp(f"{day} {hh:02d}:00")
    n = D.n12
    w = n[(n.t >= now - pd.Timedelta(hours=2)) & (n.t < now)]
    if "named" in w:
        w = w[w.named]
    w = w.iloc[-50:]
    heads = [{"time": t.strftime("%Y-%m-%d %H:%M ET"), "headline": h, "symbols": s.split(), "source": "Alpaca"}
             for t, h, s in zip(w.t, w.headline, w.symbols)][::-1]
    k = list(range(10, 16)).index(hh)
    i = D.di[day]
    spy = D.hp[("SPY", day)][k] / D.c[i - 1, D.spy] - 1
    prices = {}
    for s in list(also) + sorted({s for h in heads for s in h["symbols"]} - set(also))[: max(0, 40 - len(also))]:
        hp, j = D.hp.get((s, day)), D.col.get(s)
        if hp and j is not None and np.isfinite(hp[k]) and np.isfinite(D.c[i - 1, j]):
            prices[s] = {"price": round(hp[k], 2), "change_today_pct": round((hp[k] / D.c[i - 1, j] - 1) * 100, 2)}
    return {"now_new_york": f"{day} {hh:02d}:00", "headlines": heads, "prices_of_mentioned": prices,
            "spy_change_today_pct": round(spy * 100, 2), "holdings": {}, "news_about_our_holdings_48h": [],
            "corporate_actions_for_our_holdings": [], "buys_left_today": 2,
            "note": "replay of a past hour: no web search; big_company_moves_today is not available in this test"}


def ask(max_calls: int) -> None:
    from trader.brain import NEWS_SCHEMA
    from trader.config import load_config

    from .sim import Data, Params, run
    cfg = load_config()
    D = Data()
    P = dataclasses.replace(Params(), sources=("alpaca_12m",), start=START, per_day=99, cap=99.0, size=0.0001,
                            collect=True)
    points = run(D, P).extra["points"]
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    system = replay_prompt()
    calls = 0
    for p in points:
        key = f"{p['day']} {p['hour']:02d}"
        if key in done:
            continue
        if calls >= max_calls:
            break
        ctx = context_for(D, p["day"], p["hour"], p["candidates"])
        prompt = "Latest headlines, prices and portfolio (JSON):\n" + json.dumps(ctx, separators=(",", ":"))
        data, err = ask_claude(cfg, system, NEWS_SCHEMA, prompt)
        calls += 1
        if data is None:
            log.warning("%s: Claude failed: %s", key, err)
            continue
        done[key] = {"candidates": p["candidates"], "ideas": [x["symbol"].upper().strip() for x in data.get("ideas", [])],
                     "summary": data.get("summary", ""), "reasons": [x.get("plain_reason", "") for x in data.get("ideas", [])]}
        json.dump(done, open(OUT, "w"), indent=1)
        log.info("%s candidates %s -> Claude picks %s", key, p["candidates"], done[key]["ideas"])
    log.info("asked %d times; %d of %d checks decided", calls, len(done), len(points))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["names", "ask"])
    ap.add_argument("--max", type=int, default=200)
    a = ap.parse_args()
    names() if a.what == "names" else ask(a.max)


if __name__ == "__main__":
    main()
