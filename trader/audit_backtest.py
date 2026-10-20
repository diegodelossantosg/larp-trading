"""Two checks before giving company stocks wider stops and longer holds.

    python -m trader.audit_backtest --years 8.5 --news-months 24

1. No "winners list": instead of today's big companies (all of which survived and grew), use the
   ~100 most-traded US companies as of early 2018, including ones that later shrank, crashed or were
   delisted. Re-run "buy after a >2% drop" and "monthly strongest trends" with every keep-days/stop.
2. Real news: buy big companies one hour after news knocked them down more than 2% vs SPY (two
   years of Alpaca/Benzinga headlines), then hold 3, 5, 10 or 20 days with each stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from . import exit_backtest as eb
from . import news_backtest as nb
from .universe import BIG_STOCKS, looks_like_fund

log = logging.getLogger("audit_backtest")

SAMPLE_START, SAMPLE_END = "2018-01-02", "2018-03-31"
N_COMPANIES = 100
NEWS_HOLDS = (3, 5, 10, 20)


def universe_2018(top: int = N_COMPANIES) -> list[str]:
    """The most-traded US companies (not funds) in Q1 2018, delisted ones included."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    from .config import load_config

    cfg = load_config()
    # raw dicts: new exchanges (e.g. TXSE) aren't in alpaca-py's list yet and would fail validation
    trading = TradingClient(cfg.alpaca_api_key, cfg.alpaca_secret_key, paper=True, raw_data=True)
    assets = []
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        assets += trading.get_all_assets(GetAssetsRequest(status=status, asset_class=AssetClass.US_EQUITY))
    names = {a["symbol"]: a.get("name") or "" for a in assets
             if str(a.get("exchange", "")).upper() in ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
             and a["symbol"].isalpha() and len(a["symbol"]) <= 5 and not looks_like_fund(a.get("name") or "")}
    log.info("candidate companies: %d", len(names))
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    dollar = {}
    symbols = sorted(names)
    for i in range(0, len(symbols), 400):
        try:
            df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbols[i:i + 400], timeframe=TimeFrame.Day,
                                                        start=pd.Timestamp(SAMPLE_START, tz="UTC"),
                                                        end=pd.Timestamp(SAMPLE_END, tz="UTC"), feed=DataFeed.SIP)).df
        except Exception as e:  # one bad batch shouldn't stop the audit
            log.warning("batch %d failed: %s", i, e)
            continue
        if df.empty:
            continue
        dv = (df["close"] * df["volume"]).groupby(level="symbol").mean()
        dollar.update(dv.to_dict())
        log.info("2018 volumes: %d/%d", min(i + 400, len(symbols)), len(symbols))
    ranked = sorted(dollar, key=lambda s: -dollar[s])[:top]
    return ranked


def grids_for(daily: dict[str, pd.DataFrame], symbols: list[str]) -> dict:
    D = {s: eb.Daily(daily[s]) for s in symbols if s in daily and len(daily[s]) > 260}
    spy = eb.Daily(daily["SPY"])

    def sim(key, hold, stop):
        px, i0 = D[key[0]], key[1]
        if i0 + hold >= len(px.c) and px.dates[-1] < spy.dates[-30]:
            hold = len(px.c) - 1 - i0     # delisted before the exit: sell at its last price, loss included
            if hold < 1:
                return None
        return eb.daily_exit(px, spy, i0, hold, stop)

    return {
        "companies_with_data": len(D),
        "after_a_drop": eb.summarize_grid(eb.grid(eb.drop_entries(D, spy), eb.SHORT_HOLDS, sim), (3, "10%")),
        "monthly_trend_picks": eb.summarize_grid(eb.grid(eb.trend_entries(D, spy, 4), eb.LONG_HOLDS, sim), None),
    }


def news_exits(events: list[dict], daily: dict[str, pd.DataFrame]) -> dict:
    """Events where news knocked a big company down >2% vs SPY; buy an hour later, try every exit."""
    D = {s: eb.Daily(df) for s, df in daily.items()}
    spy = D["SPY"]
    drops = [e for e in events if -0.15 < e["reaction"] < -0.02 and e["symbol"] in D]

    def sim(e, hold, stop):
        px = D[e["symbol"]]
        i0 = px.pos.get(pd.Timestamp(e["day"]).date())
        if i0 is None or i0 + hold >= len(px.c):
            return None
        high, out, j = e["entry_price"], None, i0
        for n in range(1, hold + 1):
            j = i0 + n
            if px.l[j] <= high * (1 - stop):
                out = min(high * (1 - stop), px.o[j])
                break
            high = max(high, px.h[j])
        out = px.c[j] if out is None else out
        s1 = spy.pos.get(px.dates[j])
        if s1 is None:
            return None
        return float(out / e["entry_price"] - spy.c[s1] / e["spy_entry"] - eb.COST)

    entries = []
    for k, e in enumerate(drops):
        px = D[e["symbol"]]
        i0 = px.pos.get(pd.Timestamp(e["day"]).date())
        if i0 is not None:
            entries.append((int(e["day"][:4]), px.atr_pct[i0], k))
    rows = eb.grid(entries, NEWS_HOLDS, lambda k, h, s: sim(drops[k], h, s))
    # Years are few here (two), so also rank by the plain average.
    by_avg = sorted(rows, key=lambda r: -r["avg_vs_spy_pct"])
    return {"events": len(drops), "current_setting": [r for r in rows if (r["keep_days"], r["stop"]) == (3, "10%")],
            "best_by_average": by_avg[:10], "worst": by_avg[-3:],
            "by_keep_days_with_smart_4x": [r for r in rows if r["stop"] == "smart 4x daily swing"]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=8.5)
    ap.add_argument("--news-months", type=int, default=24)
    ap.add_argument("--out", default="docs/data/audit_backtest.json")
    args = ap.parse_args()
    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes")}

    companies = universe_2018()
    report["companies_2018"] = companies
    gone = [s for s in companies if s not in BIG_STOCKS]
    report["not_on_todays_list"] = gone
    daily = eb.fetch_daily_ohlc(sorted(set(companies) | set(BIG_STOCKS) | {"SPY"}), args.years)
    log.info("daily bars: %d symbols", len(daily))
    report["audit1_list_as_of_2018"] = grids_for(daily, companies)
    report["audit1_todays_list_same_years"] = grids_for(daily, sorted(BIG_STOCKS))
    log.info("audit 1 done")

    news, bars = nb.fetch(args.news_months)
    events = nb.events(news, bars)
    report["audit2_real_news_drops"] = news_exits(events, daily)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str)[:15000])


if __name__ == "__main__":
    main()
