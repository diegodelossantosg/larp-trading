"""Data for the realistic News Hunter replay (research/replay): news with exact times, hourly and daily prices.

    python -m trader.replay_data edgar   --out research/replay/edgar_8k.csv.gz
    python -m trader.replay_data news12  --out research/replay/news_12m.csv.gz
    python -m trader.replay_data daily   --out research/replay/daily.npz
    python -m trader.replay_data needed  --out research/replay/needed.csv.gz
    python -m trader.replay_data hourly  --needed research/replay/needed.csv.gz --chunk 0 --chunks 8 --out part0.csv.gz

Sources (both are read by the live News Hunter too):
  - SEC 8-K filings (EDGAR), 2016 on: the official company announcements, with the minute they went public.
  - Alpaca/Benzinga headlines for the whole market (Alpaca now keeps only about the last 12 months).
  - The earlier Alpaca news event files (research/events_*.csv.gz) add the 2016-2026 headline times we saved.
Prices: Alpaca SIP, split and dividend adjusted. Hourly closes at 10:00 ... 15:00 New York time (the moments
the hourly News Hunter looks), and the volume traded so far that day, only for the days some news could be seen.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import time as _time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger("replay_data")
ET = ZoneInfo("America/New_York")
START = "2016-01-01"
CHECKS = (10, 11, 12, 13, 14, 15)          # hours (ET) at which the replayed hunter looks
SEC_UA = "larp-trading research bot trading-agent@users.noreply.github.com"   # SEC wants a name and an e-mail
EVENT_FILES = ("events_long", "events_wide", "events_early", "events_mid", "events")


def universe() -> list[str]:
    from .liquid import LIQUID_STOCKS
    from .universe import BIG_STOCKS
    z = np.load("research/lab.npz", allow_pickle=False)
    etfs = set(z["etfs"])
    syms = {s for s in z["symbols"] if s not in etfs} | set(BIG_STOCKS) | set(LIQUID_STOCKS)
    return sorted(s for s in syms if s.isalpha() and len(s) <= 5)


# ---------------- SEC 8-K ----------------

def _sec_get(url: str):
    import requests
    for attempt in range(6):
        try:
            r = requests.get(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}, timeout=30)
        except Exception as e:  # network hiccup
            log.warning("%s: %s", url, e)
            _time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429, 503):
            log.warning("SEC %s for %s", r.status_code, url)
            _time.sleep(2 ** attempt + 1)
            continue
        r.raise_for_status()
        _time.sleep(0.12)                    # SEC asks for at most 10 requests a second
        return r.json()
    return None


def edgar(out: str) -> None:
    tickers = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if not tickers:
        raise SystemExit("SEC ticker list unavailable")
    cik = {v["ticker"].upper(): int(v["cik_str"]) for v in tickers.values()}
    rows, missing = [], []
    syms = universe()
    for n, sym in enumerate(syms):
        c = cik.get(sym)
        if c is None:
            missing.append(sym)
            continue
        sub = _sec_get(f"https://data.sec.gov/submissions/CIK{c:010d}.json")
        if not sub:
            missing.append(sym)
            continue
        blocks = [sub.get("filings", {}).get("recent", {})]
        for f in sub.get("filings", {}).get("files", []):
            if f.get("filingTo", "9999") >= START:
                b = _sec_get(f"https://data.sec.gov/submissions/{f['name']}")
                if b:
                    blocks.append(b)
        for b in blocks:
            forms = b.get("form", [])
            for i, form in enumerate(forms):
                if form not in ("8-K", "8-K/A"):
                    continue
                acc = b["acceptanceDateTime"][i]
                if not acc or acc[:10] < START:
                    continue
                rows.append({"symbol": sym, "cik": c, "form": form, "accepted": acc, "items": b.get("items", [""] * len(forms))[i]})
        if n % 50 == 0:
            log.info("edgar: %d/%d companies, %d filings", n + 1, len(syms), len(rows))
    df = pd.DataFrame(rows)
    # acceptanceDateTime is given as UTC ('Z'); keep New York time for the replay
    t = pd.to_datetime(df["accepted"], utc=True).dt.tz_convert(ET)
    df["time_et"] = t.dt.strftime("%Y-%m-%d %H:%M")
    df.drop(columns=["accepted"]).sort_values("time_et").to_csv(out, index=False, compression="gzip")
    log.info("edgar: %d filings for %d companies; no CIK for %d (%s...)", len(df), df.symbol.nunique(), len(missing), missing[:15])


# ---------------- Alpaca news, whole market ----------------

def news12(out: str, days: int = 400) -> None:
    import requests

    from .config import load_config
    cfg = load_config()
    h = {"APCA-API-KEY-ID": cfg.alpaca_api_key, "APCA-API-SECRET-KEY": cfg.alpaca_secret_key}
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {"start": start.isoformat(), "end": end.isoformat(), "limit": 50, "sort": "asc", "include_content": "false"}
    rows, pages = [], 0
    while True:
        for attempt in range(6):
            r = requests.get("https://data.alpaca.markets/v1beta1/news", params=params, headers=h, timeout=30)
            if r.status_code != 429:
                break
            _time.sleep(2 ** attempt)
        r.raise_for_status()
        body = r.json()
        for a in body.get("news", []):
            t = pd.Timestamp(a["created_at"]).tz_convert(ET)
            rows.append({"time_et": t.strftime("%Y-%m-%d %H:%M:%S"), "symbols": " ".join(a.get("symbols", [])[:8]),
                         "source": a.get("source", ""), "headline": (a.get("headline") or "")[:200].replace("\n", " ")})
        pages += 1
        if pages % 200 == 0:
            log.info("news12: %d headlines, at %s", len(rows), rows[-1]["time_et"] if rows else "-")
        if not body.get("next_page_token"):
            break
        params["page_token"] = body["next_page_token"]
    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
    log.info("news12: %d headlines from %s", len(rows), rows[0]["time_et"] if rows else "-")


# ---------------- daily bars ----------------

def daily(out: str) -> None:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from .config import load_config
    cfg = load_config()
    client = StockHistoricalDataClient(cfg.alpaca_api_key, cfg.alpaca_secret_key)
    syms = universe() + ["SPY"]
    frames = {}
    for i in range(0, len(syms), 50):
        try:
            df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms[i:i + 50], timeframe=TimeFrame.Day,
                                                        start=pd.Timestamp("2015-06-01", tz="UTC"),
                                                        end=datetime.now(timezone.utc) - timedelta(minutes=20),
                                                        adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
        except Exception as e:
            log.warning("daily batch %d: %s", i, e)
            continue
        for sym in df.index.get_level_values("symbol").unique():
            d = df.xs(sym, level="symbol").sort_index()
            d.index = pd.DatetimeIndex([ts.tz_convert(ET).date() for ts in d.index])
            frames[sym] = d
        log.info("daily: %d/%d", min(i + 50, len(syms)), len(syms))
    dates = frames["SPY"].index
    names = sorted(frames)
    arr = {k: np.full((len(dates), len(names)), np.nan, np.float32) for k in ("open", "high", "low", "close", "volume")}
    for j, s in enumerate(names):
        d = frames[s].reindex(dates)
        for k in arr:
            arr[k][:, j] = d[k].to_numpy(np.float32)
    np.savez_compressed(out, dates=np.array([str(d) for d in dates]), symbols=np.array(names), **arr)
    log.info("daily: %d symbols x %d days", len(names), len(dates))


# ---------------- which (symbol, day) pairs need hourly prices ----------------

def _visible_day(ts: pd.Timestamp) -> str | None:
    """The trading day on which an hourly hunter (10:00..15:00, 2-hour look-back) would see this news, if any."""
    hm = ts.hour + ts.minute / 60
    if ts.weekday() >= 5 or hm < CHECKS[0] - 2 or hm >= CHECKS[-1]:
        return None
    return ts.strftime("%Y-%m-%d")


def news_times() -> pd.DataFrame:
    """Every timed news item we have: symbol, time_et, source, text."""
    parts = []
    try:
        e = pd.read_csv("research/replay/edgar_8k.csv.gz", dtype=str)
        parts.append(pd.DataFrame({"symbol": e.symbol, "time_et": e.time_et, "source": "sec_8k", "text": "8-K items " + e["items"].fillna("")}))
    except FileNotFoundError:
        pass
    try:
        n = pd.read_csv("research/replay/news_12m.csv.gz", dtype=str).dropna(subset=["symbols"])
        n = n.assign(symbol=n.symbols.str.split()).explode("symbol")
        parts.append(pd.DataFrame({"symbol": n.symbol, "time_et": n.time_et.str[:16], "source": "alpaca_12m", "text": n.headline}))
    except FileNotFoundError:
        pass
    for f in EVENT_FILES:
        try:
            ev = pd.read_csv(f"research/{f}.csv.gz", usecols=["kind", "symbol", "news_time", "headline"], dtype=str)
        except (FileNotFoundError, ValueError):
            continue
        ev = ev[ev.kind == "news"].dropna(subset=["news_time"])
        # event files store the news time already in New York time
        parts.append(pd.DataFrame({"symbol": ev.symbol, "time_et": ev.news_time, "source": "alpaca_events", "text": ev.headline}))
    df = pd.concat(parts, ignore_index=True).dropna(subset=["symbol", "time_et"])
    return df.drop_duplicates(["symbol", "time_et", "source"])


def needed(out: str) -> None:
    uni = set(universe())
    df = news_times()
    df = df[df.symbol.isin(uni)]
    days = pd.to_datetime(df.time_et).map(_visible_day)
    pairs = pd.DataFrame({"symbol": df.symbol, "day": days}).dropna().drop_duplicates()
    pairs = pairs[pairs.day >= START]
    pairs.sort_values(["symbol", "day"]).to_csv(out, index=False, compression="gzip")
    log.info("needed: %d symbol-days, %d symbols", len(pairs), pairs.symbol.nunique())


# ---------------- hourly prices at the check times ----------------

def _get_rest(params: dict, headers: dict, pause: float = 0.3) -> dict | None:
    import requests
    for attempt in range(8):
        _time.sleep(pause)
        try:
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars", params=params, headers=headers, timeout=60)
        except Exception as e:
            log.warning("bars request: %s", e)
            _time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            _time.sleep(10 * (attempt + 1))
            continue
        if r.status_code != 200:
            log.warning("bars HTTP %s %s", r.status_code, r.text[:120])
            return None
        return r.json()
    return None


def _rows_from_bars(sym: str, bars: list[dict]) -> list[dict]:
    if not bars:
        return []
    df = pd.DataFrame(bars)
    t = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET)
    df = pd.DataFrame({"day": t.dt.strftime("%Y-%m-%d"), "hour": t.dt.hour, "close": df["c"], "volume": df["v"]})
    df = df[(df.hour >= 9) & (df.hour <= 15)]       # the 9:00 bar holds 9:30-10:00 (plus 30 minutes before the open)
    rows = []
    for day, d in df.groupby("day"):
        d = d.sort_values("hour")
        cum = d.volume.cumsum().to_numpy()
        row = {"symbol": sym, "day": day}
        # the price at HH:00 is the close of the bar that started an hour earlier
        for hh in CHECKS:
            m = (d.hour < hh).to_numpy()
            row[f"p{hh}"] = round(float(d.close.to_numpy()[m][-1]), 4) if m.any() else ""
            row[f"v{hh}"] = int(cum[m][-1]) if m.any() else ""
        rows.append(row)
    return rows


def hourly(needed_path: str, chunk: int, chunks: int, out: str) -> None:
    """Hourly prices for exactly the needed (symbol, day) pairs, one request per day for all its symbols.

    Rows already in research/replay/hourly.csv.gz are kept and not asked again."""
    from .config import load_config
    cfg = load_config()
    headers = {"APCA-API-KEY-ID": cfg.alpaca_api_key, "APCA-API-SECRET-KEY": cfg.alpaca_secret_key}
    need = pd.read_csv(needed_path, dtype=str)
    try:
        have = pd.read_csv("research/replay/hourly.csv.gz", dtype=str)
    except FileNotFoundError:
        have = pd.DataFrame(columns=["symbol", "day"])
    got = set(zip(have.symbol, have.day))
    days = sorted(set(need.day))
    spy_need = pd.DataFrame({"symbol": "SPY", "day": days})
    need = pd.concat([need, spy_need]).drop_duplicates()
    need = need[[(s, d) not in got for s, d in zip(need.symbol, need.day)]]
    by_day = need.groupby("day").symbol.apply(sorted).to_dict()
    mine = [d for k, d in enumerate(sorted(by_day)) if k % chunks == chunk]
    log.info("hourly: %d rows already, %d symbol-days to fetch over %d days", len(have), len(need), len(mine))
    rows = []
    for k, day in enumerate(mine):
        syms = by_day[day]
        start = pd.Timestamp(f"{day} 09:00", tz=ET).tz_convert("UTC")
        end = pd.Timestamp(f"{day} 16:00", tz=ET).tz_convert("UTC")
        for i in range(0, len(syms), 100):
            params = {"symbols": ",".join(syms[i:i + 100]), "timeframe": "1Hour", "limit": 10000, "adjustment": "all",
                      "feed": "sip", "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
            bars: dict[str, list] = {}
            while True:
                body = _get_rest(params, headers)
                if body is None:
                    break
                for s, b in (body.get("bars") or {}).items():
                    bars.setdefault(s, []).extend(b)
                if not body.get("next_page_token"):
                    break
                params["page_token"] = body["next_page_token"]
            for s, b in bars.items():
                rows += _rows_from_bars(s, b)
        if k % 100 == 0 or k == len(mine) - 1:
            log.info("hourly: %d/%d days, %d new rows", k + 1, len(mine), len(rows))
            pd.concat([have, pd.DataFrame(rows)]).to_csv(out, index=False, compression="gzip")   # keep what we have
    pd.concat([have, pd.DataFrame(rows)]).to_csv(out, index=False, compression="gzip")
    log.info("hourly done: %d rows in total", len(have) + len(rows))


HALF_CHECKS = tuple(range(10 * 60, 16 * 60, 30))     # 10:00, 10:30, ... 15:30 (minutes after midnight, ET)


def _half_rows(sym: str, bars: list[dict]) -> list[dict]:
    """Price and volume so far at every half hour from 10:00 to 15:30 (from 30-minute bars)."""
    if not bars:
        return []
    df = pd.DataFrame(bars)
    t = pd.to_datetime(df["t"], utc=True).dt.tz_convert(ET)
    df = pd.DataFrame({"day": t.dt.strftime("%Y-%m-%d"), "m": t.dt.hour * 60 + t.dt.minute, "close": df["c"], "volume": df["v"]})
    df = df[(df.m >= 9 * 60 + 30) & (df.m < 16 * 60)]
    rows = []
    for day, d in df.groupby("day"):
        d = d.sort_values("m")
        ends = (d.m + 30).to_numpy()            # a bar's close is the price at the end of its 30 minutes
        cum = d.volume.cumsum().to_numpy()
        row = {"symbol": sym, "day": day}
        for c in HALF_CHECKS:
            k = ends <= c
            tag = f"{c // 60:02d}{c % 60:02d}"
            row[f"p{tag}"] = round(float(d.close.to_numpy()[k][-1]), 4) if k.any() else ""
            row[f"v{tag}"] = int(cum[k][-1]) if k.any() else ""
        rows.append(row)
    return rows


def halfhour(needed_path: str, out: str) -> None:
    """Like hourly(), with 30-minute bars: prices at every half hour, for the days some news could be seen."""
    from .config import load_config
    cfg = load_config()
    headers = {"APCA-API-KEY-ID": cfg.alpaca_api_key, "APCA-API-SECRET-KEY": cfg.alpaca_secret_key}
    need = pd.read_csv(needed_path, dtype=str)
    try:
        have = pd.read_csv(out, dtype=str)
    except FileNotFoundError:
        have = pd.DataFrame(columns=["symbol", "day"])
    got = set(zip(have.symbol, have.day))
    need = pd.concat([need, pd.DataFrame({"symbol": "SPY", "day": sorted(set(need.day))})]).drop_duplicates()
    need = need[[(s, d) not in got for s, d in zip(need.symbol, need.day)]]
    by_day = need.groupby("day").symbol.apply(sorted).to_dict()
    log.info("half-hour: %d rows already, %d symbol-days over %d days", len(have), len(need), len(by_day))
    rows = []
    for k, day in enumerate(sorted(by_day)):
        syms = by_day[day]
        start = pd.Timestamp(f"{day} 09:30", tz=ET).tz_convert("UTC")
        end = pd.Timestamp(f"{day} 16:00", tz=ET).tz_convert("UTC")
        for i in range(0, len(syms), 100):
            params = {"symbols": ",".join(syms[i:i + 100]), "timeframe": "30Min", "limit": 10000, "adjustment": "all",
                      "feed": "sip", "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
            bars: dict[str, list] = {}
            while True:
                body = _get_rest(params, headers)
                if body is None:
                    break
                for s, b in (body.get("bars") or {}).items():
                    bars.setdefault(s, []).extend(b)
                if not body.get("next_page_token"):
                    break
                params["page_token"] = body["next_page_token"]
            for s, b in bars.items():
                rows += _half_rows(s, b)
        if k % 100 == 0 or k == len(by_day) - 1:
            log.info("half-hour: %d/%d days, %d new rows", k + 1, len(by_day), len(rows))
            pd.concat([have, pd.DataFrame(rows)]).to_csv(out, index=False, compression="gzip")
    pd.concat([have, pd.DataFrame(rows)]).to_csv(out, index=False, compression="gzip")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["edgar", "news12", "daily", "needed", "hourly", "halfhour"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--needed", default="research/replay/needed.csv.gz")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=1)
    a = ap.parse_args()
    if a.what == "edgar":
        edgar(a.out)
    elif a.what == "news12":
        news12(a.out)
    elif a.what == "daily":
        daily(a.out)
    elif a.what == "needed":
        needed(a.out)
    elif a.what == "halfhour":
        halfhour(a.needed, a.out)
    else:
        hourly(a.needed, a.chunk, a.chunks, a.out)


if __name__ == "__main__":
    main()
