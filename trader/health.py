"""Daily market health check: a fixed set of free, well-studied risk gauges.

- Fear gauge (VIX), from Cboe
- 10-year and 2-year Treasury yields and the yield curve, from FRED (US Federal Reserve)
- High-yield credit spread (what risky companies pay over Treasuries), from FRED
- Breadth: share of our ETFs above their 200-day and 50-day averages (from our own data)

When those sites can't be reached, the same questions are answered from Alpaca prices instead:
fear from SPY's recent volatility, rates from 7-10y Treasury ETF (IEF) moves, credit stress from
high-yield bonds (HYG) vs Treasuries (IEF).

Each gauge gets a plain label, and together they set the market mood: healthy, cautious or stressed.
Claude gets this every morning; the watcher becomes stricter when the mood is stressed.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from datetime import datetime

log = logging.getLogger(__name__)

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _get(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "larp-trading/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode()
    except Exception as e:  # network trouble must never stop trading
        log.warning("health: could not fetch %s: %s", url, e)
        return None


def parse_series(text: str | None) -> list[tuple[str, float]]:
    """(date, value) rows from a Cboe or FRED CSV, oldest first; missing values skipped."""
    if not text:
        return []
    rows = list(csv.reader(io.StringIO(text)))
    header = [h.strip().upper() for h in rows[0]]
    col = header.index("CLOSE") if "CLOSE" in header else len(header) - 1
    out = []
    for r in rows[1:]:
        try:
            d = r[0].strip()
            if "/" in d:  # Cboe: MM/DD/YYYY
                d = datetime.strptime(d, "%m/%d/%Y").strftime("%Y-%m-%d")
            out.append((d, float(r[col])))
        except (ValueError, IndexError):
            continue  # FRED marks holidays with "."
    return out


def _change(series: list[tuple[str, float]], n: int) -> float | None:
    return series[-1][1] - series[-1 - n][1] if len(series) > n else None


def _ret(df, n: int) -> float | None:
    c = df["close"]
    return float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else None


def build(technicals: dict[str, dict] | None = None, bars: dict | None = None, fetch=_get) -> dict:
    vix = parse_series(fetch(VIX_URL))
    y10 = parse_series(fetch(FRED_URL.format("DGS10")))
    y2 = parse_series(fetch(FRED_URL.format("DGS2")))
    hy = parse_series(fetch(FRED_URL.format("BAMLH0A0HYM2")))
    bars = bars or {}
    gauges: dict[str, dict] = {}
    stress = 0

    if vix:
        v, ch = vix[-1][1], _change(vix, 5)
        label = "calm" if v < 15 else "normal" if v < 22 else "nervous" if v < 30 else "panic"
        stress += {"calm": 0, "normal": 0, "nervous": 1, "panic": 2}[label]
        gauges["fear"] = {"title": "Fear (VIX)", "display": f"{v:.1f}", "value": round(v, 2),
                          "change_5d": None if ch is None else round(ch, 2), "label": label, "source": "Cboe VIX"}
    elif "SPY" in bars and len(bars["SPY"]) > 21:
        vol = float(bars["SPY"]["close"].pct_change().tail(20).std() * (252 ** 0.5) * 100)
        label = "calm" if vol < 12 else "normal" if vol < 20 else "nervous" if vol < 30 else "panic"
        stress += {"calm": 0, "normal": 0, "nervous": 1, "panic": 2}[label]
        gauges["fear"] = {"title": "Fear (S&P swings)", "display": f"{vol:.0f}%", "value": round(vol, 1),
                          "label": label, "source": "SPY 20-day volatility (annualised)"}

    if y10 and y2:
        curve = y10[-1][1] - y2[-1][1]
        ch10 = _change(y10, 20)
        label = "inverted curve (recession warning)" if curve < 0 else "normal curve"
        if ch10 is not None and ch10 > 0.4:
            stress += 1
            label += ", yields rising fast"
        gauges["rates"] = {"title": "10-year rate", "display": f"{y10[-1][1]:.2f}%", "ten_year_pct": y10[-1][1],
                           "two_year_pct": y2[-1][1], "curve_10y_minus_2y": round(curve, 2),
                           "ten_year_change_20d_bp": None if ch10 is None else round(ch10 * 100),
                           "label": label, "source": "FRED"}
    elif "IEF" in bars and (r := _ret(bars["IEF"], 20)) is not None:
        rising = r < -0.02  # bond prices down >2% in a month = yields rising fast
        stress += 1 if rising else 0
        gauges["rates"] = {"title": "Bonds (rates)", "display": f"{r * 100:+.1f}%", "ief_return_20d_pct": round(r * 100, 2),
                           "label": "yields rising fast" if rising else "yields falling" if r > 0.02 else "stable",
                           "source": "IEF 7-10y Treasury ETF, 20-day change"}

    if hy:
        s, ch = hy[-1][1], _change(hy, 20)
        rising = ch is not None and ch > 0.5
        label = "stressed" if s > 5 or rising else "normal" if s > 3.5 else "relaxed"
        stress += 1 if label == "stressed" else 0
        gauges["credit"] = {"title": "Credit stress", "display": f"{s:.2f}%", "value_pct": s,
                            "change_20d": None if ch is None else round(ch, 2), "label": label, "source": "FRED"}
    elif "HYG" in bars and "IEF" in bars:
        h, i = _ret(bars["HYG"], 20), _ret(bars["IEF"], 20)
        if h is not None and i is not None:
            gap = (h - i) * 100
            label = "stressed" if gap < -1.5 else "normal" if gap < 0.5 else "relaxed"
            stress += 1 if label == "stressed" else 0
            gauges["credit"] = {"title": "Credit stress", "display": f"{gap:+.1f}%", "hyg_vs_ief_20d_pct": round(gap, 2),
                                "label": label, "source": "high-yield (HYG) vs Treasuries (IEF), 20 days"}

    if technicals:
        rows = [t for t in technicals.values() if t.get("vs_sma200_pct") is not None]
        if rows:
            above200 = sum(t["vs_sma200_pct"] > 0 for t in rows) / len(rows)
            above50 = sum((t.get("vs_sma50_pct") or 0) > 0 for t in rows) / len(rows)
            label = "broad" if above200 > 0.6 else "mixed" if above200 > 0.4 else "weak"
            stress += 1 if label == "weak" else 0
            gauges["breadth"] = {"title": "ETFs in uptrend", "display": f"{above200 * 100:.0f}%",
                                 "above_200d_pct": round(above200 * 100), "above_50d_pct": round(above50 * 100),
                                 "label": label}

    mood = "unknown" if not gauges else "stressed" if stress >= 2 else "cautious" if stress == 1 else "healthy"
    return {"mood": mood, "stress_points": stress, "gauges": gauges,
            "checked": datetime.now().isoformat(timespec="minutes")}
