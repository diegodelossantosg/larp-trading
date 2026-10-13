"""Daily market health check: a fixed set of free, well-studied risk gauges.

- Fear gauge (VIX), from Cboe
- 10-year and 2-year Treasury yields and the yield curve, from FRED (US Federal Reserve)
- High-yield credit spread (what risky companies pay over Treasuries), from FRED
- Breadth: share of our ETFs above their 200-day and 50-day averages (from our own data)

Each gauge gets a plain label, and together they set the market mood: calm, normal or stressed.
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


def build(technicals: dict[str, dict] | None = None, fetch=_get) -> dict:
    vix = parse_series(fetch(VIX_URL))
    y10 = parse_series(fetch(FRED_URL.format("DGS10")))
    y2 = parse_series(fetch(FRED_URL.format("DGS2")))
    hy = parse_series(fetch(FRED_URL.format("BAMLH0A0HYM2")))
    gauges: dict[str, dict] = {}
    stress = 0

    if vix:
        v, ch = vix[-1][1], _change(vix, 5)
        label = "calm" if v < 15 else "normal" if v < 22 else "nervous" if v < 30 else "panic"
        stress += {"calm": 0, "normal": 0, "nervous": 1, "panic": 2}[label]
        gauges["fear_vix"] = {"value": round(v, 2), "change_5d": None if ch is None else round(ch, 2),
                              "label": label, "as_of": vix[-1][0]}
    if y10 and y2:
        curve = y10[-1][1] - y2[-1][1]
        ch10 = _change(y10, 20)
        gauges["rates"] = {"ten_year_pct": y10[-1][1], "two_year_pct": y2[-1][1],
                           "curve_10y_minus_2y": round(curve, 2),
                           "ten_year_change_20d_bp": None if ch10 is None else round(ch10 * 100),
                           "label": "inverted curve (recession warning)" if curve < 0 else "normal curve",
                           "as_of": y10[-1][0]}
        if ch10 is not None and ch10 > 0.4:
            stress += 1
            gauges["rates"]["label"] += ", yields rising fast"
    if hy:
        s, ch = hy[-1][1], _change(hy, 20)
        rising = ch is not None and ch > 0.5
        label = "stressed" if s > 5 or rising else "normal" if s > 3.5 else "relaxed"
        stress += 1 if label == "stressed" else 0
        gauges["credit_spread"] = {"value_pct": s, "change_20d": None if ch is None else round(ch, 2),
                                   "label": label, "as_of": hy[-1][0]}
    if technicals:
        rows = [t for t in technicals.values() if t.get("vs_sma200_pct") is not None]
        if rows:
            above200 = sum(t["vs_sma200_pct"] > 0 for t in rows) / len(rows)
            above50 = sum((t.get("vs_sma50_pct") or 0) > 0 for t in rows) / len(rows)
            label = "broad" if above200 > 0.6 else "mixed" if above200 > 0.4 else "weak"
            stress += 1 if label == "weak" else 0
            gauges["breadth"] = {"above_200d_pct": round(above200 * 100), "above_50d_pct": round(above50 * 100),
                                 "label": label}

    mood = "unknown" if not gauges else "stressed" if stress >= 2 else "cautious" if stress == 1 else "healthy"
    return {"mood": mood, "stress_points": stress, "gauges": gauges,
            "checked": datetime.now().isoformat(timespec="minutes")}
