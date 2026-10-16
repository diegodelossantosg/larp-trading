import numpy as np
import pandas as pd

from trader import report_card as rc


def _bars(symbols, days):
    idx = pd.bdate_range("2026-09-01", periods=40)
    out = {}
    for k, s in enumerate(symbols):
        drift = {"NVDA": 0.01, "XLE": -0.002, "SPY": 0.001}.get(s, 0.0)
        out[s] = pd.DataFrame({"close": 100 * np.exp(np.cumsum(np.full(40, drift)))}, index=idx)
    return out


def test_report_card_groups_picks_by_trade_type():
    decisions = [
        {"date": "2026-09-01", "market_view": "x", "positions": [
            {"symbol": "NVDA", "trade_type": "news_catalyst", "plain_reason": "Big AI contract."},
            {"symbol": "XLE", "trade_type": "macro_theme", "plain_reason": "Oil up."}],
         "orders": [{"symbol": "NVDA", "side": "buy", "status": "filled"},
                    {"symbol": "XLE", "side": "buy", "status": "filled"},
                    {"symbol": "SSO", "side": "buy", "status": "filled"}]},
        {"date": "2026-09-02", "source": "news", "orders": [{"symbol": "AAPL", "side": "buy", "status": "filled"}]},
    ]
    card = rc.build(decisions, "SSO", _bars)
    assert card["all"]["picks"] == 2                       # SSO and the News Hunter's buy don't count
    assert card["by_trade_type"]["news_catalyst"]["vs_spy_20d_avg_pct"] > 0
    assert card["by_trade_type"]["macro_theme"]["vs_spy_20d_avg_pct"] < 0
    assert card["by_kind"]["stock"]["picks"] == 1 and card["recent"][0]["symbol"] in ("NVDA", "XLE")
