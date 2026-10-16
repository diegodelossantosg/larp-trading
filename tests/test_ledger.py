import pandas as pd

from trader import ledger


def test_partial_and_full_sales_use_average_cost():
    fills = [
        {"id": "b1", "symbol": "XLE", "side": "buy", "qty": 2.0, "price": 60.0, "time": "2026-10-12T14:00:00+00:00"},
        {"id": "b2", "symbol": "XLE", "side": "buy", "qty": 2.0, "price": 70.0, "time": "2026-10-13T14:00:00+00:00"},
        {"id": "s1", "symbol": "XLE", "side": "sell", "qty": 1.0, "price": 80.0, "time": "2026-10-16T14:00:00+00:00"},
        {"id": "b3", "symbol": "IGV", "side": "buy", "qty": 1.0, "price": 112.0, "time": "2026-10-12T14:00:00+00:00"},
        {"id": "s2", "symbol": "IGV", "side": "sell", "qty": 1.0, "price": 107.24, "time": "2026-10-16T14:00:00+00:00"},
    ]
    decisions = [{"date": "2026-10-13", "source": "watch", "orders": [{"order_id": "b2"}]}]
    spy = pd.Series([100.0, 99.0], index=pd.to_datetime(["2026-10-12", "2026-10-16"]))
    rows = ledger.sales(fills, decisions, spy)
    xle = next(r for r in rows if r["symbol"] == "XLE")
    assert xle["avg_buy_price"] == 65.0 and xle["pnl"] == 15.0 and not xle["full"]
    assert xle["bought_by"] == "claude" and xle["days"] == 4
    igv = next(r for r in rows if r["symbol"] == "IGV")
    assert igv["full"] and igv["pnl_pct"] == -4.25 and igv["vs_spy_pct"] == -3.25
    s = ledger.summary(rows)
    assert s["won"] == 1 and s["lost"] == 1 and s["total_pnl"] == round(15.0 - 4.76, 2)
