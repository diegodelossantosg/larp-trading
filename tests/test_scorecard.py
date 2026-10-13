from trader import journal, scorecard


def test_scorecard_tracks_buys_by_source(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    buys = [{"symbol": "XLE", "side": "buy", "notional": 100, "status": "accepted"},
            {"symbol": "SPY", "side": "buy", "notional": 400, "status": "accepted"}]
    scorecard.record(buys, "claude", {"XLE": 50, "SPY": 500}, 500, "2026-10-12")
    scorecard.record([{"symbol": "SMH", "side": "buy", "notional": 80, "status": "accepted"}],
                     "watch", {"SMH": 200}, 500, "2026-10-12")
    scorecard.update({"XLE": 55, "SMH": 190, "SPY": 505}, 505, held={"XLE"}, date="2026-10-14")  # SMH sold
    rows = journal.read("scorecard.json", [])
    assert [r["symbol"] for r in rows] == ["XLE", "SMH"]  # SPY itself isn't scored
    assert rows[1]["open"] is False
    s = scorecard.summary(rows)
    assert s["claude"]["avg_return_pct"] == 10.0 and s["claude"]["avg_vs_spy_pct"] == 9.0
    assert s["claude"]["dollars"] == 10.0
    assert s["watch"]["beat_spy"] == 0 and s["watch"]["avg_return_pct"] == -5.0
    assert rows[1]["closed_date"] == "2026-10-14"
