from trader import journal, live


def test_live_has_status_and_latest_points(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    journal.write("status.json", {"equity": 1001.0})
    journal.write("intraday.json", {"points": [[f"t{i}", 1000 + i, 700] for i in range(40)]})
    out = live.build()
    assert out["status"]["equity"] == 1001.0
    assert len(out["intraday_tail"]) == 15 and out["intraday_tail"][-1][0] == "t39"
