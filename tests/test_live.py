from trader import journal, live


def test_live_has_status_and_latest_points(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    journal.write("status.json", {"equity": 1001.0})
    journal.write("intraday.json", {"points": [[f"t{i}", 1000 + i, 700] for i in range(40)]})
    out = live.build()
    assert out["status"]["equity"] == 1001.0
    assert len(out["intraday_tail"]) == 15 and out["intraday_tail"][-1][0] == "t39"


def test_repair_history_removes_a_phantom_step():
    from trader.main import _repair_history

    hist = [(1, 100_000.0), (2, 100_000.0), (3, 200_000.0), (4, 199_998.0)]
    assert _repair_history(hist, 99_998.0) == [(1, 100_000.0), (2, 100_000.0), (3, 100_000.0), (4, 99_998.0)]
    ok = [(1, 100_000.0), (2, 100_050.0)]
    assert _repair_history(ok, 100_040.0) == ok  # small differences are left alone
