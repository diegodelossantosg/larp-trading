from types import SimpleNamespace

from trader import journal, reset


class FakeClient:
    def __init__(self):
        self.closed = False

    def close_all_positions(self, cancel_orders=False):
        self.closed = cancel_orders


class FakeBroker:
    def __init__(self, open_=True, positions=None):
        self.client, self.open_, self._pos = FakeClient(), open_, positions if positions is not None else {"SSO": {}}

    def market_open(self):
        return self.open_

    def positions(self):
        if self.client.closed:
            self._pos = {}
        return self._pos


def test_reset_data_empties_history_and_removes_capital(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    journal.write("capital.json", {"capital": 1000, "baseline_equity": 100000})
    journal.write("decisions.json", [{"date": "2026-11-05"}])
    journal.write("watch.json", {"highs": {"QUAL": 227}})
    reset.reset_data(1000.0, spy=650.25)
    assert not (tmp_path / "capital.json").exists()
    assert journal.read("decisions.json", None) == []
    assert journal.read("equity.json", None) == []
    assert journal.read("watch.json", None)["highs"] == {}
    assert journal.read("sales.json", None)["rows"] == []
    assert journal.read("status.json", None)["equity"] == 1000.0
    assert journal.read("status.json", None)["spy_start"] == 650.25          # S&P 500 starts now, with the same $1,000
    assert journal.read("intraday.json", None)["points"][0][1:] == [1000.0, 650.25]


def test_liquidate_sells_everything():
    b = FakeBroker()
    assert reset.liquidate(b, wait_seconds=5) == 0
    assert b.client.closed


def test_on_open_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    cfg = SimpleNamespace(paper=True)
    journal.write("liquidate.json", {"pending": True})
    assert reset.on_open(FakeBroker(open_=False), cfg) == 0           # closed: wait, stay pending
    assert journal.read("liquidate.json", {})["pending"] is True
    b = FakeBroker()
    assert reset.on_open(b, cfg) == 0 and b.client.closed
    assert journal.read("liquidate.json", {})["pending"] is False
    b2 = FakeBroker()
    reset.on_open(b2, cfg)
    assert not b2.client.closed                                       # already done: never again


def test_on_open_refuses_live_money(tmp_path, monkeypatch):
    monkeypatch.setattr(journal, "DATA_DIR", tmp_path)
    journal.write("liquidate.json", {"pending": True})
    b = FakeBroker()
    assert reset.on_open(b, SimpleNamespace(paper=False)) == 1 and not b.client.closed
