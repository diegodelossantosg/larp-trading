from trader import health

VIX = "DATE,OPEN,HIGH,LOW,CLOSE\n" + "\n".join(f"10/0{i}/2026,15,16,14,{v}" for i, v in enumerate([14, 15, 16, 18, 20, 31], 1))
FRED = {"DGS10": "observation_date,DGS10\n" + "\n".join(f"2026-09-{d:02d},{4.0 + d * 0.03:.2f}" for d in range(1, 25)) + "\n2026-09-25,.",
        "DGS2": "observation_date,DGS2\n2026-09-24,3.90",
        "BAMLH0A0HYM2": "observation_date,BAMLH0A0HYM2\n2026-09-24,3.10"}


def fake_fetch(url):
    if "cboe" in url:
        return VIX
    return FRED[url.split("id=")[1]]


def test_parse_handles_cboe_and_fred_and_holidays():
    assert health.parse_series(VIX)[-1] == ("2026-10-06", 31.0)
    assert health.parse_series(FRED["DGS10"])[-1][0] == "2026-09-24"  # "." row skipped
    assert health.parse_series(None) == []


def test_build_scores_mood():
    tech = {"A": {"vs_sma200_pct": 5, "vs_sma50_pct": 2}, "B": {"vs_sma200_pct": -3, "vs_sma50_pct": -1},
            "C": {"vs_sma200_pct": -1, "vs_sma50_pct": 1}}
    h = health.build(tech, fetch=fake_fetch)
    g = h["gauges"]
    assert g["fear_vix"]["label"] == "panic"  # 31
    assert g["rates"]["curve_10y_minus_2y"] > 0 and "rising fast" in g["rates"]["label"]  # +0.60 in 20 days
    assert g["credit_spread"]["label"] == "relaxed"
    assert g["breadth"]["label"] == "weak"  # 1 of 3 above 200d
    assert h["mood"] == "stressed"


def test_build_survives_no_network():
    h = health.build(None, fetch=lambda url: None)
    assert h["mood"] == "unknown" and h["gauges"] == {}
