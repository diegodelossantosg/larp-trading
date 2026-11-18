from trader.name_check import filter_headlines, mentions_company


def test_wrong_company_tags_are_caught():
    assert not mentions_company("Tesco PLC Announces Total Voting Rights and Capital", "TSCO", "Tractor Supply Company")
    assert not mentions_company("JPMorgan Upgrades Anglo American to Neutral", "AAL", "American Airlines Group Inc.")
    assert not mentions_company("Kinshuk Kunwar of Reckitt Philippines Serves as Jury Chair", "RKT", "Rocket Companies, Inc. Class A Common Stock")
    assert not mentions_company("Vinci Appointed as Strategic Development Partner for Derby Market Place", "DG", "Dollar General Corporation")


def test_real_company_news_passes():
    assert mentions_company("First Solar slides after Piper Sandler cuts target", "FSLR", "First Solar, Inc. Common Stock")
    assert mentions_company("Evercore ISI Adjusts Price Target on ServiceNow to $160", "NOW", "ServiceNow, Inc. Common Stock")
    assert mentions_company("Disney's parks unit beats estimates", "DIS", "The Walt Disney Company")
    assert mentions_company("Google faces new EU probe", "GOOGL", "Alphabet Inc. Class A Common Stock")
    assert mentions_company("Why AMD shares are falling today", "AMD", "Advanced Micro Devices, Inc.")
    assert mentions_company("American Airlines cuts its outlook", "AAL", "American Airlines Group Inc.")
    assert mentions_company("Dollar General recalls products", "DG", "Dollar General Corporation")
    assert mentions_company("Anything at all", "XYZ", None)            # unknown name: don't block


def test_wordy_tickers_need_the_name():
    assert not mentions_company("It is NOW time to buy chips", "NOW", "ServiceNow, Inc.")


def test_filter_headlines_keeps_market_news_and_drops_wrong_tags():
    hs = [{"headline": "Fed holds rates", "symbols": []},
          {"headline": "Tesco PLC Announces Total Voting Rights", "symbols": ["TSCO"]},
          {"headline": "Apple and Tesco sign a deal", "symbols": ["AAPL", "TSCO"]}]
    kept, dropped = filter_headlines(hs, {"TSCO": "Tractor Supply Company", "AAPL": "Apple Inc."})
    assert [h["headline"] for h in kept] == ["Fed holds rates", "Apple and Tesco sign a deal"]
    assert kept[1]["symbols"] == ["AAPL"]
    assert len(dropped) == 2
