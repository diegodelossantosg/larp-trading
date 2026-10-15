from datetime import datetime, timezone

from trader.feeds import parse

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Acme Corp (NASDAQ: ACME) Wins $2 Billion Defense Contract</title>
<description>&lt;p&gt;Acme Corp (NASDAQ: ACME) today announced...&lt;/p&gt;</description>
<pubDate>Thu, 15 Oct 2026 14:30:00 GMT</pubDate></item>
<item><title>Old news</title><pubDate>Mon, 12 Oct 2026 14:30:00 GMT</pubDate></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>8-K - ACME CORP (0000123456) (Filer)</title><updated>2026-10-15T10:31:02-04:00</updated>
<summary type="html">Item 1.01: Entry into a Material Definitive Agreement</summary></entry></feed>"""


def test_rss_items_are_recent_and_tickers_found():
    out = parse(RSS, "PR Newswire", datetime(2026, 10, 15, 13, 0, tzinfo=timezone.utc))
    assert len(out) == 1
    assert out[0]["symbols"] == ["ACME"] and out[0]["source"] == "PR Newswire"
    assert "<p>" not in out[0]["summary"]


def test_atom_filings_are_read():
    out = parse(ATOM, "SEC 8-K filings", datetime(2026, 10, 15, 13, 0, tzinfo=timezone.utc))
    assert out[0]["headline"].startswith("8-K - ACME CORP") and "Material Definitive" in out[0]["summary"]
