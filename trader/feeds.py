"""Extra free news sources for the News Hunter, on top of Alpaca's (Benzinga) feed.

Each source is a public RSS/Atom feed. A source that is down or blocked is skipped quietly, so the
News Hunter always runs with whatever it can read.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

log = logging.getLogger(__name__)

# name -> feed URL. Grouped by what they're good for.
FEEDS: dict[str, str] = {
    # Companies announce here first (contracts, approvals, results, deals).
    "PR Newswire": "https://www.prnewswire.com/rss/news-releases-list.rss",
    "Business Wire": "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeGVtRWA==",
    # Official filings: an 8-K is filed within days of any major company event, often before the press.
    "SEC 8-K filings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&count=40&output=atom",
    # Market news.
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Seeking Alpha market news": "https://seekingalpha.com/market_currents.xml",
    # Policy: central bank and the White House.
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "White House": "https://www.whitehouse.gov/news/feed/",
}

PER_SOURCE = 15
USER_AGENT = "Mozilla/5.0 (compatible; larp-trading-news/1.0; paper-trading research)"
# The SEC asks automated readers to say who they are; point it at the project page.
SEC_USER_AGENT = "larp-trading paper-trading research github.com/diegodelossantosg/larp-trading"
_TICKER = re.compile(r"\((?:NASDAQ|NYSE|NYSE American|NYSE Arca|AMEX|Nasdaq|NasdaqGS|NasdaqGM|OTCQX|CBOE)\s*:\s*([A-Z.]{1,6})\)")
_TAG = re.compile(r"<[^>]+>")


def _text(el, *names: str) -> str:
    for n in names:
        found = el.find(n)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return ""


def _when(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        d = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse(xml: bytes, source: str, since: datetime) -> list[dict]:
    """Items newer than `since` from an RSS or Atom document, in the same shape as Market.news()."""
    root = ElementTree.fromstring(xml)
    atom = "{http://www.w3.org/2005/Atom}"
    items = root.findall(".//item") or root.findall(f".//{atom}entry")
    out = []
    for it in items:
        title = _text(it, "title", f"{atom}title")
        when = _when(_text(it, "pubDate", f"{atom}updated", f"{atom}published",
                           "{http://purl.org/dc/elements/1.1/}date"))
        if not title or (when and when < since):
            continue
        summary = _TAG.sub(" ", _text(it, "description", f"{atom}summary", f"{atom}content"))
        summary = re.sub(r"\s+", " ", summary).strip()[:300]
        out.append({
            "time": when.strftime("%Y-%m-%d %H:%M UTC") if when else "",
            "headline": title,
            "summary": summary,
            "symbols": sorted(set(_TICKER.findall(title + " " + summary)))[:6],
            "source": source,
        })
        if len(out) >= PER_SOURCE:
            break
    return out


def fetch_all(hours: float = 2, timeout: float = 8) -> tuple[list[dict], dict[str, str]]:
    """(headlines from every extra source, {source: 'ok (n)' or the error})."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    headlines, status = [], {}
    for name, url in FEEDS.items():
        try:
            agent = SEC_USER_AGENT if "sec.gov" in url else USER_AGENT
            req = urllib.request.Request(url, headers={"User-Agent": agent, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                items = parse(resp.read(), name, since)
            headlines += items
            status[name] = f"ok ({len(items)})"
        except Exception as e:  # a missing source must never stop the News Hunter
            status[name] = f"unavailable: {type(e).__name__}"
            log.info("news source %s unavailable: %s", name, e)
    return headlines, status
