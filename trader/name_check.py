"""Does a headline really talk about the company its news feed tagged it with?

Alpaca's news provider (since 2026) tags many foreign companies with a US ticker that belongs to someone else:
"Tesco" as TSCO (Tractor Supply), "Anglo American" as AAL (American Airlines), "Reckitt" as RKT (Rocket),
"Vinci" as DG (Dollar General)... In the hour-by-hour replay (research/replay) a rule that trusted those tags
bought stocks whose drop had nothing to do with the headline, and lost about 4% per buy. So a symbol only counts
for a headline that names the company (or writes its ticker).
"""

from __future__ import annotations

import re

# words that don't identify a company on their own ("American", "First", "Dollar", "Supply"...)
GENERIC = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "holdings", "holding", "group", "plc",
    "ltd", "limited", "the", "class", "common", "stock", "shares", "ordinary", "sa", "nv", "ag", "se", "llc", "lp",
    "adr", "ads", "de", "and", "of", "new", "american", "america", "united", "first", "general", "national",
    "international", "global", "energy", "financial", "capital", "health", "healthcare", "bank", "bancorp", "data",
    "digital", "systems", "solutions", "technologies", "technology", "tech", "pharmaceuticals", "pharmaceutical",
    "therapeutics", "resources", "services", "industries", "brands", "dollar", "supply", "trust", "realty", "partners",
    "entertainment", "communications", "networks", "software", "semiconductor", "devices", "motors", "airlines",
    "foods", "power", "electric", "materials", "products", "enterprises", "investment", "investments", "properties",
    "one", "us", "usa", "north", "south", "west", "east", "street", "media", "labs", "group,", "sciences",
}
# names headlines use instead of the legal one
ALIASES = {
    "GOOG": ("google", "alphabet", "youtube", "waymo"), "GOOGL": ("google", "alphabet", "youtube", "waymo"),
    "META": ("meta", "facebook", "instagram", "whatsapp"), "BRK.B": ("berkshire",), "LLY": ("lilly",),
    "JPM": ("jpmorgan", "jp morgan"), "BAC": ("bank of america", "bofa"), "WFC": ("wells fargo",),
    "GS": ("goldman",), "MS": ("morgan stanley",), "C": ("citigroup", "citi"), "UNH": ("unitedhealth",),
    "DIS": ("disney",), "KO": ("coca-cola", "coca cola", "coke"), "PEP": ("pepsico", "pepsi"),
    "HD": ("home depot",), "MCD": ("mcdonald",), "AAL": ("american airlines",), "UAL": ("united airlines",),
    "DAL": ("delta",), "LUV": ("southwest",), "GE": ("ge aerospace", "general electric"), "GM": ("general motors",),
    "F": ("ford",), "T": ("at&t",), "TSM": ("tsmc", "taiwan semiconductor"), "BABA": ("alibaba",),
    "PDD": ("temu", "pinduoduo"), "AMZN": ("amazon", "aws"), "MSFT": ("microsoft",), "AAPL": ("apple", "iphone"),
    "DG": ("dollar general",), "DLTR": ("dollar tree",), "TSCO": ("tractor supply",), "RKT": ("rocket",),
}
# tickers that are also ordinary words: their mere appearance proves nothing
WORDY = {"A", "ALL", "AN", "ARE", "BE", "CAN", "FOR", "GO", "HAS", "IT", "KEY", "LOW", "NOW", "ON", "ONE", "OR",
         "OUT", "SO", "SEE", "TRUE", "TWO", "UP", "WELL", "AI", "ANY", "BIG", "CASH", "EAT", "FAST", "FUN", "HE",
         "LIVE", "MAN", "MAIN", "NEXT", "OPEN", "PLAY", "REAL", "RUN", "SAFE", "SAVE", "SHOP", "TEAM", "U", "WAY"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9&]+(?:-[a-z0-9]+)*", text.lower())


def mentions_company(headline: str, symbol: str, name: str | None) -> bool:
    """True if the headline names the company (or writes its ticker); True when the name is unknown."""
    if not headline:
        return False
    sym = symbol.upper()
    if sym not in WORDY and len(sym) >= 2 and re.search(rf"(?<![A-Za-z]){re.escape(sym)}(?![A-Za-z])", headline):
        return True
    low = headline.lower()
    for alias in ALIASES.get(sym, ()):
        if re.search(rf"(?<![a-z]){re.escape(alias)}", low):
            return True
    if not name:
        return True
    tokens = [t for t in _words(name.split(" Class ")[0]) if t]
    distinctive = [t for t in tokens if t not in GENERIC and len(t) >= 3]
    words = set(_words(headline))
    joined = " ".join(_words(headline))
    if distinctive:
        # 'apple' matches 'Apple' and "Apple's"; a name glued together in the headline ("ServiceNow") too
        return any(t in words or f"{t}s" in words or re.search(rf"\b{re.escape(t)}", joined) for t in distinctive)
    core = [t for t in tokens if t not in ("inc", "incorporated", "corp", "corporation", "co", "company", "plc",
                                           "ltd", "limited", "the", "holdings", "group")]
    return bool(core) and " ".join(core) in joined


def filter_headlines(headlines: list[dict], names: dict[str, str | None]) -> tuple[list[dict], list[str]]:
    """Keep, for each headline, only the symbols it really names; drop headlines left with none of their symbols
    (headlines that had no symbols at all are kept: they are market or policy news). Returns (kept, dropped notes)."""
    kept, dropped = [], []
    for h in headlines:
        syms = h.get("symbols") or []
        if not syms:
            kept.append(h)
            continue
        good = [s for s in syms if mentions_company(h.get("headline", ""), s, names.get(s))]
        bad = [s for s in syms if s not in good]
        if bad:
            dropped.append(f"{','.join(bad)}: {h.get('headline', '')[:80]}")
        if good:
            kept.append({**h, "symbols": good})
    return kept, dropped
