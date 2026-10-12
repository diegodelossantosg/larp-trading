"""The ETFs the agent scans every day.

Claude may also propose ETFs outside this list; those are checked against
Alpaca (active, tradable, looks like a fund, enough daily volume) before use.
"""

from __future__ import annotations

import re

# symbol -> short description shown to Claude
CORE_ETFS: dict[str, str] = {
    # Broad US
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000 small caps",
    "RSP": "S&P 500 equal weight", "MDY": "S&P 400 mid caps",
    # Style / factor
    "VUG": "US large growth", "VTV": "US large value", "MTUM": "US momentum", "QUAL": "US quality",
    "USMV": "US min volatility", "SCHD": "US dividend",
    # Sectors
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health care",
    "XLI": "Industrials", "XLY": "Consumer discretionary", "XLP": "Consumer staples",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real estate", "XLC": "Communication services",
    # Industries / themes
    "SMH": "Semiconductors", "IGV": "Software", "XBI": "Biotech", "ITA": "Aerospace & defense",
    "KRE": "Regional banks", "XHB": "Homebuilders", "XRT": "Retail", "JETS": "Airlines",
    "TAN": "Solar", "ICLN": "Clean energy", "URA": "Uranium", "LIT": "Lithium & batteries",
    "CIBR": "Cybersecurity", "BOTZ": "Robotics & AI", "ARKK": "Disruptive innovation",
    "XOP": "Oil & gas exploration", "GDX": "Gold miners", "COPX": "Copper miners",
    # International
    "EFA": "Developed ex-US", "EEM": "Emerging markets", "VGK": "Europe", "EWJ": "Japan",
    "FXI": "China large caps", "KWEB": "China internet", "INDA": "India", "EWZ": "Brazil",
    "EWW": "Mexico", "EWT": "Taiwan", "EWY": "South Korea", "EWG": "Germany", "EWU": "UK",
    # Bonds
    "SHY": "1-3y Treasuries", "IEF": "7-10y Treasuries", "TLT": "20y+ Treasuries",
    "TIP": "TIPS (inflation-protected)", "LQD": "Investment-grade corporates", "HYG": "High-yield bonds",
    "BIL": "1-3 month T-bills (cash-like)",
    # Commodities / currencies / crypto
    "GLD": "Gold", "SLV": "Silver", "USO": "Crude oil", "UNG": "Natural gas", "DBA": "Agriculture",
    "DBC": "Broad commodities", "UUP": "US dollar", "FXE": "Euro", "FXY": "Japanese yen",
    "IBIT": "Spot bitcoin", "ETHA": "Spot ether",
    # Leveraged / inverse: daily-reset products, capped by MAX_LEVERAGED_WEIGHT
    "SH": "Inverse S&P 500 (-1x)", "PSQ": "Inverse Nasdaq 100 (-1x)",
    "SSO": "2x S&P 500", "QLD": "2x Nasdaq 100", "TQQQ": "3x Nasdaq 100", "SQQQ": "-3x Nasdaq 100",
    "SOXL": "3x Semiconductors", "TMF": "3x 20y+ Treasuries",
}

LEVERAGED = {"SH", "PSQ", "SSO", "QLD", "TQQQ", "SQQQ", "SOXL", "TMF"}

_LEVERAGED_NAME = re.compile(
    r"\b(2x|3x|-1x|-2x|-3x|ultra|ultrapro|leveraged|inverse|short|bear|bull 2x|bull 3x|daily)\b",
    re.IGNORECASE,
)
_FUND_NAME = re.compile(r"\b(etf|fund|trust|ishares|spdr|shares)\b", re.IGNORECASE)


def is_leveraged(symbol: str, name: str = "") -> bool:
    return symbol in LEVERAGED or bool(_LEVERAGED_NAME.search(name))


def looks_like_fund(name: str) -> bool:
    return bool(_FUND_NAME.search(name))
