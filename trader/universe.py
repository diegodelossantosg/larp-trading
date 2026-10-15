"""The ETFs and big company stocks the agent scans every day.

Claude may also propose ETFs or stocks outside these lists; those are checked against
Alpaca (active, tradable, enough daily volume) before use. Stocks outside BIG_STOCKS count as
small, speculative bets with tighter size limits.
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

# Big, well-known US companies: up to 10% each. The watcher scans these too.
BIG_STOCKS: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon", "META": "Meta (Facebook)", "AVGO": "Broadcom", "TSLA": "Tesla",
    "ORCL": "Oracle", "AMD": "AMD", "NFLX": "Netflix", "CRM": "Salesforce", "ADBE": "Adobe",
    "CSCO": "Cisco", "IBM": "IBM", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "MU": "Micron", "AMAT": "Applied Materials", "INTC": "Intel", "PLTR": "Palantir",
    "UBER": "Uber", "NOW": "ServiceNow", "TSM": "TSMC", "ASML": "ASML",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "WFC": "Wells Fargo", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "V": "Visa", "MA": "Mastercard", "AXP": "American Express",
    "LLY": "Eli Lilly", "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson", "ABBV": "AbbVie",
    "MRK": "Merck", "PFE": "Pfizer", "TMO": "Thermo Fisher", "ABT": "Abbott", "NVO": "Novo Nordisk",
    "WMT": "Walmart", "COST": "Costco", "HD": "Home Depot", "MCD": "McDonald's", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "PG": "Procter & Gamble", "NKE": "Nike", "SBUX": "Starbucks", "DIS": "Disney",
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "CAT": "Caterpillar", "GE": "GE Aerospace", "BA": "Boeing", "LMT": "Lockheed Martin",
    "RTX": "RTX (Raytheon)", "HON": "Honeywell", "UPS": "UPS", "LIN": "Linde",
    "T": "AT&T", "VZ": "Verizon",
}

NAMES: dict[str, str] = {**CORE_ETFS, **BIG_STOCKS}

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
