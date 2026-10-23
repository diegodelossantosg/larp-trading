"""Other large, heavily traded US companies the News Hunter also watches (besides BIG_STOCKS).

Every company that traded at least $300M a day on average over the 20 sessions to 2026-10-22
(from research/lab.npz; funds and leveraged products left out). Why $300M: in the 40-month news test,
buying a drop on news beat random days for companies above that level (in the big, design and
held-out groups alike) and did not below it. They count as "big" companies for the risk limits;
the News Hunter re-checks the $300M at buy time, so a name that has gone quiet is skipped.
Refresh by re-running the lab data download and regenerating this list.
"""

LIQUID_MIN_DOLLAR_VOLUME_M = 300.0

LIQUID_STOCKS: frozenset[str] = frozenset({
    "A", "AAL", "ABNB", "ACN", "ADI", "ADP", "ADSK", "AEM", "AEP", "AIG", "AJG", "AKAM", "ALAB",
    "ALL", "AME", "AMGN", "AMT", "ANET", "AON", "APH", "APLD", "APO", "APP", "ASTS", "AXON", "AZN",
    "AZO", "BE", "BKNG", "BKR", "BLK", "BMNR", "BMY", "BNY", "BP", "BSX", "BURL", "BX", "C", "CAH",
    "CB", "CBOE", "CCI", "CCL", "CDE", "CDNS", "CEG", "CFG", "CHTR", "CI", "CIEN", "CIFR", "CL",
    "CLS", "CMCSA", "CME", "CMG", "CMI", "COF", "COHR", "COIN", "COR", "CRCL", "CRH", "CRL", "CRWD",
    "CRWV", "CSX", "CTAS", "CTSH", "CTVA", "CVNA", "CVS", "DAL", "DASH", "DDOG", "DE", "DELL",
    "DHI", "DHR", "DKS", "DT", "DUK", "DVN", "EBAY", "ECHO", "ECL", "EFX", "ELV", "EMR", "ENTG",
    "EOG", "EQIX", "EQT", "EXC", "EXPE", "F", "FANG", "FAST", "FCX", "FDX", "FERG", "FICO", "FITB",
    "FIX", "FN", "FSLR", "FTNT", "GD", "GEV", "GILD", "GIS", "GLW", "GM", "GOOG", "GWW", "HAL",
    "HBAN", "HCA", "HL", "HLT", "HOOD", "HPE", "HPQ", "HUM", "HUT", "HWM", "IBKR", "ICE", "ILMN",
    "INTU", "IONQ", "IQV", "IR", "ISRG", "ITW", "JBL", "JCI", "KDP", "KEY", "KEYS", "KHC", "KKR",
    "KLAC", "KMB", "KMI", "KR", "KVUE", "LEN", "LHX", "LITE", "LNG", "LOW", "LRCX", "LULU", "MAR",
    "MARA", "MCHP", "MCK", "MCO", "MDB", "MDLZ", "MDT", "MELI", "MKSI", "MLM", "MMM", "MNST", "MO",
    "MPC", "MPWR", "MRNA", "MRSH", "MRVL", "MSCI", "MSI", "MSTR", "MTD", "MTSI", "NEE", "NEM",
    "NET", "NOC", "NOK", "NRG", "NTAP", "NTRA", "NU", "NXPI", "O", "ODFL", "OKLO", "OKTA", "ON",
    "ONDS", "ONON", "ORLY", "OTIS", "OXY", "P", "PANW", "PATH", "PAYX", "PBR", "PCAR", "PCG", "PGR",
    "PH", "PLD", "PM", "PNC", "PSX", "PTC", "PWR", "PYPL", "RBLX", "RBRK", "RCL", "RDDT", "REGN",
    "RF", "RIVN", "RKLB", "RKT", "ROK", "ROST", "RVMD", "SAP", "SCHW", "SHW", "SLB", "SMCI", "SNDK",
    "SNOW", "SNPS", "SO", "SOFI", "SPGI", "SPOT", "STM", "STT", "SWKS", "SYK", "TDG", "TEAM", "TEL",
    "TEM", "TER", "TFC", "TGT", "TJX", "TLN", "TMUS", "TRGP", "TRV", "TSCO", "TT", "TTMI", "TTWO",
    "TWLO", "TWST", "TXG", "U", "UAL", "UNP", "URI", "USB", "UTHR", "VALE", "VEEV", "VLO", "VMC",
    "VRT", "VRTX", "VST", "WAT", "WCN", "WDAY", "WDC", "WELL", "WM", "WMB", "WULF", "XEL", "XYZ",
    "YUM", "ZS",
})
