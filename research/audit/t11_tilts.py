"""T11: static ETF tilts vs SPY (hindsight scan). A tilt that beat SPY in a decade is usually a bet on one style, not skill: look at years beating and the bad years."""
import numpy as np, pandas as pd
from trader.lab import Lab, stats as lstats
L = Lab(); spy = L.series("SPY").pct_change().fillna(0)
rows = []
for s in ["QQQ", "XLK", "SMH", "MTUM", "QUAL", "USMV", "IWM", "MDY", "DIA", "XLV", "XLY", "XLF", "XLE", "XLI", "XLP", "XLU", "XLB", "GLD", "TLT", "IEF"]:
    r = L.series(s).pct_change().fillna(0); st = lstats(r, spy)
    by = {y: v for y, v in st["by_year_excess_pct"].items()}
    rows.append({"etf": s, "yearly_%": st["yearly_pct"], "vs_S&P": round(st["yearly_pct"] - st["bench_yearly_pct"], 1), "worst_drop_%": st["worst_drop_pct"], "years_beating": st["years_beating"],
                 "2018": by.get(2018), "2022": by.get(2022), "worst_year_vs_S&P": min(by.values())})
t = pd.DataFrame(rows).sort_values("vs_S&P", ascending=False); t.to_csv("research/audit/out_t11.csv", index=False)
pd.set_option("display.width", 200); print(t.to_string(index=False))
