"""T10: more news families, protocol dev -> holdout: headline keywords, time of day, positive news, earnings drops. Edge vs random days of the same set."""
import re, numpy as np, pandas as pd
ev = pd.read_csv("research/events.csv.gz", low_memory=False)
ev = ev[ev["set"].isin(["dev", "holdout"])]
ctl = ev[ev.kind == "control"]
base = {(s, h): (ctl[ctl["set"] == s][f"h11_r{h}"] - 0.001).mean() for s in ("dev", "holdout") for h in (5, 10)}
news = ev[ev.kind == "news"].copy(); earn = ev[ev.kind == "earnings"].copy()
news["hr"] = pd.to_datetime(news.news_time).dt.hour + pd.to_datetime(news.news_time).dt.minute / 60
def edge(d, s, h=10, entry="hour_after"):
    x = d[d["set"] == s][f"{entry}_r{h}"].dropna() - 0.001
    n = len(x)
    if n < 25: return np.nan, n, np.nan
    return round((x.mean() - base[(s, h)]) * 100, 2), n, round((x.mean() - base[(s, h)]) / (x.std() / n ** .5), 1)
KW = {"downgrade": r"downgrad", "upgrade": r"upgrad", "lawsuit/probe": r"lawsuit|sues|probe|investigat|subpoena|settle", "FDA/trial": r"\bFDA\b|trial|approval|phase",
      "guidance cut": r"cuts? (guidance|outlook|forecast)|lower(s|ed)? (guidance|outlook)|warns", "recall/safety": r"recall|crash|accident|outage|breach|hack",
      "CEO/exec": r"\bCEO\b|\bCFO\b|resign|steps down|appoint", "deal/merger": r"acquir|acquisition|merger|buyout|takeover|to buy", "contract": r"contract|award|wins",
      "short report": r"short (seller|report)|hindenburg|citron", "price target": r"price target|PT ", "offering": r"offering|dilut|convertible|share sale", "tariff/policy": r"tariff|sanction|ban|regulat|fine[sd]?\b"}
fams = {}
drop = news[(news.move_1h <= -0.02) & (news.dollar_vol_m >= 300)]
fams["ALL drops (current rule)"] = drop
for k, p in KW.items(): fams[f"headline: {k}"] = drop[drop.headline.fillna("").str.contains(p, case=False, regex=True)]
fams["headline: none of the keywords"] = drop[~drop.headline.fillna("").str.contains("|".join(KW.values()), case=False, regex=True)]
for lo, hi, nm in ((4, 9.5, "pre-market (4:00-9:30)"), (9.5, 11, "first 90 min"), (11, 14, "midday"), (14, 15, "14:00-15:00")):
    fams[f"time of day: {nm}"] = drop[(drop.hr >= lo) & (drop.hr < hi)]
up = news[(news.move_1h >= 0.02) & (news.dollar_vol_m >= 300)]
fams["POSITIVE news (+2% or more)"] = up
fams["positive news, volatile (ATR>=3%)"] = up[up.atr_pct >= 3]
fams["positive news with volume surge (>2x)"] = up[up.vol_ratio > 2]
eq = earn[(earn.dollar_vol_m >= 300)]
fams["EARNINGS day move <= -5%"] = eq[eq.day_move <= -0.05]; fams["EARNINGS miss tag"] = eq[eq.miss == 1]; fams["EARNINGS beat tag"] = eq[eq.beat == 1]
fams["EARNINGS drop, volatile"] = eq[(eq.day_move <= -0.03) & (eq.atr_pct >= 3)]
rows = []
for name, d in fams.items():
    entry = "close" if name.startswith("EARNINGS") else "hour_after"
    ed, nd, td = edge(d, "dev", 10, entry); eh, nh, th = edge(d, "holdout", 10, entry)
    rows.append({"family": name, "n_dev": nd, "edge_dev": ed, "t_dev": td, "n_holdout": nh, "edge_holdout": eh, "t_holdout": th})
t = pd.DataFrame(rows); t["both_positive"] = (t.edge_dev > 0) & (t.edge_holdout > 0)
t.to_csv("research/audit/out_t10.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 80)
print(t.to_string(index=False))
ok = t.dropna(subset=["edge_dev"]).sort_values("edge_dev", ascending=False).head(6)
print("\nthe 6 best on dev alone -> holdout edge:", ok[["family", "edge_dev", "edge_holdout"]].to_string(index=False))
