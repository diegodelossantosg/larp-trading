"""Which kinds of news drops carry the News Hunter's edge? Each slice is judged on dev AND holdout separately (never big: it is biased up)."""
import numpy as np, pandas as pd
ev = pd.read_csv("research/events.csv.gz", low_memory=False)
news = ev[(ev.kind == "news") & ev["set"].isin(["dev", "holdout"])].copy()
ctl = ev[(ev.kind == "control") & ev["set"].isin(["dev", "holdout"])]
H = 10
col = f"hour_after_r{H}"
base = {s: (ctl[ctl["set"] == s][f"h11_r{H}"] - 0.001).mean() for s in ("dev", "holdout")}
d = news[(news.move_1h <= -0.02) & (news.dollar_vol_m >= 300)].dropna(subset=[col]).copy()
d["net"] = d[col] - 0.001
topics = [c for c in d.columns if c.startswith("topic_")]
slices = {"ALL (current rule)": d.net.notna()}
for t in topics: slices[f"topic {t[6:]}"] = d[t] == 1
slices["no topic tag"] = d[topics].sum(axis=1) == 0
slices["market above 200d avg"] = d.spy_above_sma200 == 1
slices["market below 200d avg"] = d.spy_above_sma200 == 0
slices["stock above 200d avg"] = d.above_sma200 == 1
slices["stock below 200d avg"] = d.above_sma200 == 0
slices["stock above 50d avg"] = d.above_sma50 == 1
slices["stock below 50d avg"] = d.above_sma50 == 0
for lo, hi in ((.02, .03), (.03, .05), (.05, .08), (.08, 1)):
    slices[f"drop {int(lo*100)}-{int(hi*100) if hi<1 else '+'}%"] = (d.move_1h <= -lo) & (d.move_1h > -hi)
slices["volume surge (>2x normal)"] = d.vol_ratio > 2
slices["normal volume (<=2x)"] = d.vol_ratio <= 2
slices["falling for 20d before (<-5%)"] = d.ret20_before < -0.05
slices["rising for 20d before (>+5%)"] = d.ret20_before > 0.05
slices["calm stock (atr<2.5%)"] = d.atr_pct < 2.5
slices["wild stock (atr>=4%)"] = d.atr_pct >= 4
rows = []
for name, m in slices.items():
    row = {"slice": name}
    for s in ("dev", "holdout"):
        x = d[m & (d["set"] == s)]
        n = len(x)
        row[f"n_{s}"] = n
        row[f"edge_{s}"] = round((x.net.mean() - base[s]) * 100, 2) if n >= 15 else np.nan
        row[f"t_{s}"] = round((x.net.mean() - base[s]) / (x.net.std() / n ** .5), 1) if n >= 15 and x.net.std() > 0 else np.nan
    rows.append(row)
out = pd.DataFrame(rows)
out["agree"] = np.sign(out.edge_dev) == np.sign(out.edge_holdout)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
print(out.to_string(index=False))
out.to_csv("research/audit/out_slices_news.csv", index=False)
