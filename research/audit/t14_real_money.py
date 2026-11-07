"""T14: what is left of the News Hunter's edge with REAL money in Spain? Arithmetic on the measured pre-tax edge (paper trading has no taxes).
Structure A: money in SPY, news buys funded by selling SPY (Spanish FIFO: oldest lots sold first -> SPY gains get realised and taxed every year).
Structure B: 75% SPY never sold + 25% pot (cash-like when idle, e.g. short bonds) funding the news buys (SPY gains stay deferred, the pot lags SPY)."""
import pandas as pd
rows = []
for market, spy in (("2023-26 (SPY +19.3%/yr)", 19.3), ("2017-26 (SPY +15.1%/yr)", 15.1), ("a weak market (SPY +7%/yr)", 7.0)):
    for alpha in (3.0, 6.0, 9.0):          # pre-tax edge vs SPY, % of equity per year (cautious / middle / optimistic)
        tax = 0.21                          # Spanish savings tax: 19% to 28% depending on the amount; 21% as a middle value
        # A: the news gains are taxed; and churning the SPY core realises its gains (assume half to all of the SPY return becomes taxable each year)
        for churn, label in ((0.5, "A: sell SPY to fund (half the SPY gain realised yearly)"), (1.0, "A: sell SPY to fund (all of the SPY gain realised yearly)")):
            net = alpha * (1 - tax) - tax * spy * churn
            rows.append({"market": market, "pre_tax_edge": alpha, "structure": label, "after_tax_edge_vs_untaxed_SPY": round(net, 2)})
        # B: the pot (25%) idles in short bonds at ~3.5%, about 85% of the time -> lags SPY; news gains and bond interest taxed
        idle_share = 0.25 * 0.85
        lag = idle_share * (spy - 3.5)
        net = alpha * (1 - tax) - lag - tax * idle_share * 3.5
        rows.append({"market": market, "pre_tax_edge": alpha, "structure": "B: 75% SPY untouched + 25% cash pot", "after_tax_edge_vs_untaxed_SPY": round(net, 2)})
d = pd.DataFrame(rows); d.to_csv("research/audit/out_t14.csv", index=False)
pd.set_option("display.width", 200)
print(d.pivot_table(index=["market", "pre_tax_edge"], columns="structure", values="after_tax_edge_vs_untaxed_SPY", sort=False).round(1).to_string())
print("\n(SPY itself is compared without any tax: buy-and-hold defers it. Euro/dollar conversion and the 15% US dividend withholding apply to SPY too, so they cancel out in the comparison.)")
