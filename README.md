# Claude ETF and stock trading agent

A fully automated agent that, once per US trading day:

1. Pulls a year of daily prices for ~80 ETFs (US indexes, sectors, themes, countries, bonds,
   commodities, crypto, and a few leveraged/inverse funds) and ~60 big, well-known company stocks,
   and computes trend, momentum, RSI,
   MACD, volatility and **candlestick patterns** for each.
2. Reads the last 36 hours of market headlines from Alpaca's news feed.
3. Gives all of that to **Claude** with **web search**. Claude researches what is moving markets
   (central banks, economic data, tariffs, wars, what world leaders and officials are saying),
   then picks target weights for any ETFs or company stocks it believes will beat the S&P 500.
4. A **risk engine in code** clamps those weights, keeps the rest in SPY, and places the orders on
   **Alpaca** (paper trading by default).
5. Saves every decision, with Claude's reasoning, sources, orders and API cost, to `docs/data/`.
   The dashboard reads those files.

Every 5 minutes while the market is open, a rules-only **watcher** (`trader/watch.py`, GitHub
Actions, no Claude call) sells positions that fall 6% below their high, and scores every ETF on six
buy signals: daily uptrend, a healthy rise today, price above today's VWAP, bullish 15-minute
candles, a volume surge and a new 20-day high. When 5 of 6 agree (and SPY isn't falling, and the
first half hour is over) it buys 8% of the money, up to 4 times a day, funded from SPY.

After the close, a second job records portfolio value against SPY so the dashboard can compare them.

## Read this first

- **Nothing here guarantees beating the S&P 500.** Most professional managers don't, over time.
  This is an experiment that measures whether it can, honestly and against SPY.
- Paper trade for at least 3–6 months before thinking about real money. A few good weeks are noise.
- Claude only *proposes*. It can never place an order the risk engine has not approved.
- "No strong view" means the money sits in SPY, so the agent only bets against the
  index when Claude has a reason to.

## Risk rules (enforced in code, in `trader/risk.py`)

| Rule | Default | Env var |
|---|---|---|
| Money the agent may use (rest of the paper account is ignored) | $1,000 | `CAPITAL_USD` (0 = whole account) |
| Unallocated money goes to | SSO (2x S&P 500, no trend filter) | `IDLE_ASSET` |
| Most that may sit in the idle asset | 50% | `MAX_IDLE_WEIGHT` |
| Watcher: sell a position this far below its high | 6% | `STOP_TRAIL_PCT` |
| Watcher: signals that must agree / buy size / max buys per day | 5 of 6 / 8% / 4 | `WATCH_MIN_SCORE` / `WATCH_BUY_SIZE` / `WATCH_MAX_BUYS_PER_DAY` |
| Max weight in one ETF (other than SPY) | 30% | `MAX_POSITION_WEIGHT` |
| Max number of positions | 12 | `MAX_POSITIONS` |
| Max weight in one big, well-known company | 10% | `MAX_STOCK_WEIGHT` |
| Max weight in one smaller company / all smaller companies | 5% / 10% | `MAX_SMALL_STOCK_WEIGHT` / `MAX_SMALL_STOCKS_TOTAL` |
| Max total in single-company stocks | 40% | `MAX_STOCKS_TOTAL` |
| No buying a stock this close to its earnings report (trading days; unknown date = no buy) | 3 | `EARNINGS_BLACKOUT_DAYS` |
| Watcher: trailing stop for single stocks | 10% | `STOP_TRAIL_PCT_STOCK` |
| Max total in leveraged/inverse ETFs | 10% | `MAX_LEVERAGED_WEIGHT` |
| Max traded per day (one-way) | 50% of equity | `MAX_DAILY_TURNOVER` |
| Ignore weight changes smaller than | 2% of equity | `REBALANCE_BAND` |
| Stop trading if down from peak by | 25% | `MAX_DRAWDOWN_HALT` |
| Kill switch | on | `TRADING_ENABLED=false` |

ETFs and stocks that Claude suggests from outside the built-in lists must be active and tradable on
Alpaca (no over-the-counter shares). ETFs must trade at least $5M per day; smaller companies at least
$20M per day and $5 a share.

## Setup (about 15 minutes)

1. **Alpaca paper account.** Sign up at alpaca.markets. In the dashboard, switch to *Paper Trading*
   and generate an API key and secret.
2. **Claude access**, either of:
   - **Your Claude subscription (default, no extra cost).** On your computer, install Claude Code
     (`npm install -g @anthropic-ai/claude-code`), run `claude setup-token`, and log in. It prints a
     long-lived token. Each daily run uses some of your plan's usage limits.
   - **API credit.** Create a key at console.anthropic.com. You pay per run, and the exact cost
     shows on the dashboard. If an API key is set, it is used instead of the subscription.
3. **GitHub secrets.** In this repo: *Settings → Secrets and variables → Actions → New repository
   secret*. Add `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and either `CLAUDE_CODE_OAUTH_TOKEN` (the
   subscription token) or `ANTHROPIC_API_KEY`.
4. **Merge to `main`.** Scheduled GitHub Actions only run from the default branch.
5. **Dashboard.** *Settings → Pages → Build and deployment → Deploy from a branch*, then choose
   `main` and the `/docs` folder. It will be at `https://<your-user>.github.io/larp-trading/`.
6. **First run.** *Actions → Trading agent → Run workflow*, choosing `trade` with *dry run* ticked.
   Read the decision on the dashboard, then let the schedule take over.

The schedule runs on weekdays at 15:15 UTC (11:15 New York in summer, 10:15 in winter) and takes a
snapshot at 21:30 UTC. GitHub may start scheduled jobs a few minutes late. Market holidays are
skipped automatically.

> This repository is **public**, so the dashboard and trade journal are visible to anyone.
> Your API keys are not, because they live in GitHub secrets. To keep results private, make the repo
> private. GitHub Pages on a private repo needs a paid GitHub plan; otherwise, open `docs/index.html`
> locally as shown below.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in your keys
python -m trader.main trade --dry-run --force   # decide, but send no orders
python -m trader.main snapshot
python -m pytest -q
python -m http.server -d docs 8000   # dashboard at http://localhost:8000
```

## Costs

- Alpaca paper trading and its market data and news: free.
- Claude on your subscription: no extra charge. It uses Sonnet by default. Set the repository
  variable `CLAUDE_MODEL=opus` for deeper research, which uses more of your plan's limits.
- Claude on API credit: one run per day on `claude-opus-5-5` with up to 10 web searches. This is
  typically somewhere around $0.50–$2 per day, and the exact cost of each run shows on the
  dashboard. Use `CLAUDE_EFFORT=medium` or `MAX_WEB_SEARCHES=5` to spend less.
- GitHub Actions and Pages: free for public repos.

On API credit, the dashboard subtracts the Claude bill from the profit, so you see whether the
agent pays for itself.

## About tweets and social media

The X (Twitter) API costs at least $200 a month, so it isn't wired in. Claude's web search still
finds posts by world leaders and officials, and big shifts in online sentiment, once news outlets
report them, which is when most of them move markets anyway. A paid social-data feed could be added
later as one more input in `trader/market.py`.

## Going live with real money

Only after a long paper record that beats SPY *after* AI costs:

1. Create live API keys in Alpaca and replace the two Alpaca secrets.
2. Under *Settings → Secrets and variables → Actions → Variables*, set `ALPACA_PAPER=false` **and**
   `LIVE_TRADING_CONFIRMED=yes`. The code refuses to trade real money unless both are set.
3. Start with an amount you can afford to lose. Short-term gains are taxed as ordinary income.

## Layout

```
trader/
  main.py        daily run: data → Claude → risk engine → orders → journal
  brain.py       Claude prompt, web search, structured decision (subscription or API)
  indicators.py  indicators + candlestick patterns
  risk.py        position limits, rebalancing, turnover cap, drawdown halt
  broker.py      Alpaca account, ETF vetting, order execution
  market.py      Alpaca daily bars and news
  universe.py    the ETF list
docs/index.html  dashboard (reads docs/data/*.json)
.github/workflows/trade.yml   schedule
```

Not financial advice.
