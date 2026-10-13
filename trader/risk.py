"""Turn Claude's target weights into a safe set of orders.

Claude proposes; this module disposes. Every rule here is enforced in code,
whatever the model says.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    notional: float  # dollars
    reason: str = ""


@dataclass
class Plan:
    targets: dict[str, float]  # final weights after limits, sums to <= 1
    orders: list[Order] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def apply_limits(
    proposed: dict[str, float],
    *,
    allowed: set[str],
    leveraged: set[str],
    idle_asset: str,
    max_position_weight: float,
    max_positions: int,
    max_leveraged_weight: float,
    max_idle_weight: float = 1.0,
) -> tuple[dict[str, float], list[str]]:
    """Clamp proposed weights to the risk rules; the remainder goes to the idle asset.

    The idle asset (SPY) may hold at most max_idle_weight. Anything above that is spread over
    the other chosen ETFs (up to their own caps); what still doesn't fit stays in cash.
    """
    notes: list[str] = []
    weights: dict[str, float] = {}
    for sym, w in proposed.items():
        sym = sym.upper()
        if sym not in allowed:
            notes.append(f"dropped {sym}: not an approved, tradable ETF")
            continue
        if w <= 0:
            continue
        cap = 1.0 if sym == idle_asset else max_position_weight
        if w > cap:
            notes.append(f"{sym} capped {w:.0%} -> {cap:.0%}")
            w = cap
        weights[sym] = weights.get(sym, 0.0) + w

    active = sorted((s for s in weights if s != idle_asset), key=lambda s: -weights[s])
    for sym in active[max_positions:]:
        notes.append(f"dropped {sym}: more than {max_positions} positions")
        del weights[sym]

    lev_total = sum(w for s, w in weights.items() if s in leveraged)
    if lev_total > max_leveraged_weight:
        scale = max_leveraged_weight / lev_total
        for s in weights:
            if s in leveraged:
                weights[s] *= scale
        notes.append(f"leveraged/inverse ETFs scaled {lev_total:.0%} -> {max_leveraged_weight:.0%}")

    total = sum(weights.values())
    if total > 1.0:
        weights = {s: w / total for s, w in weights.items()}
        notes.append(f"weights summed to {total:.0%}; scaled to 100%")
        total = 1.0

    if total < 1.0:
        weights[idle_asset] = weights.get(idle_asset, 0.0) + (1.0 - total)

    excess = weights.get(idle_asset, 0.0) - max_idle_weight
    if excess > 1e-9:
        weights[idle_asset] = max_idle_weight
        notes.append(f"{idle_asset} capped at {max_idle_weight:.0%}")
        # Water-fill the excess into the other (non-leveraged) picks, pro rata, up to their caps.
        for _ in range(10):
            room = {s: max_position_weight - w for s, w in weights.items()
                    if s != idle_asset and s not in leveraged and max_position_weight - w > 1e-9}
            base = sum(weights[s] for s in room)
            if excess <= 1e-9 or not room or base <= 0:
                break
            given = 0.0
            for s in room:
                add = min(room[s], excess * weights[s] / base)
                weights[s] += add
                given += add
            excess -= given
        if excess > 1e-4:
            notes.append(f"{excess:.0%} kept in cash: not enough other ETFs to keep "
                         f"{idle_asset} at or below {max_idle_weight:.0%}")
    return {s: round(w, 4) for s, w in weights.items() if w > 0.0001}, notes


def plan_orders(
    targets: dict[str, float],
    current_values: dict[str, float],
    equity: float,
    *,
    rebalance_band: float,
    min_order_usd: float,
    max_daily_turnover: float,
    cash_buffer: float = 0.01,
    protected: set[str] | frozenset = frozenset(),
) -> tuple[list[Order], list[str]]:
    """Dollar orders to move current holdings toward targets. Sells come first.

    Positions in `protected` (recent watcher buys) are never trimmed here, so a buy isn't undone
    the next morning; their trailing stop still applies. Buys never spend more than the agent's
    cash plus what it sells.
    """
    notes: list[str] = []
    investable = equity * (1 - cash_buffer)
    deltas: dict[str, float] = {}
    for sym in set(targets) | set(current_values):
        target_usd = targets.get(sym, 0.0) * investable
        delta = target_usd - current_values.get(sym, 0.0)
        exiting = sym not in targets and current_values.get(sym, 0.0) > 0
        if not exiting and abs(delta) < rebalance_band * equity:
            continue
        if abs(delta) < min_order_usd:
            continue
        if delta < 0 and sym in protected:
            notes.append(f"kept {sym}: bought by the watcher recently (minimum holding period)")
            continue
        deltas[sym] = delta

    turnover = sum(abs(d) for d in deltas.values()) / 2  # one-way
    limit = max_daily_turnover * equity
    if turnover > limit:
        scale = limit / turnover
        deltas = {s: d * scale for s, d in deltas.items()}
        notes.append(f"turnover {turnover:,.0f} USD > limit {limit:,.0f} USD; orders scaled by {scale:.2f}")

    cash = investable - sum(current_values.values())
    spend = sum(d for d in deltas.values() if d > 0)
    budget = max(0.0, cash + sum(-d for d in deltas.values() if d < 0))
    if spend > budget + 0.01:
        scale = budget / spend
        deltas = {s: (d * scale if d > 0 else d) for s, d in deltas.items()}
        notes.append(f"buys scaled by {scale:.2f} to stay within the agent's own money")

    sells = [Order(s, "sell", round(-d, 2)) for s, d in deltas.items() if d < 0]
    buys = [Order(s, "buy", round(d, 2)) for s, d in deltas.items() if d > 0]
    sells.sort(key=lambda o: -o.notional)
    buys.sort(key=lambda o: -o.notional)
    return sells + buys, notes


def drawdown(equity_history: list[float]) -> float:
    """Current drawdown from the running peak, as a positive fraction."""
    if not equity_history:
        return 0.0
    peak = max(equity_history)
    return 0.0 if peak <= 0 else max(0.0, 1 - equity_history[-1] / peak)
