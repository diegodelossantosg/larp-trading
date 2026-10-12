"""Settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "data"


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    alpaca_api_key: str
    alpaca_secret_key: str
    paper: bool
    claude_model: str
    claude_effort: str
    idle_asset: str
    max_position_weight: float
    max_positions: int
    max_leveraged_weight: float
    max_daily_turnover: float
    rebalance_band: float
    min_order_usd: float
    max_drawdown_halt: float
    trading_enabled: bool
    max_web_searches: int


def load_config() -> Config:
    _load_dotenv()
    paper = _bool("ALPACA_PAPER", True)
    if not paper and os.environ.get("LIVE_TRADING_CONFIRMED", "no").lower() != "yes":
        raise SystemExit(
            "Refusing to trade real money: ALPACA_PAPER=false but LIVE_TRADING_CONFIRMED is not 'yes'."
        )
    missing = [k for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing settings: {', '.join(missing)} (see .env.example)")
    return Config(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        alpaca_api_key=os.environ.get("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
        paper=paper,
        claude_model=os.environ.get("CLAUDE_MODEL", "claude-opus-5-5"),
        claude_effort=os.environ.get("CLAUDE_EFFORT", "high"),
        idle_asset=os.environ.get("IDLE_ASSET", "SPY").upper(),
        max_position_weight=_float("MAX_POSITION_WEIGHT", 0.30),
        max_positions=int(_float("MAX_POSITIONS", 8)),
        max_leveraged_weight=_float("MAX_LEVERAGED_WEIGHT", 0.10),
        max_daily_turnover=_float("MAX_DAILY_TURNOVER", 0.50),
        rebalance_band=_float("REBALANCE_BAND", 0.02),
        min_order_usd=_float("MIN_ORDER_USD", 25),
        max_drawdown_halt=_float("MAX_DRAWDOWN_HALT", 0.25),
        trading_enabled=_bool("TRADING_ENABLED", True),
        max_web_searches=int(_float("MAX_WEB_SEARCHES", 10)),
    )
