"""APEX-3 Macro News and Economic Calendar Data Models.

Immutable, thread-safe data models for tracking global macroeconomic catalysts,
US/India economic releases, breaking news wire sentiment, and blackout windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional


@dataclass(frozen=True)
class MacroEvent:
    """Scheduled macroeconomic event from economic calendar feeds."""
    event_id: str
    timestamp_utc: datetime
    currency: str  # "USD", "INR", "CNY", "EUR"
    impact: Literal["HIGH", "MEDIUM", "LOW"]
    title: str  # e.g., "US Non-Farm Payrolls", "US CPI m/m", "FOMC Interest Rate Decision"
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    inverse_for_usd: bool = False  # True for Unemployment Rate & Jobless Claims where higher actual = weaker USD = bullish Gold/Silver


@dataclass(frozen=True)
class NewsHeadline:
    """Scraped or streamed financial news headline."""
    headline_id: str  # SHA-256 of normalized title
    published_utc: datetime
    source: str
    title: str
    bullion_sentiment: float  # -1.0 (strongly bearish Gold/Silver) to +1.0 (strongly bullish Gold/Silver)
    is_breaking_shock: bool = False


@dataclass(frozen=True)
class MacroRegimeState:
    """Aggregated macroeconomic and news regime snapshot for StrategyEngine."""
    in_blackout_window: bool
    minutes_to_next_high_event: Optional[float]
    active_event_title: str
    macro_score: float  # -1.0 to +1.0
    catalyst_aligned_buy: bool
    catalyst_aligned_sell: bool
    last_updated_utc: datetime

