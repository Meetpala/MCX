"""APEX-3 Test Suite Fixtures and Mock Data."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytest

from apex3.config import AppConfig, load_config
from apex3.models import Candle, InstrumentInfo, MarginInfo, Quote

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def base_config() -> AppConfig:
    """Fixture providing loaded default application configuration."""
    return load_config()


@pytest.fixture
def sample_instrument() -> InstrumentInfo:
    return InstrumentInfo(
        symbol="RELIANCE",
        exchange="NSE",
        tick_size=0.05,
        lot_size=1,
        upper_circuit=3500.0,
        lower_circuit=2500.0,
        mis_available=True,
    )


@pytest.fixture
def sample_margin() -> MarginInfo:
    return MarginInfo(
        equity_available=100000.0,
        equity_used=0.0,
        collateral=0.0,
        net_value=100000.0,
        timestamp=datetime.now(IST),
    )


def generate_candles(
    count: int,
    base_price: float = 1000.0,
    trend: float = 1.0,
    start_time: datetime | None = None,
    timeframe_minutes: int = 5,
) -> list[Candle]:
    """Helper to generate synthetic candles for deterministic tests."""
    if start_time is None:
        start_time = datetime(2026, 10, 20, 9, 15, tzinfo=IST)

    candles = []
    p = base_price
    for i in range(count):
        t = start_time + timedelta(minutes=i * timeframe_minutes)
        c_open = p
        c_close = p + trend
        c_high = max(c_open, c_close) + 2.0
        c_low = min(c_open, c_close) - 2.0
        p = c_close
        candles.append(
            Candle(
                timestamp=t,
                open=c_open,
                high=c_high,
                low=c_low,
                close=c_close,
                volume=15000,
                is_complete=True,
            )
        )
    return candles
