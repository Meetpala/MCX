"""Tests for Signal Generator and 8 Entry Gates."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytest

from apex3.config import SignalConfig
from apex3.models import Candle, Decision, InstrumentInfo, OpeningRange, Quote, Side
from apex3.strategy.regime import MarketRegime
from apex3.strategy.signal import build_opening_range, evaluate_signal
from apex3.tests.conftest import generate_candles

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def signal_cfg() -> SignalConfig:
    return SignalConfig(
        min_relative_volume=1.5,
        volume_avg_period=20,
        max_spread_bps=8.0,
        min_circuit_distance_pct=2.0,
        relative_strength_lookback=20,
    )


@pytest.fixture
def inst_info() -> InstrumentInfo:
    return InstrumentInfo(
        symbol="INFY",
        upper_circuit=2000.0,
        lower_circuit=1000.0,
        tick_size=0.05,
    )


def test_long_signal_full_pass(signal_cfg, inst_info):
    """Verify PROPOSED_LONG when all 8 gates pass."""
    or_range = OpeningRange(symbol="INFY", high=1500.0, low=1493.0, is_complete=True, candle_count=3)
    stock_candles = generate_candles(count=25, base_price=1496.0, trend=0.1)
    # Make the latest candle breakout above OR high (1500) with strong volume
    last_candle = Candle(
        timestamp=datetime.now(IST),
        open=1501.0,
        high=1504.0,
        low=1500.5,
        close=1503.5,  # > 1500.0
        volume=50000,   # > 1.5x average
        is_complete=True,
    )
    stock_candles.append(last_candle)
    bench_candles = generate_candles(count=26, base_price=20000.0, trend=0.1)

    quote = Quote(
        symbol="INFY",
        exchange="NSE",
        last_price=1503.5,
        bid=1503.45,
        ask=1503.55,  # Tight spread < 8 bps
        volume=50000,
        timestamp=datetime.now(IST),
        is_fresh=True,
    )

    sig = evaluate_signal(
        symbol="INFY",
        stock_candles_5m=stock_candles,
        benchmark_candles_5m=bench_candles,
        opening_range=or_range,
        market_regime=MarketRegime.BULLISH,
        instrument=inst_info,
        quote=quote,
        config=signal_cfg,
        is_entry_window=True,
    )

    assert sig.decision == Decision.PROPOSED_LONG
    assert sig.side == Side.BUY
    assert len(sig.gates) >= 9  # all gates passed


def test_short_signal_mirror_pass(signal_cfg, inst_info):
    """Verify PROPOSED_SHORT exact mirror condition when all 8 gates pass."""
    or_range = OpeningRange(symbol="INFY", high=1480.0, low=1470.0, is_complete=True, candle_count=3)
    stock_candles = generate_candles(count=25, base_price=1475.0, trend=-0.2)
    # Make the latest candle breakout below OR low (1470) with strong volume
    last_candle = Candle(
        timestamp=datetime.now(IST),
        open=1469.0,
        high=1469.5,
        low=1460.0,
        close=1462.0,  # < 1470.0
        volume=50000,
        is_complete=True,
    )
    stock_candles.append(last_candle)
    bench_candles = generate_candles(count=26, base_price=20000.0, trend=-0.1)

    quote = Quote(
        symbol="INFY",
        exchange="NSE",
        last_price=1462.0,
        bid=1461.95,
        ask=1462.05,
        volume=50000,
        timestamp=datetime.now(IST),
        is_fresh=True,
    )

    sig = evaluate_signal(
        symbol="INFY",
        stock_candles_5m=stock_candles,
        benchmark_candles_5m=bench_candles,
        opening_range=or_range,
        market_regime=MarketRegime.BEARISH,
        instrument=inst_info,
        quote=quote,
        config=signal_cfg,
        is_entry_window=True,
    )

    assert sig.decision == Decision.PROPOSED_SHORT
    assert sig.side == Side.SELL


def test_signal_fails_on_excessive_spread(signal_cfg, inst_info):
    """Reject candidate when bid-ask spread > 8 bps."""
    or_range = OpeningRange(symbol="INFY", high=1500.0, low=1480.0, is_complete=True, candle_count=3)
    stock_candles = generate_candles(count=25, base_price=1505.0, trend=1.0)
    bench_candles = generate_candles(count=25, base_price=20000.0, trend=0.1)

    # Wide spread: bid 1500, ask 1515 -> ~100 bps
    quote = Quote(
        symbol="INFY",
        exchange="NSE",
        last_price=1508.0,
        bid=1500.0,
        ask=1515.0,
        volume=50000,
        timestamp=datetime.now(IST),
        is_fresh=True,
    )

    sig = evaluate_signal(
        symbol="INFY",
        stock_candles_5m=stock_candles,
        benchmark_candles_5m=bench_candles,
        opening_range=or_range,
        market_regime=MarketRegime.BULLISH,
        instrument=inst_info,
        quote=quote,
        config=signal_cfg,
        is_entry_window=True,
    )

    assert sig.decision == Decision.WAIT
    assert "EXCESSIVE_SPREAD" in sig.reason_codes
