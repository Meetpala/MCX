"""Tests for Market Regime Filter (NIFTY 50 OHLC trend & ADX)."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytest

from apex3.config import RegimeConfig
from apex3.models import Candle, Quote
from apex3.strategy.regime import MarketRegime, evaluate_market_regime
from apex3.tests.conftest import generate_candles

IST = ZoneInfo("Asia/Kolkata")


def test_regime_bullish_symmetry():
    """Verify bullish regime detection when price > EMA-20, rising slope, strong ADX."""
    config = RegimeConfig(ema_period=10, min_adx=20.0, adx_period=10, ema_slope_lookback=3)
    # Generate strong uptrend candles
    candles = generate_candles(count=40, base_price=20000.0, trend=50.0)
    eval_result = evaluate_market_regime(candles, config)

    assert eval_result.regime == MarketRegime.BULLISH
    assert eval_result.gate.status.value == "PASS"
    assert eval_result.ema_slope_val > 0
    assert eval_result.current_close > eval_result.ema_20
    assert eval_result.adx_val >= 20.0


def test_regime_bearish_symmetry():
    """Verify bearish regime detection when price < EMA-20, falling slope, strong ADX."""
    config = RegimeConfig(ema_period=10, min_adx=20.0, adx_period=10, ema_slope_lookback=3)
    # Generate strong downtrend candles
    candles = generate_candles(count=40, base_price=20000.0, trend=-50.0)
    eval_result = evaluate_market_regime(candles, config)

    assert eval_result.regime == MarketRegime.BEARISH
    assert eval_result.gate.status.value == "PASS"
    assert eval_result.ema_slope_val < 0
    assert eval_result.current_close < eval_result.ema_20
    assert eval_result.adx_val >= 20.0


def test_regime_neutral_on_insufficient_data():
    """Verify regime fails closed to NEUTRAL when bars are insufficient."""
    config = RegimeConfig(ema_period=20, min_adx=22.0, adx_period=14)
    candles = generate_candles(count=10, base_price=20000.0, trend=10.0)
    eval_result = evaluate_market_regime(candles, config)

    assert eval_result.regime == MarketRegime.NEUTRAL
    assert eval_result.gate.status.value == "FAIL"
    assert "Insufficient completed" in eval_result.gate.reason


def test_regime_fails_on_stale_quote():
    """Verify regime fails closed when quote is stale."""
    config = RegimeConfig(ema_period=10, min_adx=20.0, adx_period=10)
    candles = generate_candles(count=40, base_price=20000.0, trend=50.0)
    stale_quote = Quote(
        symbol="NIFTY 50",
        exchange="NSE",
        last_price=21000.0,
        bid=20999.0,
        ask=21001.0,
        volume=100000,
        timestamp=datetime.now(IST) - timedelta(seconds=10),
        is_fresh=False,
    )
    eval_result = evaluate_market_regime(candles, config, benchmark_quote=stale_quote)
    assert eval_result.regime == MarketRegime.NEUTRAL
    assert "stale" in eval_result.gate.reason.lower()
