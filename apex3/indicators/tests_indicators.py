"""Comprehensive tests for the APEX-3 indicator library.

Run with:  pytest apex3/indicators/tests_indicators.py -v

Each indicator function has at least 3 tests:
    1. Normal / happy-path calculation
    2. Edge case (insufficient data)
    3. Known-value verification (hand-calculated or reference values)
"""

from __future__ import annotations

import math

import pytest

from apex3.indicators.ema import compute_ema, ema_slope
from apex3.indicators.atr import compute_atr
from apex3.indicators.adx import compute_adx
from apex3.indicators.vwap import compute_session_vwap, compute_rolling_vwap
from apex3.indicators.swing import find_swing_lows, find_swing_highs


# ──────────────────────────────────────────────────────────────────────
#  EMA tests
# ──────────────────────────────────────────────────────────────────────
class TestComputeEma:
    """Tests for compute_ema."""

    def test_normal_calculation(self) -> None:
        """EMA of a simple 5-element series with period 3."""
        prices = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = compute_ema(prices, period=3)
        # Seed SMA = (2+4+6)/3 = 4.0
        # k = 2/4 = 0.5
        # EMA[1] = 8*0.5 + 4*0.5 = 6.0
        # EMA[2] = 10*0.5 + 6*0.5 = 8.0
        assert len(result) == 3
        assert result[0] == pytest.approx(4.0)
        assert result[1] == pytest.approx(6.0)
        assert result[2] == pytest.approx(8.0)

    def test_insufficient_data(self) -> None:
        """Return empty list when prices < period."""
        assert compute_ema([1.0, 2.0], period=5) == []

    def test_known_values_period_1(self) -> None:
        """EMA with period=1 should equal raw prices (k = 1.0)."""
        prices = [10.0, 20.0, 30.0]
        result = compute_ema(prices, period=1)
        assert result == pytest.approx(prices)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be >= 1"):
            compute_ema([1.0], period=0)

    def test_empty_prices_raises(self) -> None:
        with pytest.raises(ValueError, match="prices must not be empty"):
            compute_ema([], period=3)


class TestEmaSlope:
    """Tests for ema_slope."""

    def test_rising(self) -> None:
        assert ema_slope([1.0, 2.0, 3.0, 4.0], lookback=3) == 1.0

    def test_falling(self) -> None:
        assert ema_slope([4.0, 3.0, 2.0, 1.0], lookback=3) == -1.0

    def test_flat(self) -> None:
        assert ema_slope([5.0, 5.0, 5.0, 5.0], lookback=3) == 0.0

    def test_insufficient_data(self) -> None:
        """Return None when not enough values for the lookback."""
        assert ema_slope([1.0, 2.0], lookback=3) is None

    def test_invalid_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            ema_slope([1.0, 2.0], lookback=0)


# ──────────────────────────────────────────────────────────────────────
#  ATR tests
# ──────────────────────────────────────────────────────────────────────
class TestComputeAtr:
    """Tests for compute_atr."""

    def test_normal_calculation(self) -> None:
        """ATR on a small 5-bar dataset with period=2."""
        highs  = [12.0, 13.0, 14.0, 13.5, 15.0]
        lows   = [10.0, 11.0, 12.0, 11.5, 13.0]
        closes = [11.0, 12.0, 13.0, 12.0, 14.0]
        result = compute_atr(highs, lows, closes, period=2)
        # TR[0] = 12-10 = 2.0 (no prev close)
        # TR[1] = max(13-11, |13-11|, |11-11|) = 2.0
        # TR[2] = max(14-12, |14-12|, |12-12|) = 2.0
        # TR[3] = max(13.5-11.5, |13.5-13|, |11.5-13|) = max(2, 0.5, 1.5) = 2.0
        # TR[4] = max(15-13, |15-12|, |13-12|) = max(2, 3, 1) = 3.0
        # seed = (TR[1]+TR[2])/2 = 2.0
        # ATR[1] = (2.0*1 + 2.0)/2 = 2.0
        # ATR[2] = (2.0*1 + 3.0)/2 = 2.5
        assert len(result) == 3
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(2.5)

    def test_insufficient_data(self) -> None:
        """Return empty list when bars < period + 1."""
        highs  = [10.0, 11.0]
        lows   = [9.0, 10.0]
        closes = [9.5, 10.5]
        assert compute_atr(highs, lows, closes, period=14) == []

    def test_known_gap_scenario(self) -> None:
        """TR accounts for gap-up correctly (|H - prev_close| > H - L)."""
        #  Bar 0: H=100, L=95, C=97  (no prev close, TR=5)
        #  Bar 1: gap-up H=110, L=108, C=109
        #         TR = max(110-108, |110-97|, |108-97|)
        #            = max(2, 13, 11) = 13
        #  Bar 2: H=112, L=107, C=111
        #         TR = max(5, |112-109|, |107-109|)
        #            = max(5, 3, 2) = 5
        highs  = [100.0, 110.0, 112.0]
        lows   = [95.0, 108.0, 107.0]
        closes = [97.0, 109.0, 111.0]
        result = compute_atr(highs, lows, closes, period=1)
        # period=1: seed = TR[1] = 13.0
        # ATR[1] = (13*0 + 5)/1 = 5.0
        assert len(result) == 2
        assert result[0] == pytest.approx(13.0)
        assert result[1] == pytest.approx(5.0)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be >= 1"):
            compute_atr([1.0], [1.0], [1.0], period=0)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_atr([1.0, 2.0], [1.0], [1.0], period=1)


# ──────────────────────────────────────────────────────────────────────
#  ADX tests
# ──────────────────────────────────────────────────────────────────────
class TestComputeAdx:
    """Tests for compute_adx."""

    def test_normal_output_structure(self) -> None:
        """ADX returns dict with all three keys of equal length."""
        # Generate a simple trending series (30 bars, period 5)
        n = 30
        highs  = [100.0 + i * 1.5 for i in range(n)]
        lows   = [98.0 + i * 1.5 for i in range(n)]
        closes = [99.0 + i * 1.5 for i in range(n)]
        result = compute_adx(highs, lows, closes, period=5)
        assert "adx" in result
        assert "plus_di" in result
        assert "minus_di" in result
        assert len(result["adx"]) == len(result["plus_di"])
        assert len(result["adx"]) == len(result["minus_di"])
        assert len(result["adx"]) > 0

    def test_insufficient_data(self) -> None:
        """Return empty lists when bars < 2*period + 1."""
        highs  = [10.0 + i for i in range(10)]
        lows   = [9.0 + i for i in range(10)]
        closes = [9.5 + i for i in range(10)]
        result = compute_adx(highs, lows, closes, period=14)
        assert result == {"adx": [], "plus_di": [], "minus_di": []}

    def test_strong_uptrend_high_adx(self) -> None:
        """In a clean uptrend, ADX should be high and +DI > −DI."""
        n = 60
        highs  = [100.0 + i * 2.0 for i in range(n)]
        lows   = [98.0 + i * 2.0 for i in range(n)]
        closes = [99.0 + i * 2.0 for i in range(n)]
        result = compute_adx(highs, lows, closes, period=14)
        # In a clean trend, last ADX should be well above 25
        assert result["adx"][-1] > 25.0
        # +DI should dominate −DI
        assert result["plus_di"][-1] > result["minus_di"][-1]

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be >= 2"):
            compute_adx([1.0], [1.0], [1.0], period=1)


# ──────────────────────────────────────────────────────────────────────
#  VWAP tests
# ──────────────────────────────────────────────────────────────────────
class TestComputeSessionVwap:
    """Tests for compute_session_vwap."""

    def test_normal_calculation(self) -> None:
        """VWAP of a 3-bar session with known values."""
        highs   = [105.0, 108.0, 110.0]
        lows    = [100.0, 103.0, 106.0]
        closes  = [103.0, 106.0, 108.0]
        volumes = [1000.0, 1500.0, 2000.0]
        # TP[0] = (105+100+103)/3 = 102.6667
        # TP[1] = (108+103+106)/3 = 105.6667
        # TP[2] = (110+106+108)/3 = 108.0
        # VWAP = (102.6667*1000 + 105.6667*1500 + 108*2000) / 4500
        #      = (102666.7 + 158500.0 + 216000.0) / 4500
        #      = 477166.7 / 4500 ≈ 106.037
        result = compute_session_vwap(highs, lows, closes, volumes)
        expected_tpv = (
            (105 + 100 + 103) / 3 * 1000
            + (108 + 103 + 106) / 3 * 1500
            + (110 + 106 + 108) / 3 * 2000
        )
        expected = expected_tpv / 4500
        assert result == pytest.approx(expected)

    def test_zero_volume_raises(self) -> None:
        """Zero total volume should raise ValueError."""
        with pytest.raises(ValueError, match="Total volume is zero"):
            compute_session_vwap([10.0], [9.0], [9.5], [0.0])

    def test_single_bar(self) -> None:
        """VWAP of a single bar equals its typical price."""
        h, l, c, v = 110.0, 100.0, 105.0, 500.0
        expected_tp = (h + l + c) / 3.0
        assert compute_session_vwap([h], [l], [c], [v]) == pytest.approx(
            expected_tp
        )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            compute_session_vwap([], [], [], [])


class TestComputeRollingVwap:
    """Tests for compute_rolling_vwap."""

    def test_normal_rolling(self) -> None:
        """Rolling VWAP produces one value per bar."""
        highs   = [10.0, 12.0, 14.0]
        lows    = [8.0, 10.0, 12.0]
        closes  = [9.0, 11.0, 13.0]
        volumes = [100.0, 200.0, 300.0]
        result = compute_rolling_vwap(highs, lows, closes, volumes)
        assert len(result) == 3
        # First bar VWAP = typical price of bar 0
        tp0 = (10 + 8 + 9) / 3.0
        assert result[0] == pytest.approx(tp0)

    def test_matches_session_vwap_at_end(self) -> None:
        """Last element of rolling VWAP should equal session VWAP."""
        highs   = [105.0, 108.0, 110.0]
        lows    = [100.0, 103.0, 106.0]
        closes  = [103.0, 106.0, 108.0]
        volumes = [1000.0, 1500.0, 2000.0]
        rolling = compute_rolling_vwap(highs, lows, closes, volumes)
        session = compute_session_vwap(highs, lows, closes, volumes)
        assert rolling[-1] == pytest.approx(session)

    def test_single_bar_rolling(self) -> None:
        """Single-bar rolling VWAP is a list with one typical price."""
        result = compute_rolling_vwap([100.0], [90.0], [95.0], [500.0])
        assert len(result) == 1
        assert result[0] == pytest.approx((100 + 90 + 95) / 3.0)


# ──────────────────────────────────────────────────────────────────────
#  Swing tests
# ──────────────────────────────────────────────────────────────────────
class TestFindSwingLows:
    """Tests for find_swing_lows."""

    def test_normal_detection(self) -> None:
        """Detect a clear V-shaped swing low with lookback=2."""
        lows = [10.0, 8.0, 6.0, 4.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        result = find_swing_lows(lows, lookback=2)
        # Index 4 (value 2.0) is the swing low
        assert (4, 2.0) in result

    def test_insufficient_data(self) -> None:
        """Return empty list when series too short."""
        lows = [5.0, 3.0, 5.0]
        # Need 2*3+1 = 7 bars for lookback=3
        assert find_swing_lows(lows, lookback=3) == []

    def test_known_multiple_swings(self) -> None:
        """Detect two swing lows in a W-pattern."""
        lows = [10.0, 8.0, 5.0, 8.0, 10.0, 8.0, 3.0, 8.0, 10.0]
        result = find_swing_lows(lows, lookback=2)
        indices = [idx for idx, _ in result]
        assert 2 in indices  # first valley at 5.0
        assert 6 in indices  # second valley at 3.0

    def test_flat_is_not_swing(self) -> None:
        """Equal adjacent values should NOT be swing lows (strict <)."""
        lows = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        assert find_swing_lows(lows, lookback=2) == []

    def test_invalid_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            find_swing_lows([1.0], lookback=0)


class TestFindSwingHighs:
    """Tests for find_swing_highs."""

    def test_normal_detection(self) -> None:
        """Detect a clear inverted-V swing high with lookback=2."""
        highs = [2.0, 4.0, 6.0, 8.0, 10.0, 8.0, 6.0, 4.0, 2.0]
        result = find_swing_highs(highs, lookback=2)
        assert (4, 10.0) in result

    def test_insufficient_data(self) -> None:
        """Return empty list when series too short."""
        highs = [5.0, 10.0, 5.0]
        assert find_swing_highs(highs, lookback=3) == []

    def test_known_double_top(self) -> None:
        """Detect two swing highs in an M-pattern."""
        highs = [2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 10.0, 4.0, 2.0]
        result = find_swing_highs(highs, lookback=2)
        indices = [idx for idx, _ in result]
        assert 2 in indices   # first peak at 8.0
        assert 6 in indices   # second peak at 10.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="price list must not be empty"):
            find_swing_highs([], lookback=2)
