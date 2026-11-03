"""Average True Range (ATR) using Wilder's smoothing for APEX-3.

ATR measures volatility by decomposing the entire range of a bar.

True Range is the greatest of:
    1. Current High − Current Low
    2. |Current High − Previous Close|
    3. |Current Low  − Previous Close|

Wilder's smoothing (exponential with alpha = 1/period):
    ATR_t = ATR_(t-1) * (period - 1) / period  +  TR_t / period

The first ATR value is the simple average of the first `period` true ranges.
For the very first bar (index 0) where there is no previous close, True Range
is simply High − Low.
"""

from __future__ import annotations


def compute_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float]:
    """Compute Average True Range using Wilder's smoothing.

    Args:
        highs:  List of high prices per bar.
        lows:   List of low prices per bar.
        closes: List of close prices per bar.
        period: ATR look-back period. Must be >= 1.

    Returns:
        List of ATR values.  Length = len(highs) - period.
        The first ATR value corresponds to bar index `period` (0-based).
        Returns an empty list when there are fewer than period + 1 bars
        (we need `period` true-range values, which requires period + 1
        price bars since TR uses a previous close starting from bar 1).

        Special case: when period == 1, the first ATR value equals TR of
        bar 1 and subsequent values are also just TR (Wilder's smoothing
        with period=1 degenerates to the raw TR).

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_ohlc(highs, lows, closes, period)

    n = len(highs)

    # --- compute true range series ---
    # Bar 0: no previous close, TR = H - L
    true_ranges: list[float] = [highs[0] - lows[0]]
    for i in range(1, n):
        prev_close = closes[i - 1]
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
        true_ranges.append(tr)

    # We need at least `period` true-range values (indices 1..period)
    # to seed the first ATR.  Bar 0's TR is only used if period == 1.
    # For period >= 2 we seed from TR[1] .. TR[period], requiring n >= period+1.
    if n < period + 1 and period > 1:
        return []

    # Seed: simple average of the first `period` true-range values
    # starting from index 1 (bar 1) so we always use TR values that have
    # a valid previous close — except when period == 1 where we start from
    # index 1 as well.
    start = 1 if period > 1 else 1
    if n < start + period:
        return []

    seed = sum(true_ranges[start : start + period]) / period
    atr_values: list[float] = [seed]

    # Wilder's smoothing for subsequent bars
    for i in range(start + period, n):
        prev_atr = atr_values[-1]
        atr_values.append(
            (prev_atr * (period - 1) + true_ranges[i]) / period
        )

    return atr_values


def _validate_ohlc(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
) -> None:
    """Shared validation for OHLC-based indicators."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if not highs or not lows or not closes:
        raise ValueError("highs, lows, and closes must not be empty")
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError(
            f"highs ({len(highs)}), lows ({len(lows)}), and closes "
            f"({len(closes)}) must all have the same length"
        )
