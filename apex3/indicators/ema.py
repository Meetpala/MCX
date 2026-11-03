"""Exponential Moving Average (EMA) calculations for APEX-3.

EMA gives more weight to recent prices, making it more responsive
than a simple moving average. The smoothing multiplier is 2/(period+1).

Formula:
    EMA_today = (Price_today * k) + (EMA_yesterday * (1 - k))
    where k = 2 / (period + 1)
"""

from __future__ import annotations


def compute_ema(prices: list[float], period: int) -> list[float]:
    """Compute Exponential Moving Average over a price series.

    The first EMA value is seeded with the SMA of the first `period` prices.
    Subsequent values apply the standard EMA formula.

    Args:
        prices: List of price values (e.g. closing prices).
        period: EMA look-back period. Must be >= 1.

    Returns:
        List of EMA values aligned to the END of `prices`.
        Length = len(prices) - period + 1.
        Returns an empty list if there are fewer prices than `period`.

    Raises:
        ValueError: If `period` < 1 or `prices` is empty.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if not prices:
        raise ValueError("prices must not be empty")

    if len(prices) < period:
        return []

    # Seed with SMA of the first `period` prices
    sma_seed: float = sum(prices[:period]) / period
    multiplier: float = 2.0 / (period + 1)

    ema_values: list[float] = [sma_seed]
    for price in prices[period:]:
        prev = ema_values[-1]
        ema_values.append(price * multiplier + prev * (1.0 - multiplier))

    return ema_values


def ema_slope(ema_values: list[float], lookback: int = 3) -> float | None:
    """Determine the slope direction of an EMA series.

    Compares the last EMA value to the value `lookback` bars earlier.

    Args:
        ema_values: Pre-computed EMA series.
        lookback: Number of bars to look back for slope comparison.
                  Must be >= 1.

    Returns:
        +1.0 if the EMA is rising (last > lookback bars ago).
        -1.0 if the EMA is falling (last < lookback bars ago).
         0.0 if the EMA is flat (exactly equal).
        None  if there are insufficient values (need at least lookback + 1).

    Raises:
        ValueError: If `lookback` < 1 or `ema_values` is empty.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if not ema_values:
        raise ValueError("ema_values must not be empty")

    if len(ema_values) < lookback + 1:
        return None

    diff: float = ema_values[-1] - ema_values[-(lookback + 1)]
    if diff > 0:
        return 1.0
    elif diff < 0:
        return -1.0
    return 0.0
