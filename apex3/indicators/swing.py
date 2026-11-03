"""Structural swing-point detection for APEX-3.

A swing low is a bar whose low is lower than the lows of `lookback`
bars on EACH side (left and right).  Similarly, a swing high has a
high that is higher than the highs of `lookback` bars on each side.

IMPORTANT: Only COMPLETED (closed) candles can be swing points.
The current unfinished bar is NEVER a confirmed swing.  The caller
must exclude the live bar before passing data to these functions.
"""

from __future__ import annotations


def find_swing_lows(
    lows: list[float],
    lookback: int = 3,
) -> list[tuple[int, float]]:
    """Detect confirmed swing lows in a price series.

    A bar at index *i* is a swing low if:
        lows[i] < lows[j]  for all j in [i-lookback, i+lookback], j ≠ i

    Args:
        lows:     List of low prices (completed candles only).
        lookback: Number of bars required on each side.  Must be >= 1.

    Returns:
        List of (index, price) tuples for each confirmed swing low,
        ordered by index.  Returns an empty list if the series is
        too short (need at least 2 * lookback + 1 bars).

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_swing_inputs(lows, lookback)

    min_length = 2 * lookback + 1
    if len(lows) < min_length:
        return []

    swings: list[tuple[int, float]] = []
    # The last possible swing-low index is len(lows) - lookback - 1
    for i in range(lookback, len(lows) - lookback):
        candidate = lows[i]
        is_swing = True
        for offset in range(1, lookback + 1):
            if candidate >= lows[i - offset] or candidate >= lows[i + offset]:
                is_swing = False
                break
        if is_swing:
            swings.append((i, candidate))

    return swings


def find_swing_highs(
    highs: list[float],
    lookback: int = 3,
) -> list[tuple[int, float]]:
    """Detect confirmed swing highs in a price series.

    A bar at index *i* is a swing high if:
        highs[i] > highs[j]  for all j in [i-lookback, i+lookback], j ≠ i

    Args:
        highs:    List of high prices (completed candles only).
        lookback: Number of bars required on each side.  Must be >= 1.

    Returns:
        List of (index, price) tuples for each confirmed swing high,
        ordered by index.  Returns an empty list if the series is
        too short (need at least 2 * lookback + 1 bars).

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_swing_inputs(highs, lookback)

    min_length = 2 * lookback + 1
    if len(highs) < min_length:
        return []

    swings: list[tuple[int, float]] = []
    for i in range(lookback, len(highs) - lookback):
        candidate = highs[i]
        is_swing = True
        for offset in range(1, lookback + 1):
            if candidate <= highs[i - offset] or candidate <= highs[i + offset]:
                is_swing = False
                break
        if is_swing:
            swings.append((i, candidate))

    return swings


def _validate_swing_inputs(prices: list[float], lookback: int) -> None:
    """Validate swing-detection inputs."""
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if not prices:
        raise ValueError("price list must not be empty")
