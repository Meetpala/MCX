"""Session VWAP (Volume-Weighted Average Price) for APEX-3.

IMPORTANT: VWAP is for TRADEABLE STOCKS only.
           The NIFTY 50 index does NOT use VWAP because it has no
           real traded volume — index "volume" is synthetic and
           meaningless for VWAP purposes.

VWAP calculation:
    Typical Price = (High + Low + Close) / 3
    Cumulative_TPV = Σ (Typical_Price_i × Volume_i)
    Cumulative_Vol = Σ Volume_i
    VWAP = Cumulative_TPV / Cumulative_Vol

Session boundaries: The CALLER is responsible for passing only
intraday bars belonging to a single session.  VWAP resets at
the start of each trading day (09:15 IST for NSE).
"""

from __future__ import annotations


def compute_session_vwap(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> float:
    """Compute the cumulative session VWAP (single scalar value).

    Returns the VWAP as of the last bar in the provided session data.

    Args:
        highs:   High prices for the session bars.
        lows:    Low prices for the session bars.
        closes:  Close prices for the session bars.
        volumes: Traded volumes for the session bars.
                 Must be for a TRADEABLE STOCK (not an index).

    Returns:
        The session VWAP as a single float.

    Raises:
        ValueError: If inputs are invalid or total volume is zero.
    """
    _validate_vwap_inputs(highs, lows, closes, volumes)

    cum_tpv: float = 0.0
    cum_vol: float = 0.0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        typical_price = (h + l + c) / 3.0
        cum_tpv += typical_price * v
        cum_vol += v

    if cum_vol == 0.0:
        raise ValueError(
            "Total volume is zero — cannot compute VWAP. "
            "Ensure this is a tradeable stock, not an index."
        )

    return cum_tpv / cum_vol


def compute_rolling_vwap(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> list[float]:
    """Compute a running VWAP series across the session.

    Each element i represents the cumulative VWAP from bar 0 through bar i.

    Args:
        highs:   High prices for the session bars.
        lows:    Low prices for the session bars.
        closes:  Close prices for the session bars.
        volumes: Traded volumes for the session bars.
                 Must be for a TRADEABLE STOCK (not an index).

    Returns:
        List of VWAP values, one per bar.  If cumulative volume at
        any bar is zero, that bar's VWAP is set to the typical price
        of that bar (degenerate case — should not occur for real
        traded stocks during market hours).

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_vwap_inputs(highs, lows, closes, volumes)

    cum_tpv: float = 0.0
    cum_vol: float = 0.0
    vwap_series: list[float] = []

    for h, l, c, v in zip(highs, lows, closes, volumes):
        typical_price = (h + l + c) / 3.0
        cum_tpv += typical_price * v
        cum_vol += v

        if cum_vol == 0.0:
            # Edge case: no volume yet — use typical price as placeholder
            vwap_series.append(typical_price)
        else:
            vwap_series.append(cum_tpv / cum_vol)

    return vwap_series


def _validate_vwap_inputs(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> None:
    """Validate VWAP inputs.

    Note: VWAP should only be used for tradeable stocks.
    NIFTY 50 index does NOT have meaningful volume for VWAP.
    """
    if not highs or not lows or not closes or not volumes:
        raise ValueError("highs, lows, closes, and volumes must not be empty")
    lengths = {len(highs), len(lows), len(closes), len(volumes)}
    if len(lengths) != 1:
        raise ValueError(
            f"All inputs must have the same length: "
            f"highs={len(highs)}, lows={len(lows)}, "
            f"closes={len(closes)}, volumes={len(volumes)}"
        )
