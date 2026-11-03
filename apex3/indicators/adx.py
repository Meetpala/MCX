"""Average Directional Index (ADX) with +DI / −DI for APEX-3.

ADX quantifies trend strength on a 0-100 scale.  It is derived from the
Directional Movement indicators (+DM / −DM) smoothed with Wilder's method.

Wilder's smoothing (alpha = 1/period):
    smoothed_t = smoothed_(t-1) - smoothed_(t-1) / period + value_t

Calculation steps (all on 5-minute OHLC candles or any timeframe):
    1. Compute +DM and −DM per bar.
    2. Compute True Range (TR) per bar.
    3. Apply Wilder's smoothing to +DM, −DM, and TR over `period` bars.
    4. +DI = 100 * smoothed(+DM) / smoothed(TR)
       −DI = 100 * smoothed(−DM) / smoothed(TR)
    5. DX  = 100 * |+DI − −DI| / (+DI + −DI)
    6. ADX = Wilder-smoothed DX over `period` bars.
"""

from __future__ import annotations


def compute_adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> dict[str, list[float]]:
    """Compute ADX, +DI, and −DI using Wilder's smoothing.

    Works on any OHLC timeframe (designed for 5-minute candles).

    Args:
        highs:  List of high prices per bar.
        lows:   List of low prices per bar.
        closes: List of close prices per bar.
        period: ADX look-back period.  Must be >= 2.

    Returns:
        Dictionary with keys:
            'adx'      — list of ADX values
            'plus_di'  — list of +DI values (same length as adx)
            'minus_di' — list of −DI values (same length as adx)

        All three lists have the same length.  The first ADX value
        appears after 2 * period bars of input data.  Returns empty
        lists if there is insufficient data.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_inputs(highs, lows, closes, period)

    n = len(highs)
    # Need at least 2*period + 1 bars to produce a single ADX value
    if n < 2 * period + 1:
        return {"adx": [], "plus_di": [], "minus_di": []}

    # --- Step 1: raw +DM, −DM, TR per bar (starting from bar 1) ---
    plus_dm_raw: list[float] = []
    minus_dm_raw: list[float] = []
    tr_raw: list[float] = []

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        mdm = down_move if (down_move > up_move and down_move > 0) else 0.0

        plus_dm_raw.append(pdm)
        minus_dm_raw.append(mdm)

        prev_close = closes[i - 1]
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
        tr_raw.append(tr)

    # --- Step 2: Wilder-smooth +DM, −DM, TR over first `period` values ---
    def wilder_seed(values: list[float]) -> float:
        return sum(values[:period])

    smooth_pdm = wilder_seed(plus_dm_raw)
    smooth_mdm = wilder_seed(minus_dm_raw)
    smooth_tr = wilder_seed(tr_raw)

    # We'll collect +DI, −DI starting from index `period - 1` in the raw
    # arrays (which corresponds to bar `period` in the original data).
    plus_di_all: list[float] = []
    minus_di_all: list[float] = []

    def _di(smooth_dm: float, smooth_tr_val: float) -> float:
        return 100.0 * smooth_dm / smooth_tr_val if smooth_tr_val != 0 else 0.0

    plus_di_all.append(_di(smooth_pdm, smooth_tr))
    minus_di_all.append(_di(smooth_mdm, smooth_tr))

    for i in range(period, len(tr_raw)):
        smooth_pdm = smooth_pdm - smooth_pdm / period + plus_dm_raw[i]
        smooth_mdm = smooth_mdm - smooth_mdm / period + minus_dm_raw[i]
        smooth_tr = smooth_tr - smooth_tr / period + tr_raw[i]

        plus_di_all.append(_di(smooth_pdm, smooth_tr))
        minus_di_all.append(_di(smooth_mdm, smooth_tr))

    # --- Step 3: DX series ---
    dx_all: list[float] = []
    for pdi, mdi in zip(plus_di_all, minus_di_all):
        denom = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / denom if denom != 0 else 0.0
        dx_all.append(dx)

    # --- Step 4: Wilder-smooth DX to get ADX ---
    if len(dx_all) < period:
        return {"adx": [], "plus_di": [], "minus_di": []}

    # Seed ADX with SMA of first `period` DX values
    adx_seed = sum(dx_all[:period]) / period
    adx_values: list[float] = [adx_seed]

    for i in range(period, len(dx_all)):
        prev_adx = adx_values[-1]
        adx_values.append((prev_adx * (period - 1) + dx_all[i]) / period)

    # Align +DI / −DI to match ADX output length
    # ADX starts at dx_all index (period - 1), i.e. plus_di_all index (period - 1)
    offset = period - 1
    plus_di_out = plus_di_all[offset : offset + len(adx_values)]
    minus_di_out = minus_di_all[offset : offset + len(adx_values)]

    return {
        "adx": adx_values,
        "plus_di": plus_di_out,
        "minus_di": minus_di_out,
    }


def _validate_inputs(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
) -> None:
    """Validate ADX inputs."""
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    if not highs or not lows or not closes:
        raise ValueError("highs, lows, and closes must not be empty")
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError(
            f"highs ({len(highs)}), lows ({len(lows)}), and closes "
            f"({len(closes)}) must all have the same length"
        )
