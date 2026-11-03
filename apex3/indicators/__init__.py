"""APEX-3 Technical Indicator Library.

Pure-function implementations of all technical indicators used by the
APEX-3 intraday trading system.  Every function accepts plain Python
lists (no pandas dependency) and has no side effects.

Modules:
    ema   — Exponential Moving Average & slope detection
    atr   — Average True Range (Wilder's smoothing)
    adx   — Average Directional Index with +DI / −DI
    vwap  — Session VWAP (tradeable stocks only, NOT for index)
    swing — Structural swing-high / swing-low detection
"""

from apex3.indicators.adx import compute_adx
from apex3.indicators.atr import compute_atr
from apex3.indicators.ema import compute_ema, ema_slope
from apex3.indicators.swing import find_swing_highs, find_swing_lows
from apex3.indicators.vwap import compute_rolling_vwap, compute_session_vwap

__all__ = [
    "compute_ema",
    "ema_slope",
    "compute_atr",
    "compute_adx",
    "compute_session_vwap",
    "compute_rolling_vwap",
    "find_swing_lows",
    "find_swing_highs",
]
