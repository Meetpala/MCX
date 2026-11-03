"""
core/regime_and_strategy.py
---------------------------
Real-time bar aggregators, mathematical indicators (Session-Anchored VWAP,
Wilder's ATR, Choppiness Index, ADX, EMA, RSI), REGIME_WAITING gate,
and VVR-BPC (Volatility-Normalized VWAP-ATR Regime Breakout & Pullback Continuation) strategy engine.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
import logging
import math
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    IST,
    SessionWindow,
    get_current_session_window,
    get_mcx_schedule,
)

logger = logging.getLogger("MCXTrader.Strategy")


class MarketRegime(str, Enum):
    REGIME_WAITING = "REGIME_WAITING"
    BULL_ACTIVE = "BULL_ACTIVE"
    BEAR_ACTIVE = "BEAR_ACTIVE"


class SignalType(str, Enum):
    LONG_BREAKOUT = "LONG_BREAKOUT"
    LONG_PULLBACK = "LONG_PULLBACK"
    SHORT_BREAKDOWN = "SHORT_BREAKDOWN"
    SHORT_PULLBACK = "SHORT_PULLBACK"


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = False

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, 1e-6)

    @property
    def body_ratio(self) -> float:
        return self.body / self.range

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    direction: str  # "BUY" or "SELL"
    entry_price: float
    signal_candle: Candle
    atr14: float
    vwap: float
    vwap_sd: float
    anchor_high: Optional[float]
    anchor_low: Optional[float]
    description: str


def compute_ema(series: List[float], span: int) -> List[float]:
    """Calculates Exponential Moving Average (EMA)."""
    if not series:
        return []
    alpha = 2.0 / (span + 1.0)
    emas = [series[0]]
    for val in series[1:]:
        emas.append(alpha * val + (1.0 - alpha) * emas[-1])
    return emas


def compute_rsi(prices: List[float], period: int = 14) -> float:
    """Calculates Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return 50.0

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent_changes = changes[-period:]

    gains = [max(c, 0.0) for c in recent_changes]
    losses = [abs(min(c, 0.0)) for c in recent_changes]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_wilder_atr(candles: List[Candle], period: int = 14) -> List[float]:
    """Calculates Wilder's Average True Range (ATR_14)."""
    if len(candles) < 2:
        return [c.range for c in candles]

    true_ranges = [candles[0].range]
    for i in range(1, len(candles)):
        curr = candles[i]
        prev_c = candles[i - 1].close
        tr = max(curr.high - curr.low, abs(curr.high - prev_c), abs(curr.low - prev_c))
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return [np.mean(true_ranges[: i + 1]) for i in range(len(true_ranges))]

    atrs = [float(np.mean(true_ranges[:period]))]
    alpha = 1.0 / period

    for i in range(period, len(true_ranges)):
        new_atr = (true_ranges[i] * alpha) + (atrs[-1] * (1.0 - alpha))
        atrs.append(new_atr)

    # Pad initial values so output aligns with candle indices
    prefix = [atrs[0]] * (period - 1)
    return prefix + atrs


def compute_choppiness_index(candles: List[Candle], period: int = 14) -> float:
    """
    Computes Choppiness Index (CI_14) on 15m candles:
    CI = 100 * LOG10(Sum(TR_14) / (MaxHigh_14 - MinLow_14)) / LOG10(14)
    CI > 52 indicates sideways chop. CI < 48 indicates directional trend.
    """
    if len(candles) < period + 1:
        return 50.0

    window = candles[-period:]
    true_ranges = []
    # Include prior close for TR calculation
    for i in range(len(window)):
        curr = window[i]
        # Prior candle
        idx_in_all = len(candles) - period + i
        prev_c = candles[idx_in_all - 1].close
        tr = max(curr.high - curr.low, abs(curr.high - prev_c), abs(curr.low - prev_c))
        true_ranges.append(tr)

    sum_tr = sum(true_ranges)
    max_h = max(c.high for c in window)
    min_l = min(c.low for c in window)
    high_low_span = max_h - min_l

    if high_low_span <= 1e-6 or sum_tr <= 1e-6:
        return 50.0

    ci = 100.0 * (math.log10(sum_tr / high_low_span) / math.log10(period))
    return min(max(ci, 0.0), 100.0)


def compute_adx(candles: List[Candle], period: int = 14) -> Tuple[float, List[float]]:
    """
    Computes Wilder's ADX_14 on candles.
    Returns (current_adx, adx_history).
    """
    if len(candles) < period * 2:
        return 20.0, [20.0] * len(candles)

    trs = []
    plus_dms = []
    minus_dms = []

    for i in range(1, len(candles)):
        curr = candles[i]
        prev = candles[i - 1]

        tr = max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        trs.append(tr)

        up_move = curr.high - prev.high
        down_move = prev.low - curr.low

        if up_move > down_move and up_move > 0:
            plus_dms.append(up_move)
        else:
            plus_dms.append(0.0)

        if down_move > up_move and down_move > 0:
            minus_dms.append(down_move)
        else:
            minus_dms.append(0.0)

    # Wilder's smoothing
    smooth_tr = [sum(trs[:period])]
    smooth_plus_dm = [sum(plus_dms[:period])]
    smooth_minus_dm = [sum(minus_dms[:period])]

    for i in range(period, len(trs)):
        smooth_tr.append(smooth_tr[-1] - (smooth_tr[-1] / period) + trs[i])
        smooth_plus_dm.append(smooth_plus_dm[-1] - (smooth_plus_dm[-1] / period) + plus_dms[i])
        smooth_minus_dm.append(smooth_minus_dm[-1] - (smooth_minus_dm[-1] / period) + minus_dms[i])

    dx_list = []
    for i in range(len(smooth_tr)):
        str_val = smooth_tr[i]
        if str_val <= 1e-6:
            dx_list.append(0.0)
            continue

        plus_di = 100.0 * (smooth_plus_dm[i] / str_val)
        minus_di = 100.0 * (smooth_minus_dm[i] / str_val)
        di_sum = plus_di + minus_di

        if di_sum <= 1e-6:
            dx = 0.0
        else:
            dx = 100.0 * (abs(plus_di - minus_di) / di_sum)
        dx_list.append(dx)

    if len(dx_list) < period:
        return 20.0, [20.0] * len(candles)

    adx_values = [sum(dx_list[:period]) / period]
    for i in range(period, len(dx_list)):
        new_adx = (adx_values[-1] * (period - 1) + dx_list[i]) / period
        adx_values.append(new_adx)

    pad_count = len(candles) - len(adx_values)
    full_adx = [adx_values[0]] * pad_count + adx_values
    return full_adx[-1], full_adx


class SessionVWAPTracker:
    """
    Session-Anchored VWAP and Standard Deviation Bands reset every morning at 09:00 IST.
    P_i = (H_i + L_i + C_i) / 3
    VWAP_t = sum(P_i * V_i) / sum(V_i)
    sigma_t = sqrt( sum(V_i * (P_i - VWAP_t)^2) / sum(V_i) )
    """

    def __init__(self):
        self.session_date: Optional[datetime.date] = None
        self.cum_pv: float = 0.0
        self.cum_vol: float = 0.0
        self.cum_pv2: float = 0.0
        self.vwap: float = 0.0
        self.sigma: float = 0.0
        self.points: List[Tuple[float, float]] = []  # (typical_price, volume)

    def reset_if_new_day(self, current_dt: datetime):
        c_date = current_dt.date()
        if self.session_date != c_date:
            self.session_date = c_date
            self.cum_pv = 0.0
            self.cum_vol = 0.0
            self.cum_pv2 = 0.0
            self.vwap = 0.0
            self.sigma = 0.0
            self.points.clear()
            logger.info(f"Session VWAP reset for trade date {c_date}")

    def update_bar(self, typical_price: float, volume: float):
        if volume <= 0:
            volume = 1.0  # Minimal weight if volume unavailable

        self.cum_pv += typical_price * volume
        self.cum_vol += volume
        self.points.append((typical_price, volume))

        if self.cum_vol > 0:
            self.vwap = self.cum_pv / self.cum_vol
            # Calculate standard deviation over all session bars
            variance_weighted = sum(v * ((p - self.vwap) ** 2) for p, v in self.points) / self.cum_vol
            self.sigma = math.sqrt(max(variance_weighted, 0.0))
        else:
            self.vwap = typical_price
            self.sigma = 0.0


class StrategyEngine:
    """
    Volatility-Normalized VWAP-ATR Regime Breakout & Pullback Continuation (VVR-BPC)
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.vwap_tracker = SessionVWAPTracker()

        # Candle aggregators
        self.candles_5m: List[Candle] = []
        self.candles_15m: List[Candle] = []

        # Current incomplete building bars
        self.curr_5m: Optional[Candle] = None
        self.curr_15m: Optional[Candle] = None

        # Anchor Ranges (09:00 - 09:45 ORB and Pre-COMEX ranges)
        self.morning_orb_high: Optional[float] = None
        self.morning_orb_low: Optional[float] = None
        self.pre_comex_high: Optional[float] = None
        self.pre_comex_low: Optional[float] = None

        # Tracking state for Pullback setups
        self.session_bull_breakout_confirmed: bool = False
        self.session_bear_breakdown_confirmed: bool = False

        # Regime state
        self.current_regime: MarketRegime = MarketRegime.REGIME_WAITING
        self.last_regime_reason: str = "Initializing"

    def feed_tick(self, timestamp: datetime, price: float, volume: float = 1.0) -> Optional[TradeSignal]:
        """
        Feeds real-time tick into bar aggregators.
        Returns TradeSignal if a 5m bar just closed and generated a verified entry signal.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)

        self.vwap_tracker.reset_if_new_day(timestamp)

        # Aggregate into 5-minute bar
        five_min_closed = self._aggregate_5m_bar(timestamp, price, volume)
        self._aggregate_15m_bar(timestamp, price, volume)

        # Update anchor ranges based on session time
        self._update_anchor_ranges(timestamp, price)

        if five_min_closed:
            closed_5m = self.candles_5m[-1]
            self.vwap_tracker.update_bar(closed_5m.typical_price, closed_5m.volume)
            # Evaluate regime and strategy triggers on completed 5m bar
            return self.evaluate_triggers(timestamp)

        return None

    def _aggregate_5m_bar(self, ts: datetime, price: float, volume: float) -> bool:
        """Aggregates tick into 5-minute candle. Returns True if a bar closed."""
        # Align to 5-minute boundary
        minute_bucket = (ts.minute // 5) * 5
        bar_start = ts.replace(minute=minute_bucket, second=0, microsecond=0)

        if self.curr_5m is None:
            self.curr_5m = Candle(bar_start, price, price, price, price, volume)
            return False

        if bar_start > self.curr_5m.timestamp:
            # Previous bar is completed
            self.curr_5m.is_closed = True
            self.candles_5m.append(self.curr_5m)
            # Start new bar
            self.curr_5m = Candle(bar_start, price, price, price, price, volume)
            return True
        else:
            # Update current bar
            self.curr_5m.high = max(self.curr_5m.high, price)
            self.curr_5m.low = min(self.curr_5m.low, price)
            self.curr_5m.close = price
            self.curr_5m.volume += volume
            return False

    def _aggregate_15m_bar(self, ts: datetime, price: float, volume: float) -> bool:
        """Aggregates tick into 15-minute candle. Returns True if a bar closed."""
        minute_bucket = (ts.minute // 15) * 15
        bar_start = ts.replace(minute=minute_bucket, second=0, microsecond=0)

        if self.curr_15m is None:
            self.curr_15m = Candle(bar_start, price, price, price, price, volume)
            return False

        if bar_start > self.curr_15m.timestamp:
            self.curr_15m.is_closed = True
            self.candles_15m.append(self.curr_15m)
            self.curr_15m = Candle(bar_start, price, price, price, price, volume)
            return True
        else:
            self.curr_15m.high = max(self.curr_15m.high, price)
            self.curr_15m.low = min(self.curr_15m.low, price)
            self.curr_15m.close = price
            self.curr_15m.volume += volume
            return False

    def _update_anchor_ranges(self, ts: datetime, price: float):
        """Maintains Morning ORB (09:00 - 09:45) and Pre-COMEX ranges."""
        schedule = get_mcx_schedule(ts.date())
        curr_t = ts.time()

        # Morning ORB 09:00 - 09:45
        if time(9, 0) <= curr_t <= time(9, 45):
            if self.morning_orb_high is None:
                self.morning_orb_high = price
                self.morning_orb_low = price
            else:
                self.morning_orb_high = max(self.morning_orb_high, price)
                self.morning_orb_low = min(self.morning_orb_low, price)

        # Pre-COMEX Range (16:30 to comex_prime_start: 17:45 summer / 18:45 winter)
        if time(16, 30) <= curr_t <= schedule.comex_prime_start:
            if self.pre_comex_high is None:
                self.pre_comex_high = price
                self.pre_comex_low = price
            else:
                self.pre_comex_high = max(self.pre_comex_high, price)
                self.pre_comex_low = min(self.pre_comex_low, price)

    def get_active_anchor(self, ts: datetime) -> Tuple[Optional[float], Optional[float]]:
        """Returns the most relevant anchor high and low for current session time."""
        schedule = get_mcx_schedule(ts.date())
        curr_t = ts.time()

        # In or after Pre-COMEX window, use Pre-COMEX range if established
        if curr_t >= schedule.comex_prime_start and self.pre_comex_high is not None:
            return self.pre_comex_high, self.pre_comex_low
        # Otherwise use Morning ORB range
        return self.morning_orb_high, self.morning_orb_low

    def check_regime_waiting_gate(self, now_dt: datetime) -> Tuple[bool, str]:
        """
        Evaluates State 0: The "Wait for Market Condition" Gate (REGIME_WAITING).
        Returns (is_blocked, reason).
        """
        if len(self.candles_5m) < 20 or len(self.candles_15m) < 14:
            return True, "Insufficient bar history (<20 5m or <14 15m bars)"

        # 1. Chop Gate: 15m CI_14 > 52 AND 15m ADX_14 < 20
        ci14 = compute_choppiness_index(self.candles_15m, period=14)
        curr_adx, adx_history = compute_adx(self.candles_15m, period=14)

        if ci14 > 52.0 and curr_adx < 20.0:
            return True, f"Chop Gate active: CI_14={ci14:.1f} (>52) and ADX_14={curr_adx:.1f} (<20)"

        # 2. VWAP Magnet Trap
        vwap = self.vwap_tracker.vwap
        sigma = self.vwap_tracker.sigma
        latest_c5 = self.candles_5m[-1]
        c_t = latest_c5.close

        c5_closes = [c.close for c in self.candles_5m]
        ema20_5m = compute_ema(c5_closes, 20)[-1]
        ema50_5m = compute_ema(c5_closes, 50)[-1]

        magnet_lower = vwap - 0.35 * sigma
        magnet_upper = vwap + 0.35 * sigma
        ema_separation = abs(ema20_5m - ema50_5m) / max(c_t, 1.0)

        if (magnet_lower <= c_t <= magnet_upper) and (ema_separation < 0.0008):
            return True, (
                f"VWAP Magnet Trap active: Price {c_t:.1f} within +/-0.35 sigma "
                f"[{magnet_lower:.1f}, {magnet_upper:.1f}] and EMAs intertwined (sep={ema_separation:.6f} < 0.0008)"
            )

        # 3. Volatility Compression / Dead Tape
        atrs_5m = compute_wilder_atr(self.candles_5m, period=14)
        curr_atr = atrs_5m[-1]
        atr_sma50 = float(np.mean(atrs_5m[-50:])) if len(atrs_5m) >= 50 else curr_atr

        volumes_5m = [c.volume for c in self.candles_5m]
        vol_sma20 = float(np.mean(volumes_5m[-20:])) if len(volumes_5m) >= 20 else 1.0

        if (curr_atr < 0.65 * atr_sma50) or (latest_c5.volume < 0.8 * vol_sma20):
            return True, (
                f"Volatility Compression/Dead Tape: ATR={curr_atr:.2f} (SMA50={atr_sma50:.2f}) "
                f"or Vol={latest_c5.volume:.0f} (SMA20={vol_sma20:.0f})"
            )

        # 4. Midday Chop Guard (11:30 - 13:30)
        session_window = get_current_session_window(now_dt)
        if session_window == SessionWindow.MIDDAY_CHOP_GUARD:
            if not (curr_adx > 28.0 and latest_c5.volume > 2.0 * vol_sma20):
                return True, (
                    f"Midday Chop Guard (11:30-13:30): ADX={curr_adx:.1f} (needs >28) "
                    f"or Vol={latest_c5.volume:.0f} (needs >2.0x SMA20={vol_sma20:.0f})"
                )

        return False, "Market conditions validated"

    def evaluate_triggers(self, now_dt: datetime) -> Optional[TradeSignal]:
        """
        Evaluates Bullish and Bearish entry rules on closed 5m candle.
        """
        if len(self.candles_5m) < 20 or len(self.candles_15m) < 14:
            return None

        # Check REGIME_WAITING Gate
        blocked, reason = self.check_regime_waiting_gate(now_dt)
        if blocked:
            self.current_regime = MarketRegime.REGIME_WAITING
            self.last_regime_reason = reason
            return None

        latest_5m = self.candles_5m[-1]
        prior_5m = self.candles_5m[-2] if len(self.candles_5m) >= 2 else latest_5m

        vwap = self.vwap_tracker.vwap
        sigma = self.vwap_tracker.sigma
        atrs_5m = compute_wilder_atr(self.candles_5m, period=14)
        curr_atr = atrs_5m[-1]

        # 15m indicators
        c15_closes = [c.close for c in self.candles_15m]
        ema20_15m = compute_ema(c15_closes, 20)[-1]
        ema50_15m = compute_ema(c15_closes, 50)[-1]
        curr_15m_close = c15_closes[-1]

        ci14 = compute_choppiness_index(self.candles_15m, period=14)
        curr_adx, adx_history = compute_adx(self.candles_15m, period=14)
        adx_rising_3 = (
            len(adx_history) >= 4
            and adx_history[-1] > adx_history[-2] > adx_history[-3] > adx_history[-4]
        )

        # 5m indicators
        c5_closes = [c.close for c in self.candles_5m]
        ema20_5m = compute_ema(c5_closes, 20)[-1]
        ema50_5m = compute_ema(c5_closes, 50)[-1]
        rsi14_5m = compute_rsi(c5_closes, period=14)
        vol_sma20 = float(np.mean([c.volume for c in self.candles_5m[-20:]]))

        # Track confirmation of breakout levels across session
        if latest_5m.high > vwap + 1.0 * sigma:
            self.session_bull_breakout_confirmed = True
        if latest_5m.low < vwap - 1.0 * sigma:
            self.session_bear_breakdown_confirmed = True

        anchor_high, anchor_low = self.get_active_anchor(now_dt)

        # --- 3.3 BULL_ACTIVE RULES ---
        bull_macro = (
            curr_15m_close > vwap
            and ema20_15m > ema50_15m
            and (curr_adx >= 20.0 or (adx_rising_3 and ci14 < 50.0))
        )

        if bull_macro:
            self.current_regime = MarketRegime.BULL_ACTIVE

            # Trigger A: Institutional Compression Breakout
            if anchor_high is not None:
                is_breakout = (latest_5m.close > anchor_high) and (prior_5m.close <= anchor_high)
                is_strong_body = latest_5m.is_bullish and (latest_5m.body_ratio >= 0.60)
                is_volume_conf = latest_5m.volume >= 1.4 * vol_sma20
                is_not_overextended = (latest_5m.close <= vwap + 2.0 * sigma) and (55.0 <= rsi14_5m <= 73.0)

                if is_breakout and is_strong_body and is_volume_conf and is_not_overextended:
                    logger.info(f"Triggered BULL Breakout at {latest_5m.close:.2f} above {anchor_high:.2f}")
                    return TradeSignal(
                        timestamp=latest_5m.timestamp,
                        symbol=self.symbol,
                        signal_type=SignalType.LONG_BREAKOUT,
                        direction="BUY",
                        entry_price=latest_5m.close,
                        signal_candle=latest_5m,
                        atr14=curr_atr,
                        vwap=vwap,
                        vwap_sd=sigma,
                        anchor_high=anchor_high,
                        anchor_low=anchor_low,
                        description=f"Institutional Compression Breakout above Anchor {anchor_high:.1f}",
                    )

            # Trigger B: First Institutional Pullback Continuation
            if self.session_bull_breakout_confirmed:
                pullback_vol_low = latest_5m.volume < vol_sma20
                support_upper = max(ema20_5m, vwap + 0.5 * sigma)
                support_lower = min(ema20_5m, vwap + 0.5 * sigma) - (0.5 * curr_atr)

                tested_support = (latest_5m.low <= support_upper) and (latest_5m.low >= support_lower)
                bullish_reversal = (
                    latest_5m.close > prior_5m.high
                    and latest_5m.is_bullish
                    and latest_5m.close > ema50_5m
                )

                if pullback_vol_low and tested_support and bullish_reversal:
                    logger.info(f"Triggered BULL Pullback Continuation at {latest_5m.close:.2f}")
                    return TradeSignal(
                        timestamp=latest_5m.timestamp,
                        symbol=self.symbol,
                        signal_type=SignalType.LONG_PULLBACK,
                        direction="BUY",
                        entry_price=latest_5m.close,
                        signal_candle=latest_5m,
                        atr14=curr_atr,
                        vwap=vwap,
                        vwap_sd=sigma,
                        anchor_high=anchor_high,
                        anchor_low=anchor_low,
                        description="First Bullish Pullback Continuation into EMA20/VWAP+0.5SD",
                    )

        # --- 3.4 BEAR_ACTIVE RULES ---
        bear_macro = (
            curr_15m_close < vwap
            and ema20_15m < ema50_15m
            and (curr_adx >= 20.0 or (adx_rising_3 and ci14 < 50.0))
        )

        if bear_macro:
            self.current_regime = MarketRegime.BEAR_ACTIVE

            # Trigger A: Institutional Breakdown
            if anchor_low is not None:
                is_breakdown = (latest_5m.close < anchor_low) and (prior_5m.close >= anchor_low)
                is_strong_body = latest_5m.is_bearish and (latest_5m.body_ratio >= 0.60)
                is_volume_conf = latest_5m.volume >= 1.4 * vol_sma20
                is_not_overextended = (latest_5m.close >= vwap - 2.0 * sigma) and (27.0 <= rsi14_5m <= 45.0)

                if is_breakdown and is_strong_body and is_volume_conf and is_not_overextended:
                    logger.info(f"Triggered BEAR Breakdown at {latest_5m.close:.2f} below {anchor_low:.2f}")
                    return TradeSignal(
                        timestamp=latest_5m.timestamp,
                        symbol=self.symbol,
                        signal_type=SignalType.SHORT_BREAKDOWN,
                        direction="SELL",
                        entry_price=latest_5m.close,
                        signal_candle=latest_5m,
                        atr14=curr_atr,
                        vwap=vwap,
                        vwap_sd=sigma,
                        anchor_high=anchor_high,
                        anchor_low=anchor_low,
                        description=f"Institutional Breakdown below Anchor {anchor_low:.1f}",
                    )

            # Trigger B: First Bearish Pullback Continuation
            if self.session_bear_breakdown_confirmed:
                pullback_vol_low = latest_5m.volume < vol_sma20
                resistance_upper = max(ema20_5m, vwap - 0.5 * sigma) + (0.5 * curr_atr)
                resistance_lower = min(ema20_5m, vwap - 0.5 * sigma)

                tested_resistance = (latest_5m.high >= resistance_lower) and (latest_5m.high <= resistance_upper)
                bearish_reversal = (
                    latest_5m.close < prior_5m.low
                    and latest_5m.is_bearish
                    and latest_5m.close < ema50_5m
                )

                if pullback_vol_low and tested_resistance and bearish_reversal:
                    logger.info(f"Triggered BEAR Pullback Continuation at {latest_5m.close:.2f}")
                    return TradeSignal(
                        timestamp=latest_5m.timestamp,
                        symbol=self.symbol,
                        signal_type=SignalType.SHORT_PULLBACK,
                        direction="SELL",
                        entry_price=latest_5m.close,
                        signal_candle=latest_5m,
                        atr14=curr_atr,
                        vwap=vwap,
                        vwap_sd=sigma,
                        anchor_high=anchor_high,
                        anchor_low=anchor_low,
                        description="First Bearish Pullback Continuation into EMA20/VWAP-0.5SD",
                    )

        return None

