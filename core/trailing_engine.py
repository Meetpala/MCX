"""
core/trailing_engine.py
-----------------------
4-Stage Monotonic Ratchet Trailing Stop-Loss Engine with Zerodha 25-modification
limit protection and tick rounding.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import math
from typing import Optional, Tuple

from config.settings import IST
from core.margin_and_sizer import compute_all_in_breakeven_offset

logger = logging.getLogger("MCXTrader.TrailingEngine")


class TrailingStage(int, Enum):
    STAGE_0_INITIAL_BREATHING = 0     # R_peak < 1.0R: Hold initial SL_0
    STAGE_1_COST_PLUS_BREAKEVEN = 1   # 1.0R <= R_peak < 1.5R: Ratchet to BE+
    STAGE_2_LITTLE_AHEAD_RATCHET = 2  # 1.5R <= R_peak < 3.0R: Step-Ratchet + 1.25x ATR cushion
    STAGE_3_PARABOLIC_CLIMAX = 3      # R_peak >= 3.0R or Climax conditions: Tight 0.65x ATR / 1-bar trail


def round_to_tick(price: float, tick_size: float = 1.0, round_up_for_buy: bool = False) -> float:
    """Rounds price to nearest valid tick."""
    if tick_size <= 0:
        return price
    ticks = price / tick_size
    if round_up_for_buy:
        rounded_ticks = math.ceil(ticks - 1e-9)
    else:
        rounded_ticks = round(ticks)
    return round(rounded_ticks * tick_size, 4)


@dataclass
class ActiveTradeState:
    trade_id: str
    symbol: str
    is_long: bool
    entry_price: float
    initial_sl: float
    current_sl: float
    unit_risk_1r: float
    breakeven_offset: float
    lot_multiplier: float
    tick_size: float
    peak_price: float
    current_stage: TrailingStage = TrailingStage.STAGE_0_INITIAL_BREATHING
    sl_order_id: Optional[str] = None
    mod_count: int = 0
    last_mod_time: Optional[datetime] = None
    history_sl: list = field(default_factory=list)

    @property
    def true_breakeven_price(self) -> float:
        if self.is_long:
            return round_to_tick(self.entry_price + self.breakeven_offset, self.tick_size)
        else:
            return round_to_tick(self.entry_price - self.breakeven_offset, self.tick_size)


class TrailingEngine:
    """
    Implements Section 5 of the specification:
    4-Stage Asymmetric Ratchet Trailing Engine with strict monotonicity,
    volatility cushion, and Zerodha modification protection.
    """

    def __init__(self, tick_size: float = 1.0):
        self.tick_size = tick_size

    def calculate_initial_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        atr14: float,
        signal_candle_low: float,
        signal_candle_high: float,
        max_atr_cap_multiplier: float = 2.2,
    ) -> Tuple[float, float, bool]:
        """
        Calculates SL_0 according to Section 5.1:
        Long: min(signal_candle.low - 2 ticks, entry_price - 1.5 * ATR_14)
              capped at max distance 2.2 * ATR_14.
        Short: max(signal_candle.high + 2 ticks, entry_price + 1.5 * ATR_14)
              capped at max distance 2.2 * ATR_14.

        Returns: (sl_0, unit_risk_1r, is_valid_rr)
        """
        min_buffer = 1.5 * atr14
        max_cap = max_atr_cap_multiplier * atr14

        if is_long:
            structural_sl = signal_candle_low - (2.0 * self.tick_size)
            atr_sl = entry_price - min_buffer
            candidate_sl = min(structural_sl, atr_sl)

            # Cap maximum distance to 2.2 * ATR_14
            max_risk_sl = entry_price - max_cap
            if candidate_sl < max_risk_sl:
                sl_0 = max_risk_sl
                is_valid = True
            else:
                sl_0 = candidate_sl
                is_valid = True
            sl_0 = round_to_tick(sl_0, self.tick_size)
            risk_1r = abs(entry_price - sl_0)
        else:
            structural_sl = signal_candle_high + (2.0 * self.tick_size)
            atr_sl = entry_price + min_buffer
            candidate_sl = max(structural_sl, atr_sl)

            # Cap maximum distance to 2.2 * ATR_14
            max_risk_sl = entry_price + max_cap
            if candidate_sl > max_risk_sl:
                sl_0 = max_risk_sl
                is_valid = True
            else:
                sl_0 = candidate_sl
                is_valid = True
            sl_0 = round_to_tick(sl_0, self.tick_size)
            risk_1r = abs(sl_0 - entry_price)

        if risk_1r < 2.0 * self.tick_size:
            risk_1r = 2.0 * self.tick_size
            sl_0 = entry_price - risk_1r if is_long else entry_price + risk_1r

        return sl_0, risk_1r, is_valid

    def create_trade_state(
        self,
        trade_id: str,
        symbol: str,
        is_long: bool,
        entry_price: float,
        initial_sl: float,
        unit_risk_1r: float,
        lot_multiplier: float,
        sl_order_id: Optional[str] = None,
    ) -> ActiveTradeState:
        """Initializes state tracking for a newly filled position."""
        be_offset = compute_all_in_breakeven_offset(
            entry_price=entry_price,
            lot_multiplier=lot_multiplier,
            tick_size=self.tick_size,
        )

        state = ActiveTradeState(
            trade_id=trade_id,
            symbol=symbol,
            is_long=is_long,
            entry_price=entry_price,
            initial_sl=initial_sl,
            current_sl=initial_sl,
            unit_risk_1r=unit_risk_1r,
            breakeven_offset=be_offset,
            lot_multiplier=lot_multiplier,
            tick_size=self.tick_size,
            peak_price=entry_price,
            current_stage=TrailingStage.STAGE_0_INITIAL_BREATHING,
            sl_order_id=sl_order_id,
            mod_count=0,
            last_mod_time=None,
            history_sl=[initial_sl],
        )
        return state

    def update_price_and_compute_trailing(
        self,
        trade: ActiveTradeState,
        current_price: float,
        atr14: float,
        vwap: float = 0.0,
        vwap_sd: float = 0.0,
        rsi_5m: float = 50.0,
        prior_5m_low: Optional[float] = None,
        prior_5m_high: Optional[float] = None,
    ) -> Tuple[float, TrailingStage, bool, str]:
        """
        Updates peak price and evaluates the 4-Stage Asymmetric Ratchet Trailing rules.

        Returns:
            (new_sl, new_stage, should_modify, reason)
        """
        # Update peak price
        if trade.is_long:
            trade.peak_price = max(trade.peak_price, current_price)
            r_peak = (trade.peak_price - trade.entry_price) / max(trade.unit_risk_1r, 1e-6)
        else:
            trade.peak_price = min(trade.peak_price, current_price)
            r_peak = (trade.entry_price - trade.peak_price) / max(trade.unit_risk_1r, 1e-6)

        r_peak = max(r_peak, 0.0)

        # Stage determination
        is_parabolic = (
            r_peak >= 3.0
            or (vwap_sd > 0 and trade.is_long and current_price >= vwap + 2.5 * vwap_sd)
            or (vwap_sd > 0 and not trade.is_long and current_price <= vwap - 2.5 * vwap_sd)
            or (trade.is_long and rsi_5m > 78.0)
            or (not trade.is_long and rsi_5m < 22.0)
        )

        candidate_sl = trade.current_sl
        new_stage = trade.current_stage

        if is_parabolic:
            new_stage = TrailingStage.STAGE_3_PARABOLIC_CLIMAX
            if trade.is_long:
                tight_atr = trade.peak_price - (0.65 * atr14)
                if prior_5m_low is not None:
                    candle_trail = prior_5m_low - (2.0 * trade.tick_size)
                    candidate_sl = max(candle_trail, tight_atr)
                else:
                    candidate_sl = tight_atr
            else:
                tight_atr = trade.peak_price + (0.65 * atr14)
                if prior_5m_high is not None:
                    candle_trail = prior_5m_high + (2.0 * trade.tick_size)
                    candidate_sl = min(candle_trail, tight_atr)
                else:
                    candidate_sl = tight_atr

        elif r_peak >= 1.5:
            new_stage = TrailingStage.STAGE_2_LITTLE_AHEAD_RATCHET
            # Formula:
            # Long: max(BE+ + 0.60 * (R_peak - 1.0) * 1R, P_peak - 1.25 * ATR_14)
            # Short: min(BE- - 0.60 * (R_peak - 1.0) * 1R, P_peak + 1.25 * ATR_14)
            incremental_r = r_peak - 1.0
            if trade.is_long:
                ratchet_floor = trade.true_breakeven_price + (0.60 * incremental_r * trade.unit_risk_1r)
                atr_cushion = trade.peak_price - (1.25 * atr14)
                candidate_sl = max(ratchet_floor, atr_cushion)
            else:
                ratchet_floor = trade.true_breakeven_price - (0.60 * incremental_r * trade.unit_risk_1r)
                atr_cushion = trade.peak_price + (1.25 * atr14)
                candidate_sl = min(ratchet_floor, atr_cushion)

        elif r_peak >= 1.0:
            new_stage = TrailingStage.STAGE_1_COST_PLUS_BREAKEVEN
            # Ratchet to BE+
            candidate_sl = trade.true_breakeven_price

        else:
            new_stage = TrailingStage.STAGE_0_INITIAL_BREATHING
            candidate_sl = trade.initial_sl

        # Strict Monotonicity Invariant:
        # Long: SL must NEVER decrease
        # Short: SL must NEVER increase
        if trade.is_long:
            new_sl = max(trade.current_sl, candidate_sl)
        else:
            new_sl = min(trade.current_sl, candidate_sl)

        new_sl = round_to_tick(new_sl, trade.tick_size)

        # Check Zerodha modification rules
        should_modify, reason = self.check_modification_eligibility(trade, new_sl, atr14)

        if should_modify:
            trade.current_sl = new_sl
            trade.current_stage = new_stage
            trade.history_sl.append(new_sl)

        return new_sl, new_stage, should_modify, reason

    def check_modification_eligibility(
        self,
        trade: ActiveTradeState,
        new_sl: float,
        atr14: float,
        now_dt: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Enforces Zerodha 25-modification limits:
        1. Minimum step distance: |SL_new - SL_current| >= max(0.20 * ATR_14, 5 ticks)
        2. At least 4 seconds elapsed since last modification
        3. Hard cap tracking: If mod_count == 22, flag need for cancel & replace
        """
        if now_dt is None:
            now_dt = datetime.now(IST)
        elif now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=IST)

        delta_sl = abs(new_sl - trade.current_sl)
        if delta_sl <= 1e-6:
            return False, "SL unchanged"

        # 1. Minimum step distance filter
        min_threshold = max(0.20 * atr14, 5.0 * trade.tick_size)
        if delta_sl < min_threshold:
            return False, f"Delta {delta_sl:.2f} below minimum threshold {min_threshold:.2f}"

        # 2. Rate limit: 4 seconds between modifications
        if trade.last_mod_time is not None:
            elapsed = (now_dt - trade.last_mod_time).total_seconds()
            if elapsed < 4.0:
                return False, f"Rate limited: only {elapsed:.1f}s elapsed (min 4.0s required)"

        # 3. 25-Modification limit check
        if trade.mod_count >= 22:
            return True, "MOD_COUNT_CRITICAL: Order reached 22 modifications; must cancel & replace"

        return True, "Modification valid"

    def record_modification(self, trade: ActiveTradeState, now_dt: Optional[datetime] = None):
        """Updates trade modification counters after kite.modify_order() succeeds."""
        if now_dt is None:
            now_dt = datetime.now(IST)
        elif now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=IST)

        trade.mod_count += 1
        trade.last_mod_time = now_dt
        logger.info(f"Recorded SL modification #{trade.mod_count} for order {trade.sl_order_id} -> {trade.current_sl}")

    def reset_modification_counter(self, trade: ActiveTradeState, new_sl_order_id: str):
        """Called when old SL order is cancelled and a new SL order is placed at mod_count=22."""
        logger.info(f"Resetting modification counter for fresh SL order {new_sl_order_id} (old={trade.sl_order_id})")
        trade.sl_order_id = new_sl_order_id
        trade.mod_count = 0
        trade.last_mod_time = datetime.now(IST)

