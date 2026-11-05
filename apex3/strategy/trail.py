"""APEX-3 Protective Stop and Monotonic Trailing Stop Manager.

Rules:
1. Initial Stop:
   - Long: structural swing low - (ATR_pad * ATR)
   - Short: structural swing high + (ATR_pad * ATR)
   - Rounded to instrument tick size in protective direction.
2. Favorable movement tracking:
   - R = abs(actual_average_entry - initial_stop) per share.
   - High-Water Mark (HWM) for longs, Low-Water Mark (LWM) for shorts.
3. Trail Activation:
   - Before +1R: Keep initial protective stop.
   - At +1R: Advance stop to lock breakeven + round-trip costs + profit_lock_rupees.
   - Beyond +1R: Highest valid (long) or lowest valid (short) of:
     * Current stop
     * Cost-adjusted profit floor
     * ATR trail from HWM/LWM (normal or tightened multiplier)
     * Last confirmed 1-minute structural swing point with pad
4. Monotonicity:
   - Long stop NEVER moves down.
   - Short stop NEVER moves up.
5. Modification Budget:
   - Max 25 modifications per order (Kite limit).
   - Only propose modification if improvement >= min_step_ticks * tick_size.
   - Stop modifying if modification_count >= (max_modifications - modification_headroom).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Optional

from apex3.config import StopsConfig, CostsConfig
from apex3.indicators.swing import find_swing_highs, find_swing_lows
from apex3.models import Candle, PositionState, Side, get_mcx_lot_multiplier
from apex3.strategy.sizer import calculate_mcx_intraday_charges


@dataclass(frozen=True)
class StopEvaluation:
    """Result of stop calculation or trail update."""
    should_modify: bool
    proposed_stop: float
    current_stop: float
    trigger_price: float
    limit_price: float
    r_multiple: float
    best_favorable_price: float
    trail_activated: bool
    modification_count: int
    reason: str
    is_exhaustion: bool = False
    is_late_session: bool = False
    hwm_price: float = 0.0
    hwm_r: float = 0.0
    ratchet_stage: int = 0


TICK_SIZE = 0.05


def floor_to_tick(price: float, tick: float = TICK_SIZE, tick_size: Optional[float] = None) -> float:
    t = tick_size if (tick_size is not None and tick_size > 0) else (tick if tick > 0 else TICK_SIZE)
    decimals = 4 if t < 0.01 else 2
    return round(math.floor(price / t + 1e-9) * t, decimals)


def ceil_to_tick(price: float, tick: float = TICK_SIZE, tick_size: Optional[float] = None) -> float:
    t = tick_size if (tick_size is not None and tick_size > 0) else (tick if tick > 0 else TICK_SIZE)
    decimals = 4 if t < 0.01 else 2
    return round(math.ceil(price / t - 1e-9) * t, decimals)


def round_to_tick(price: float, tick: float = TICK_SIZE, tick_size: Optional[float] = None, round_up: Optional[bool] = None) -> float:
    """Round price to the nearest instrument tick size."""
    t = tick_size if (tick_size is not None and tick_size > 0) else tick
    if t <= 0:
        return price
    decimals = 4 if t < 0.01 else 2
    ticks = price / t
    if round_up is True:
        return round(math.ceil(ticks - 1e-9) * t, decimals)
    elif round_up is False:
        return round(math.floor(ticks + 1e-9) * t, decimals)
    return round(round(ticks) * t, decimals)


@dataclass
class StopTargetLevels:
    side: str               # "LONG" or "SHORT"
    entry_price: float
    initial_stop: float
    current_stop: float
    target_price: float
    risk_per_share: float   # R = |Entry - Initial Stop|
    risk_pct: float         # R / Entry Price
    fee_lock_applied: bool = False
    tick_size: float = TICK_SIZE


def compute_initial_stop_and_target(
    side: str,
    entry_price: float,
    swing_low_5: float,
    swing_high_5: float,
    atr14: float,
    cfg: dict,
    tick_size: float = TICK_SIZE,
) -> StopTargetLevels:
    """Computes Structural Stop (Swing Low/High +/- 1.5x ATR) and Fixed 2.0R Target."""
    st_cfg = cfg.get("stops_and_targets", cfg.get("stops", {})) if isinstance(cfg, dict) else {}
    atr_mult = float(st_cfg.get("atr_stop_multiplier", st_cfg.get("initial_atr_multiplier", 1.5)))
    target_r = float(st_cfg.get("target_r_multiple", st_cfg.get("target_r", 2.0)))
    t = tick_size if (tick_size and tick_size > 0) else TICK_SIZE

    side_str = side.value if hasattr(side, "value") else str(side).upper()
    if side_str in ("LONG", "BUY"):
        initial_stop = floor_to_tick(swing_low_5 - (atr_mult * atr14), tick=t)
        risk_r = round(entry_price - initial_stop, 4)
        target_price = round_to_tick(entry_price + (target_r * risk_r), tick=t)
    elif side_str in ("SHORT", "SELL"):
        initial_stop = ceil_to_tick(swing_high_5 + (atr_mult * atr14), tick=t)
        risk_r = round(initial_stop - entry_price, 4)
        target_price = round_to_tick(entry_price - (target_r * risk_r), tick=t)
    else:
        raise ValueError(f"Unsupported side: {side}")

    risk_pct = (risk_r / entry_price) if entry_price > 0 else 0.0

    return StopTargetLevels(
        side=side_str,
        entry_price=entry_price,
        initial_stop=initial_stop,
        current_stop=initial_stop,
        target_price=target_price,
        risk_per_share=risk_r,
        risk_pct=risk_pct,
        fee_lock_applied=False,
        tick_size=t,
    )


def evaluate_fee_protected_stop(
    levels: StopTargetLevels,
    current_price: float,
    current_time_str: str,
    cfg: dict,
    tick_size: Optional[float] = None,
) -> Optional[float]:
    """Continuous trailing is disabled (trail_enabled: false) so winners are not choked.
    However, a single one-time step is triggered if:
      1. Unrealized profit >= +1.60R (locks +0.60R profit-preserving buffer), OR
      2. Current time >= 14:15 IST and unrealized profit >= +1.15R (locks +0.35R late-day buffer).
    Returns the new stop price if updated, otherwise None.
    """
    st_cfg = cfg.get("stops_and_targets", cfg.get("stops", {})) if isinstance(cfg, dict) else {}
    if not st_cfg.get("fee_lock_enabled", True) or levels.fee_lock_applied:
        return None

    if levels.risk_per_share <= 0:
        return None

    t = tick_size if (tick_size is not None and tick_size > 0) else getattr(levels, "tick_size", TICK_SIZE)
    side_str = levels.side.upper()
    if side_str in ("LONG", "BUY"):
        unrealized_r = (current_price - levels.entry_price) / levels.risk_per_share
    else:
        unrealized_r = (levels.entry_price - current_price) / levels.risk_per_share

    trigger_r = float(st_cfg.get("fee_lock_trigger_r", 1.60))
    sched_cfg = cfg.get("session_schedule", cfg.get("schedule", {})) if isinstance(cfg, dict) else {}
    late_time = sched_cfg.get("late_day_protection_time", "14:15")
    late_trigger_r = float(st_cfg.get("late_day_lock_trigger_r", 1.15))
    offset_pct = float(st_cfg.get("fee_lock_offset_pct", 0.0015))
    profit_r = float(st_cfg.get("fee_lock_profit_r", 0.60))
    late_profit_r = float(st_cfg.get("late_day_lock_profit_r", 0.35))

    target_lock_r = None
    if unrealized_r >= trigger_r:
        target_lock_r = profit_r
    elif current_time_str >= late_time and unrealized_r >= late_trigger_r:
        target_lock_r = late_profit_r

    if target_lock_r is None:
        return None

    if side_str in ("LONG", "BUY"):
        candidate = max(
            levels.entry_price * (1.0 + offset_pct),
            levels.entry_price + (target_lock_r * levels.risk_per_share),
        )
        protected_stop = ceil_to_tick(candidate, tick=t)
        if protected_stop > levels.current_stop and protected_stop < current_price:
            levels.current_stop = protected_stop
            levels.fee_lock_applied = True
            return protected_stop
    else:
        candidate = min(
            levels.entry_price * (1.0 - offset_pct),
            levels.entry_price - (target_lock_r * levels.risk_per_share),
        )
        protected_stop = floor_to_tick(candidate, tick=t)
        if protected_stop < levels.current_stop and protected_stop > current_price:
            levels.current_stop = protected_stop
            levels.fee_lock_applied = True
            return protected_stop

    return None


def evaluate_progressive_ratchet(
    pos: PositionState,
    current_price: float,
    candles_5m: list[Candle],
    atr_14: float,
    config: Any = None,
    tick_size: float = 1.0,
    macro_state: Optional[Any] = None,
) -> StopEvaluation:
    """Evaluate 5-Stage Progressive Ratchet Trailing Stop for MCX Precious Metals.

    Stages:
    - Stage 0 (< 1.0R): Maintain initial structural stop SL_0.
    - Stage 1 (>= 1.0R): Move to BE+ (Fill +/- max(Charges/(Q*M) + 3 ticks, 0.18R)).
    - Stage 2 (1.4R to 2.0R): +0.65R lock at 1.40R; steps to +0.95R lock at 1.75R.
    - Stage 3 (2.0R to 3.2R): Multi-R Trend Ratchet: max(Fill + (HWM_R - 0.65R), SwingLow_3 - 0.35*ATR_14).
    - Stage 4 (>= 3.2R or Climax): Parabolic Harvest: max(Fill + (HWM_R - 0.35R), L_prev - 2 ticks, HWM - 0.55*ATR_14).

    Strict monotonicity: BUY stop never retreats, SELL stop never advances.
    Minimum modification step: max(3 ticks, 0.12R).
    Modification budget cap: 20 modifications per order.
    """
    entry_price = pos.average_entry
    initial_stop = pos.initial_stop_price if pos.initial_stop_price > 0 else pos.protective_stop_price
    current_stop = pos.protective_stop_price if pos.protective_stop_price > 0 else initial_stop
    mod_count = getattr(pos, "modification_count", 0) or getattr(pos, "stop_modification_count", 0)

    # Resolve ratchet configuration
    if hasattr(config, "ratchet"):
        ratchet_cfg = config.ratchet
    elif isinstance(config, dict) and "ratchet" in config:
        ratchet_cfg = config["ratchet"]
    elif config is not None:
        ratchet_cfg = getattr(config, "stops", config)
    else:
        ratchet_cfg = None

    max_mods = getattr(ratchet_cfg, "max_stop_modifications", 20) if ratchet_cfg else 20
    t_size = tick_size if (tick_size and tick_size > 0) else getattr(pos, "tick_size", 1.0)
    if t_size <= 0:
        t_size = 1.0

    if entry_price <= 0 or initial_stop <= 0:
        return StopEvaluation(
            should_modify=False,
            proposed_stop=current_stop,
            current_stop=current_stop,
            trigger_price=current_stop,
            limit_price=current_stop,
            r_multiple=0.0,
            best_favorable_price=current_price,
            trail_activated=False,
            modification_count=mod_count,
            reason="Invalid position entry or initial stop",
            hwm_price=current_price,
            hwm_r=0.0,
            ratchet_stage=0,
        )

    is_long = (pos.side == Side.BUY)
    unit_risk_1r = abs(entry_price - initial_stop)
    if unit_risk_1r <= 0:
        unit_risk_1r = max(1.0, 2.0 * t_size)

    # 1. Update High-Water Mark / Best Favorable Price
    existing_hwm_price = getattr(pos, "hwm_price", 0.0) or pos.best_favorable_price
    if is_long:
        hwm_price = max(existing_hwm_price, current_price, entry_price)
        favorable_excursion = hwm_price - entry_price
    else:
        if existing_hwm_price > 0:
            hwm_price = min(existing_hwm_price, current_price, entry_price)
        else:
            hwm_price = min(current_price, entry_price)
        favorable_excursion = entry_price - hwm_price

    hwm_r = favorable_excursion / unit_risk_1r if unit_risk_1r > 0 else 0.0

    # 2. Stage Evaluation
    candidate_sl = current_stop
    stage_name = "Stage 0 (Structural Breathing Room)"
    ratchet_stage = 0
    trail_activated = False

    stage1_buffer_ticks = getattr(ratchet_cfg, "stage1_buffer_ticks", 2) if ratchet_cfg else 2
    min_be_r = getattr(ratchet_cfg, "stage1_min_be_r", 0.0) if ratchet_cfg else 0.0

    lot_mult = getattr(pos, "lot_multiplier", 1) or get_mcx_lot_multiplier(pos.symbol)
    qty = max(1, getattr(pos, "quantity", 1))
    charges_all = calculate_mcx_intraday_charges(entry_price, entry_price, qty, lot_mult)
    charge_offset_pts = math.ceil((charges_all / (qty * lot_mult)) / t_size) * t_size
    be_offset = max(charge_offset_pts + (stage1_buffer_ticks * t_size), min_be_r * unit_risk_1r)

    completed = [c for c in candles_5m if getattr(c, "is_complete", True)]
    prior_c = completed[-2] if len(completed) >= 2 else (completed[-1] if completed else None)

    # Stage 4 (Parabolic / Climax Harvest):
    # HWM_R >= 3.20R or Climax
    if hwm_r >= 3.20:
        ratchet_stage = 4
        stage_name = "Stage 4 (Parabolic / Climax Harvest)"
        trail_activated = True
        if is_long:
            ratchet_floor = entry_price + ((hwm_r - 0.35) * unit_risk_1r)
            prior_low_trail = (prior_c.low - 2.0 * t_size) if prior_c else ratchet_floor
            atr_trail = hwm_price - (0.55 * atr_14)
            candidate_sl = ceil_to_tick(max(ratchet_floor, prior_low_trail, atr_trail), tick=t_size)
        else:
            ratchet_floor = entry_price - ((hwm_r - 0.35) * unit_risk_1r)
            prior_high_trail = (prior_c.high + 2.0 * t_size) if prior_c else ratchet_floor
            atr_trail = hwm_price + (0.55 * atr_14)
            candidate_sl = floor_to_tick(min(ratchet_floor, prior_high_trail, atr_trail), tick=t_size)

    elif hwm_r >= 2.00:
        ratchet_stage = 3
        stage_name = "Stage 3 (Multi-R Trend Ratchet)"
        trail_activated = True
        if is_long:
            ratchet_floor = entry_price + ((hwm_r - 0.65) * unit_risk_1r)
            if len(completed) >= 3:
                swing_low_3 = min(c.low for c in completed[-3:])
                swing_trail = swing_low_3 - (0.35 * atr_14)
            else:
                swing_trail = ratchet_floor
            candidate_sl = ceil_to_tick(max(ratchet_floor, swing_trail), tick=t_size)
        else:
            ratchet_floor = entry_price - ((hwm_r - 0.65) * unit_risk_1r)
            if len(completed) >= 3:
                swing_high_3 = max(c.high for c in completed[-3:])
                swing_trail = swing_high_3 + (0.35 * atr_14)
            else:
                swing_trail = ratchet_floor
            candidate_sl = floor_to_tick(min(ratchet_floor, swing_trail), tick=t_size)

    elif hwm_r >= 1.40:
        ratchet_stage = 2
        stage_name = "Stage 2 ('Little-Ahead' Profit Cushion)"
        trail_activated = True
        lock_r = 0.95 if hwm_r >= 1.75 else 0.65
        if is_long:
            candidate_sl = ceil_to_tick(entry_price + (lock_r * unit_risk_1r), tick=t_size)
        else:
            candidate_sl = floor_to_tick(entry_price - (lock_r * unit_risk_1r), tick=t_size)

    elif hwm_r >= 1.00:
        ratchet_stage = 1
        stage_name = "Stage 1 (Breakeven+)"
        trail_activated = True
        if is_long:
            candidate_sl = ceil_to_tick(entry_price + be_offset, tick=t_size)
        else:
            candidate_sl = floor_to_tick(entry_price - be_offset, tick=t_size)

    else:
        ratchet_stage = 0
        stage_name = "Stage 0 (Structural Breathing Room)"
        candidate_sl = initial_stop

    # Pre-Event Shield & Opposing Breaking Shock Tightener
    current_favorable = (current_price - entry_price) if is_long else (entry_price - current_price)
    current_r = current_favorable / unit_risk_1r if unit_risk_1r > 0 else 0.0

    news_cfg = getattr(config, "news", None) if config else None
    if isinstance(config, dict) and "news" in config:
        news_cfg = config["news"]
    if macro_state is None and isinstance(config, dict) and "macro_state" in config:
        macro_state = config["macro_state"]

    be_threshold_r = float(getattr(news_cfg, "pre_event_lock_be_threshold_r", 0.75) if news_cfg else 0.75)
    opposing_thresh = float(getattr(news_cfg, "opposing_shock_threshold", 0.75) if news_cfg else 0.75)
    be_plus = ceil_to_tick(entry_price + be_offset, tick=t_size) if is_long else floor_to_tick(entry_price - be_offset, tick=t_size)

    if macro_state is not None:
        mins = getattr(macro_state, "minutes_to_next_high_event", None)
        if mins is not None and 0.0 < mins <= 5.0 and current_r >= be_threshold_r:
            if is_long:
                candidate_sl = max(candidate_sl, be_plus)
            else:
                candidate_sl = min(candidate_sl, be_plus)
            stage_name = "Pre-Event Shield (BE+)"
            trail_activated = True

        m_score = getattr(macro_state, "macro_score", 0.0)
        if current_r >= 0.50:
            if is_long and m_score <= -opposing_thresh:
                shock_sl = floor_to_tick(current_price - (0.5 * atr_14), tick=t_size)
                candidate_sl = max(candidate_sl, be_plus, shock_sl)
                stage_name = "Opposing Shock Tightener"
                trail_activated = True
            elif not is_long and m_score >= opposing_thresh:
                shock_sl = ceil_to_tick(current_price + (0.5 * atr_14), tick=t_size)
                candidate_sl = min(candidate_sl, be_plus, shock_sl)
                stage_name = "Opposing Shock Tightener"
                trail_activated = True

    # 3. Enforce Strict Monotonicity Invariant
    if is_long:
        best_sl = max(current_stop, candidate_sl)
        market_cap = current_price - (2.0 * t_size)
        best_sl = min(best_sl, market_cap) if current_price > market_cap else best_sl
        final_stop = floor_to_tick(best_sl, tick=t_size)
        improvement = final_stop - current_stop
    else:
        best_sl = min(current_stop, candidate_sl)
        market_cap = current_price + (2.0 * t_size)
        best_sl = max(best_sl, market_cap) if current_price < market_cap else best_sl
        final_stop = ceil_to_tick(best_sl, tick=t_size)
        improvement = current_stop - final_stop

    # 4. Modification step size: max(3 ticks, 0.12R)
    min_step = max(3.0 * t_size, 0.12 * unit_risk_1r)
    is_improvement_sufficient = improvement >= min_step

    if mod_count >= max_mods:
        should_modify = False
        reason = f"Max modification cap ({max_mods}) reached; stop unchanged at {current_stop}"
    elif is_improvement_sufficient:
        should_modify = True
        reason = f"Ratchet update ({stage_name}): {current_stop} -> {final_stop} (HWM_R={hwm_r:.2f}R)"
    else:
        should_modify = False
        reason = f"Improvement {improvement:.2f} < min step {min_step:.2f}; stop unchanged"

    if not should_modify:
        final_stop = current_stop

    # Compute trigger and limit
    trigger, limit = calculate_stop_limit_prices(
        pos.side,
        final_stop,
        config if hasattr(config, "sl_limit_buffer_pct") else StopsConfig(),
        t_size,
    )

    return StopEvaluation(
        should_modify=should_modify,
        proposed_stop=final_stop,
        current_stop=current_stop,
        trigger_price=trigger,
        limit_price=limit,
        r_multiple=hwm_r,
        best_favorable_price=hwm_price,
        trail_activated=trail_activated,
        modification_count=mod_count,
        reason=reason,
        hwm_price=hwm_price,
        hwm_r=hwm_r,
        ratchet_stage=ratchet_stage,
    )


def calculate_mcx_initial_stop(
    side: Union[Side, str],
    entry_price: float,
    swing_low_5: float,
    swing_high_5: float,
    atr14: float,
    signal_high: Optional[float] = None,
    signal_low: Optional[float] = None,
    tick_size: float = 1.0,
) -> tuple[float, float, float]:
    """Calculate structural initial stop for MCX Bullion Futures with sweet-spot validation.

    Formula:
    - Long (BUY): floor_to_tick(min(swing_low_5, signal_low) - 1.35 * atr14, tick)
    - Short (SELL): ceil_to_tick(max(swing_high_5, signal_high) + 1.35 * atr14, tick)
    - Sweet spot gate: 0.18% <= stop_width_pct <= 0.65%

    Returns:
        (initial_stop, risk_per_share_1r, stop_width_pct)
    """
    side_str = side.value if hasattr(side, "value") else str(side).upper()
    is_buy = side_str in ("BUY", "LONG")

    s_low = min(swing_low_5, signal_low) if signal_low is not None else swing_low_5
    s_high = max(swing_high_5, signal_high) if signal_high is not None else swing_high_5

    if is_buy:
        raw_stop = s_low - (1.35 * atr14)
        if raw_stop >= entry_price:
            raw_stop = entry_price - (atr14 * 1.0)
        initial_stop = floor_to_tick(raw_stop, tick=tick_size)
        risk_per_share = entry_price - initial_stop
    else:
        raw_stop = s_high + (1.35 * atr14)
        if raw_stop <= entry_price:
            raw_stop = entry_price + (atr14 * 1.0)
        initial_stop = ceil_to_tick(raw_stop, tick=tick_size)
        risk_per_share = initial_stop - entry_price

    stop_width_pct = (risk_per_share / entry_price) if entry_price > 0 else 0.0
    return initial_stop, risk_per_share, stop_width_pct


def calculate_initial_stop(
    side: Union[Side, str],
    entry_price: float,
    signal_low: float,
    signal_high: float,
    atr_value: float,
    config: Optional[StopsConfig] = None,
    tick_size: float = 0.05,
) -> float:
    """Calculate initial protective stop beyond the signal candle / swing with ATR pad.

    Args:
        side: BUY or SELL (enum or str)
        entry_price: Expected or fill entry price
        signal_low: Low of the signal bar / swing low
        signal_high: High of the signal bar / swing high
        atr_value: 5-minute ATR value
        config: StopsConfig
        tick_size: Instrument tick size

    Returns:
        Initial stop price rounded in protective direction.
    """
    atr_mult = getattr(config, "initial_atr_multiplier", 1.35) if config is not None else 1.35
    pad = atr_mult * atr_value

    side_str = side.value if hasattr(side, "value") else str(side).upper()
    if side_str in ("BUY", "LONG"):
        # Stop below the signal low
        raw_stop = signal_low - pad
        # Must be below entry price
        if raw_stop >= entry_price:
            raw_stop = entry_price - (atr_value * 1.0)
        # Round down (more protective for long)
        return round_to_tick(raw_stop, tick_size=tick_size, round_up=False)
    else:  # SELL / SHORT
        # Stop above the signal high
        raw_stop = signal_high + pad
        # Must be above entry price
        if raw_stop <= entry_price:
            raw_stop = entry_price + (atr_value * 1.0)
        # Round up (more protective for short)
        return round_to_tick(raw_stop, tick_size=tick_size, round_up=True)


def calculate_stop_limit_prices(
    side: Side,
    stop_trigger: float,
    config: StopsConfig,
    tick_size: float = 0.05,
    lower_circuit: float = 0.0,
    upper_circuit: float = 0.0,
) -> tuple[float, float]:
    """Calculate trigger price and limit price for a Stop-Loss Limit (SL) order.

    A stop-limit order requires:
    - Long exit (SELL SL): Limit price < Trigger price (more marketable), clamped >= lower_circuit.
    - Short exit (BUY SL): Limit price > Trigger price (more marketable), clamped <= upper_circuit.

    WARNING: Even with a marketable buffer, SL limit orders can remain unfilled
    during circuits or rapid price gapping.

    Returns:
        (trigger_price, limit_price)
    """
    t = tick_size if (tick_size and tick_size > 0) else TICK_SIZE
    buffer_multiplier = config.sl_limit_buffer_pct / 100.0

    if side == Side.BUY:
        # Position is LONG -> exit order is SELL
        trigger = round_to_tick(stop_trigger, tick_size=t, round_up=False)
        raw_limit = trigger * (1.0 - buffer_multiplier)
        limit = round_to_tick(raw_limit, tick_size=t, round_up=False)
        if lower_circuit > 0.0:
            min_limit = ceil_to_tick(lower_circuit, tick_size=t)
            if limit < min_limit:
                limit = min_limit
            if trigger <= limit:
                trigger = round_to_tick(limit + t, tick_size=t, round_up=True)
        return trigger, limit
    else:  # Side.SELL
        # Position is SHORT -> exit order is BUY
        trigger = round_to_tick(stop_trigger, tick_size=t, round_up=True)
        raw_limit = trigger * (1.0 + buffer_multiplier)
        limit = round_to_tick(raw_limit, tick_size=t, round_up=True)
        if upper_circuit > 0.0:
            max_limit = floor_to_tick(upper_circuit, tick_size=t)
            if limit > max_limit:
                limit = max_limit
            if trigger >= limit:
                trigger = round_to_tick(limit - t, tick_size=t, round_up=False)
        return trigger, limit


def evaluate_trailing_stop(
    position: PositionState,
    current_price: float,
    atr_5m: float,
    candles_1m: list[Candle],
    config: StopsConfig,
    costs_config: CostsConfig,
    current_ist_time: time,
    adx_5m: Optional[float] = None,
    tick_size: float = 0.05,
) -> StopEvaluation:
    """Evaluate candidate trailing stop and enforce strict monotonicity.

    Args:
        position: Current PositionState
        current_price: Latest traded price
        atr_5m: Current 5m ATR
        candles_1m: Completed 1m candles for swing structure detection
        config: StopsConfig with trail parameters
        costs_config: CostsConfig to estimate round-trip costs
        current_ist_time: Current time in IST
        adx_5m: Optional 5m ADX for trend exhaustion detection
        tick_size: Instrument tick size

    Returns:
        StopEvaluation indicating whether to modify and the exact proposed prices.
    """
    trail_cfg = config.trail
    current_stop = position.protective_stop_price
    initial_stop = position.initial_stop_price
    entry_price = position.average_entry
    mod_count = position.best_favorable_price  # We will track mod count on position

    # Basic validity check
    if position.quantity <= 0 or entry_price <= 0 or current_stop <= 0:
        return StopEvaluation(
            should_modify=False,
            proposed_stop=current_stop,
            current_stop=current_stop,
            trigger_price=current_stop,
            limit_price=current_stop,
            r_multiple=0.0,
            best_favorable_price=current_price,
            trail_activated=False,
            modification_count=0,
            reason="Position not active or invalid entry/stop",
        )

    # 1. Update High-Water Mark / Low-Water Mark
    if position.side == Side.BUY:
        hwm = max(position.best_favorable_price, current_price, entry_price)
        best_favorable = hwm
        favorable_excursion = best_favorable - entry_price
    else:
        lwm = (
            min(position.best_favorable_price, current_price, entry_price)
            if position.best_favorable_price > 0
            else min(current_price, entry_price)
        )
        best_favorable = lwm
        favorable_excursion = entry_price - best_favorable

    # 2. R value and R multiple
    r_val = abs(entry_price - initial_stop) if initial_stop > 0 else (entry_price * 0.01)
    current_r = favorable_excursion / r_val if r_val > 0 else 0.0

    # 3. Check Trail Activation (+1R threshold)
    trail_activated = position.trail_activated or (current_r >= trail_cfg.activation_r)

    if not trail_activated:
        # Before +1R: Keep initial stop
        trigger, limit = calculate_stop_limit_prices(position.side, current_stop, config, tick_size)
        return StopEvaluation(
            should_modify=False,
            proposed_stop=current_stop,
            current_stop=current_stop,
            trigger_price=trigger,
            limit_price=limit,
            r_multiple=current_r,
            best_favorable_price=best_favorable,
            trail_activated=False,
            modification_count=0,
            reason=f"Favorable movement ({current_r:.2f}R) < activation threshold ({trail_cfg.activation_r:.1f}R)",
        )

    # 4. Trail is Active: Determine Tightening Rules
    # Late session check
    late_parts = trail_cfg.late_session_tighten_time.split(":")
    late_time = time(int(late_parts[0]), int(late_parts[1]))
    is_late_session = current_ist_time >= late_time

    # Trend exhaustion check (ADX < threshold)
    is_exhaustion = (adx_5m is not None) and (adx_5m < trail_cfg.exhaustion_adx_threshold)

    # Select ATR multiplier
    if is_late_session or is_exhaustion:
        atr_mult = trail_cfg.atr_multiplier_tight
    else:
        atr_mult = trail_cfg.atr_multiplier_normal

    # 5. Round-trip Cost and Profit Lock Floor
    # Estimate total round-trip charges per share
    est_brokerage_per_share = (costs_config.brokerage_per_order * 2.0) / max(1, position.quantity)
    est_taxes_per_share = entry_price * ((costs_config.stt_pct + costs_config.exchange_txn_pct) / 100.0)
    round_trip_cost_per_share = est_brokerage_per_share + est_taxes_per_share
    profit_lock = trail_cfg.profit_lock_rupees

    # 6. Candidate Stop Levels
    candidate_stops: list[tuple[float, str]] = []

    # Level A: Current existing stop (base for monotonicity)
    candidate_stops.append((current_stop, "Existing stop"))

    # Level B: Cost-adjusted profit lock floor
    if position.side == Side.BUY:
        floor_level = entry_price + round_trip_cost_per_share + profit_lock
        candidate_stops.append((floor_level, "Cost+Profit floor"))
    else:
        floor_level = entry_price - (round_trip_cost_per_share + profit_lock)
        candidate_stops.append((floor_level, "Cost+Profit floor"))

    # Level C: ATR Trail from HWM/LWM
    if position.side == Side.BUY:
        atr_trail = best_favorable - (atr_mult * atr_5m)
        candidate_stops.append((atr_trail, f"ATR trail ({atr_mult}x)"))
    else:
        atr_trail = best_favorable + (atr_mult * atr_5m)
        candidate_stops.append((atr_trail, f"ATR trail ({atr_mult}x)"))

    # Level D: 1-minute Structural Swing Low/High
    completed_1m = [c for c in candles_1m if c.is_complete]
    if len(completed_1m) >= 7:
        pad_rupees = trail_cfg.swing_pad_ticks * tick_size
        if position.side == Side.BUY:
            lows = [c.low for c in completed_1m]
            swings = find_swing_lows(lows, lookback=3)
            if swings:
                latest_swing_low = swings[-1][1]
                candidate_stops.append((latest_swing_low - pad_rupees, "1m Swing Low"))
        else:
            highs = [c.high for c in completed_1m]
            swings = find_swing_highs(highs, lookback=3)
            if swings:
                latest_swing_high = swings[-1][1]
                candidate_stops.append((latest_swing_high + pad_rupees, "1m Swing High"))

    # 7. Select Most Protective Level & Enforce Monotonicity
    if position.side == Side.BUY:
        # Long position: higher stop is more protective. Monotonic: candidate >= current_stop
        best_candidate, best_reason = max(candidate_stops, key=lambda x: x[0])
        # Never allow stop to exceed current market price (must remain protective stop below market)
        best_candidate = min(best_candidate, current_price - (2 * tick_size))
        # Monotonicity check
        final_stop = max(current_stop, best_candidate)
        final_stop_rounded = round_to_tick(final_stop, tick_size=tick_size, round_up=False)
        improvement = final_stop_rounded - current_stop
    else:
        # Short position: lower stop is more protective. Monotonic: candidate <= current_stop
        best_candidate, best_reason = min(candidate_stops, key=lambda x: x[0])
        # Must remain protective stop above market
        best_candidate = max(best_candidate, current_price + (2 * tick_size))
        # Monotonicity check
        final_stop = min(current_stop, best_candidate)
        final_stop_rounded = round_to_tick(final_stop, tick_size=tick_size, round_up=True)
        improvement = current_stop - final_stop_rounded

    # 8. Check Minimum Step Requirement
    min_step = trail_cfg.min_step_ticks * tick_size
    is_improvement_sufficient = improvement >= min_step

    trigger, limit = calculate_stop_limit_prices(position.side, final_stop_rounded, config, tick_size)

    reason = (
        f"Trail update ({best_reason}): {current_stop:.2f} -> {final_stop_rounded:.2f} (+{improvement:.2f})"
        if is_improvement_sufficient
        else f"Improvement {improvement:.2f} < min step {min_step:.2f}; stop unchanged"
    )

    return StopEvaluation(
        should_modify=is_improvement_sufficient,
        proposed_stop=final_stop_rounded,
        current_stop=current_stop,
        trigger_price=trigger,
        limit_price=limit,
        r_multiple=current_r,
        best_favorable_price=best_favorable,
        trail_activated=trail_activated,
        modification_count=0,
        reason=reason,
        is_exhaustion=is_exhaustion,
        is_late_session=is_late_session,
    )


def calculate_target_price(
    side: Side,
    entry_price: float,
    initial_stop: float,
    target_r: float = 2.0,
    tick_size: float = 0.05,
    pos: Optional[Any] = None,
    config: Optional[Any] = None,
) -> float:
    """Calculate take-profit target price at a fixed R-multiple from entry (Option A).

    Args:
        side: BUY or SELL
        entry_price: Average entry fill price
        initial_stop: Initial structural stop price
        target_r: R-multiple for target (e.g. 2.0 or 2.5)
        tick_size: Instrument tick size
        pos: Optional position state; if pos.catalyst_aligned is True, expands to 3.50R
        config: Optional configuration object

    Returns:
        Rounded target price.
    """
    r_mult = target_r
    if pos is not None and getattr(pos, "catalyst_aligned", False):
        news_cfg = getattr(config, "news", None) if config else None
        if news_cfg:
            r_mult = float(getattr(news_cfg, "catalyst_target_r", 3.50))
        elif isinstance(config, dict) and "news" in config:
            r_mult = float(config["news"].get("catalyst_target_r", 3.50))
        else:
            r_mult = 3.50

    r_val = abs(entry_price - initial_stop)
    if side == Side.BUY:
        target = entry_price + (r_mult * r_val)
        return round_to_tick(target, tick_size=tick_size, round_up=False)
    else:
        target = entry_price - (r_mult * r_val)
        return round_to_tick(target, tick_size=tick_size, round_up=True)

