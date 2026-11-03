"""Tests for Protective Stop and Monotonic Trailing Stop Manager."""
from datetime import time
import pytest

from apex3.config import CostsConfig, StopsConfig
from apex3.models import Candle, PositionState, Side
from apex3.strategy.trail import (
    calculate_initial_stop,
    calculate_stop_limit_prices,
    calculate_target_price,
    evaluate_trailing_stop,
    round_to_tick,
)


@pytest.fixture
def stops_cfg() -> StopsConfig:
    return StopsConfig()


@pytest.fixture
def costs_cfg() -> CostsConfig:
    return CostsConfig()


def test_round_to_tick_protective_direction():
    """Verify rounding in protective direction for Long (down) and Short (up)."""
    # For Long, protective means lower stop (round down)
    assert round_to_tick(999.03, tick_size=0.05, round_up=False) == 999.00
    assert round_to_tick(999.08, tick_size=0.05, round_up=False) == 999.05

    # For Short, protective means higher stop (round up)
    assert round_to_tick(1001.02, tick_size=0.05, round_up=True) == 1001.05
    assert round_to_tick(1001.06, tick_size=0.05, round_up=True) == 1001.10


def test_initial_stop_calculation(stops_cfg):
    """Verify initial stop placement beyond signal candle with ATR pad."""
    # Long: Entry 1000, Signal Low 995, ATR 2.0, Multiplier 1.5 -> Pad = 3.0 -> Stop = 992.0
    stop_long = calculate_initial_stop(
        side=Side.BUY,
        entry_price=1000.0,
        signal_low=995.0,
        signal_high=1005.0,
        atr_value=2.0,
        config=stops_cfg,
        tick_size=0.05,
    )
    assert stop_long == 992.00
    assert stop_long < 1000.0

    # Short: Entry 1000, Signal High 1005, ATR 2.0, Multiplier 1.5 -> Pad = 3.0 -> Stop = 1008.0
    stop_short = calculate_initial_stop(
        side=Side.SELL,
        entry_price=1000.0,
        signal_low=995.0,
        signal_high=1005.0,
        atr_value=2.0,
        config=stops_cfg,
        tick_size=0.05,
    )
    assert stop_short == 1008.00
    assert stop_short > 1000.0


def test_trail_inactive_before_1r(stops_cfg, costs_cfg):
    """Trail must remain inactive and keep initial stop when movement < +1R."""
    # Entry 1000, Initial Stop 990 (R = 10). Current price 1005 (+0.5R)
    pos = PositionState(
        symbol="TEST",
        side=Side.BUY,
        quantity=10,
        average_entry=1000.0,
        initial_stop_price=990.0,
        protective_stop_price=990.0,
        best_favorable_price=1005.0,
        trail_activated=False,
    )
    eval_res = evaluate_trailing_stop(
        position=pos,
        current_price=1005.0,
        atr_5m=2.0,
        candles_1m=[],
        config=stops_cfg,
        costs_config=costs_cfg,
        current_ist_time=time(10, 0),
    )
    assert not eval_res.trail_activated
    assert not eval_res.should_modify
    assert eval_res.proposed_stop == 990.0


def test_trail_activates_at_1r_and_locks_cost_floor(stops_cfg, costs_cfg):
    """At +1R, trail activates and advances stop to at least entry + costs + profit lock."""
    # Entry 1000, Initial Stop 990 (R = 10). Current price 1012 (+1.2R)
    pos = PositionState(
        symbol="TEST",
        side=Side.BUY,
        quantity=10,
        average_entry=1000.0,
        initial_stop_price=990.0,
        protective_stop_price=990.0,
        best_favorable_price=1012.0,
        trail_activated=False,
    )
    eval_res = evaluate_trailing_stop(
        position=pos,
        current_price=1012.0,
        atr_5m=2.0,
        candles_1m=[],
        config=stops_cfg,
        costs_config=costs_cfg,
        current_ist_time=time(10, 0),
    )
    assert eval_res.trail_activated
    assert eval_res.should_modify
    assert eval_res.proposed_stop > 1000.0  # Above entry, locking profit


def test_monotonicity_long_never_moves_down(stops_cfg, costs_cfg):
    """A long stop must NEVER move down even if price pulls back sharply."""
    pos = PositionState(
        symbol="TEST",
        side=Side.BUY,
        quantity=10,
        average_entry=1000.0,
        initial_stop_price=990.0,
        protective_stop_price=1015.0,  # Already trailed to 1015
        best_favorable_price=1025.0,
        trail_activated=True,
    )
    # Price pulls back to 1017
    eval_res = evaluate_trailing_stop(
        position=pos,
        current_price=1017.0,
        atr_5m=5.0,  # Large ATR would suggest lower stop
        candles_1m=[],
        config=stops_cfg,
        costs_config=costs_cfg,
        current_ist_time=time(10, 30),
    )
    assert eval_res.proposed_stop >= 1015.0  # Never below existing stop!
    assert not eval_res.should_modify


def test_monotonicity_short_never_moves_up(stops_cfg, costs_cfg):
    """A short stop must NEVER move up even if price bounces."""
    pos = PositionState(
        symbol="TEST",
        side=Side.SELL,
        quantity=10,
        average_entry=1000.0,
        initial_stop_price=1010.0,
        protective_stop_price=985.0,  # Already trailed to 985
        best_favorable_price=975.0,
        trail_activated=True,
    )
    # Price bounces to 983
    eval_res = evaluate_trailing_stop(
        position=pos,
        current_price=983.0,
        atr_5m=5.0,
        candles_1m=[],
        config=stops_cfg,
        costs_config=costs_cfg,
        current_ist_time=time(10, 30),
    )
    assert eval_res.proposed_stop <= 985.0  # Never above existing stop!
    assert not eval_res.should_modify


def test_calculate_target_price_buy_and_sell():
    """Verify Option A fixed 2.0R target calculation for Long and Short."""
    # Long: Entry 1000.0, Stop 990.0 -> R = 10.0 -> 2.0R Target = 1020.0
    tgt_long = calculate_target_price(
        side=Side.BUY,
        entry_price=1000.0,
        initial_stop=990.0,
        target_r=2.0,
        tick_size=0.05,
    )
    assert tgt_long == 1020.00

    # Short: Entry 1000.0, Stop 1010.0 -> R = 10.0 -> 2.0R Target = 980.0
    tgt_short = calculate_target_price(
        side=Side.SELL,
        entry_price=1000.0,
        initial_stop=1010.0,
        target_r=2.0,
        tick_size=0.05,
    )
    assert tgt_short == 980.00

