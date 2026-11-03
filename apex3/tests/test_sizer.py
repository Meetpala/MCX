"""Tests for Position Sizing and Equal Allocation Module."""
import pytest

from apex3.config import LimitsConfig, RiskConfig
from apex3.models import InstrumentInfo, MarginInfo, OrderMarginEstimate, Side
from apex3.strategy.sizer import compute_usable_margin, size_position


@pytest.fixture
def limits_cfg() -> LimitsConfig:
    return LimitsConfig(
        max_symbols=2,
        max_leverage=4.0,
        min_accepted_leverage=4.0,
        safety_reserve_pct=5.0,
    )


@pytest.fixture
def risk_cfg() -> RiskConfig:
    return RiskConfig(
        risk_per_trade_pct=0.25,
        max_combined_open_risk_pct=0.75,
        daily_loss_limit_pct=1.0,
        slippage_allowance_pct=0.10,
    )


@pytest.fixture
def inst_info() -> InstrumentInfo:
    return InstrumentInfo(symbol="RELIANCE", tick_size=0.05, lot_size=1)


def test_usable_margin_safety_reserve():
    """Verify 5% safety reserve deduction from available margin."""
    m = MarginInfo(equity_available=100000.0)
    usable = compute_usable_margin(m, safety_reserve_pct=5.0)
    assert usable == 95000.0


def test_equal_allocation_division(limits_cfg, risk_cfg, inst_info):
    """Verify notional budget divides equally across max_symbols slots (e.g. max_symbols=2 -> 190,000 per slot)."""
    m = MarginInfo(equity_available=100000.0, net_value=100000.0)

    # With max_symbols=2: budget = 95,000 * 4.0 / 2 = 190,000
    res = size_position(
        symbol="RELIANCE",
        side=Side.BUY,
        entry_price=1000.0,
        stop_price=990.0,
        num_qualified_symbols=1,
        margin_info=m,
        instrument=inst_info,
        limits_config=limits_cfg,
        risk_config=risk_cfg,
    )
    assert res.per_symbol_budget == 190000.0
    assert res.q_notional == 190


def test_risk_capping_takes_precedence(limits_cfg, risk_cfg, inst_info):
    """When stop distance is wide, Q_risk must restrict quantity below Q_notional."""
    m = MarginInfo(equity_available=100000.0, net_value=100000.0)
    # Entry 1000, Stop 950 -> Loss per share = 50 + 1 (slippage) = 51
    # Trade risk budget = 0.25% of 100,000 = 250
    # Q_risk = floor(250 / 51) = 4 shares
    res = size_position(
        symbol="RELIANCE",
        side=Side.BUY,
        entry_price=1000.0,
        stop_price=950.0,
        num_qualified_symbols=1,
        margin_info=m,
        instrument=inst_info,
        limits_config=limits_cfg,
        risk_config=risk_cfg,
    )
    assert res.is_valid
    assert res.quantity == 4
    assert res.quantity < res.q_notional


def test_rejection_when_leverage_below_minimum(limits_cfg, risk_cfg, inst_info):
    """Skip symbol when broker reported leverage < min_accepted_leverage."""
    m = MarginInfo(equity_available=100000.0, net_value=100000.0)
    est = OrderMarginEstimate(required_margin=30000.0, leverage_available=3.3)  # < 4.0

    res = size_position(
        symbol="RELIANCE",
        side=Side.BUY,
        entry_price=1000.0,
        stop_price=990.0,
        num_qualified_symbols=1,
        margin_info=m,
        instrument=inst_info,
        limits_config=limits_cfg,
        risk_config=risk_cfg,
        order_margin_estimate=est,
    )
    assert not res.is_valid
    assert res.rejection_code == "LEVERAGE_BELOW_MINIMUM"


def test_rejection_invalid_stop_direction(limits_cfg, risk_cfg, inst_info):
    """Long stop >= entry or short stop <= entry must fail validation."""
    m = MarginInfo(equity_available=100000.0)
    res_long = size_position(
        symbol="RELIANCE",
        side=Side.BUY,
        entry_price=1000.0,
        stop_price=1005.0,  # Invalid for long
        num_qualified_symbols=1,
        margin_info=m,
        instrument=inst_info,
        limits_config=limits_cfg,
        risk_config=risk_cfg,
    )
    assert not res_long.is_valid
    assert res_long.rejection_code == "INVALID_STOP_DIRECTION"
