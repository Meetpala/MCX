"""Tests for Portfolio Risk Manager."""
import pytest

from apex3.config import LimitsConfig, RiskConfig
from apex3.models import MarginInfo, PositionState, Side
from apex3.strategy.risk import calculate_position_risk, evaluate_portfolio_risk


@pytest.fixture
def limits_cfg() -> LimitsConfig:
    return LimitsConfig(max_symbols=3)


@pytest.fixture
def risk_cfg() -> RiskConfig:
    return RiskConfig(
        risk_per_trade_pct=0.25,
        max_combined_open_risk_pct=0.75,
        daily_loss_limit_pct=1.0,
    )


def test_position_risk_calculation():
    """Verify stop distance risk calculation and break-even behavior."""
    # Long position with entry 1000, stop 980, qty 10 -> Risk = 10 * 20 = 200
    pos_long = PositionState(
        symbol="INFY",
        side=Side.BUY,
        quantity=10,
        average_entry=1000.0,
        protective_stop_price=980.0,
    )
    assert calculate_position_risk(pos_long) == 200.0

    # Long position where stop trailed to 1010 (profit locked) -> Risk = 0
    pos_locked = PositionState(
        symbol="INFY",
        side=Side.BUY,
        quantity=10,
        average_entry=1000.0,
        protective_stop_price=1010.0,
    )
    assert calculate_position_risk(pos_locked) == 0.0


def test_daily_loss_limit_breach(limits_cfg, risk_cfg):
    """When daily loss reaches 1.0%, can_open_new_trade must be False."""
    m = MarginInfo(equity_available=100000.0, net_value=100000.0)
    # Net loss of ₹1,200 (-1.2%)
    status = evaluate_portfolio_risk(
        open_positions=[],
        margin_info=m,
        proposed_trade_risk_rupees=200.0,
        limits_config=limits_cfg,
        risk_config=risk_cfg,
        additional_realized_pnl=-1200.0,
    )
    assert not status.can_open_new_trade
    assert status.is_daily_loss_breached
    assert status.gate.status.value == "FAIL"


def test_max_positions_limit(limits_cfg, risk_cfg):
    """Holding 3 positions must block opening a 4th."""
    m = MarginInfo(equity_available=100000.0, net_value=100000.0)
    pos_list = [
        PositionState(symbol="P1", quantity=10, average_entry=100.0, protective_stop_price=95.0),
        PositionState(symbol="P2", quantity=10, average_entry=100.0, protective_stop_price=95.0),
        PositionState(symbol="P3", quantity=10, average_entry=100.0, protective_stop_price=95.0),
    ]
    status = evaluate_portfolio_risk(
        open_positions=pos_list,
        margin_info=m,
        proposed_trade_risk_rupees=200.0,
        limits_config=limits_cfg,
        risk_config=risk_cfg,
    )
    assert not status.can_open_new_trade
    assert status.is_max_positions_reached


def test_combined_open_risk_cap(limits_cfg, risk_cfg):
    """Combined risk across positions + proposed must not exceed 0.75% of equity."""
    m = MarginInfo(equity_available=100000.0, net_value=100000.0)
    # Existing position risk = 600 (0.6%)
    pos = PositionState(symbol="P1", quantity=30, average_entry=1000.0, protective_stop_price=980.0)
    # Proposed trade risk = 250 (0.25%) -> Total = 850 (0.85% > 0.75%)
    status = evaluate_portfolio_risk(
        open_positions=[pos],
        margin_info=m,
        proposed_trade_risk_rupees=250.0,
        limits_config=limits_cfg,
        risk_config=risk_cfg,
    )
    assert not status.can_open_new_trade
    assert any("Combined open risk" in r for r in status.reasons)
