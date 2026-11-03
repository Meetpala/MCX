"""Tests for Order Reconciliation and Controlled Flatten Sequence."""
import pytest

from apex3.broker.paper import PaperBroker
from apex3.journal.sqlite_journal import Journal
from apex3.models import OrderRequest, OrderType, PositionState, Product, Side
from apex3.reconciliation.reconciler import Reconciler
from apex3.reconciliation.state_machine import PositionPhase, StateMachine


@pytest.fixture
def test_setup(tmp_path):
    db_file = str(tmp_path / "test_reconcile.db")
    journal = Journal(db_path=db_file)
    broker = PaperBroker(initial_margin=100000.0)
    sm = StateMachine(session_tag_prefix="APEX3")
    reconciler = Reconciler(broker=broker, state_machine=sm, journal=journal, session_tag="APEX3")
    return broker, sm, journal, reconciler


def test_detect_unknown_manual_positions(test_setup):
    """When an unmanaged/foreign MIS position exists on broker, audit must fail."""
    broker, sm, journal, reconciler = test_setup

    # Simulate a manual position on the broker
    broker._positions["INFY"] = PositionState(
        symbol="INFY",
        side=Side.BUY,
        quantity=50,
        average_entry=1500.0,
    )

    status = reconciler.audit_startup_state(watched_symbols=["RELIANCE", "TCS"])
    assert not status.is_clean
    assert status.has_unknown_positions
    assert "INFY" in status.unknown_symbols


def test_verify_unprotected_positions(test_setup):
    """When a position is open without a protective stop order, flag emergency."""
    broker, sm, journal, reconciler = test_setup

    pos = sm.get_or_create_position("RELIANCE")
    pos.quantity = 20
    pos.phase = PositionPhase.OPEN_UNPROTECTED
    pos.protective_order_id = ""  # No stop

    unprotected = reconciler.verify_protective_stops()
    assert "RELIANCE" in unprotected


def test_execute_flatten_sequence(test_setup):
    """Verify controlled square-off cancels pending entries and exits open positions."""
    broker, sm, journal, reconciler = test_setup

    # Setup an active position
    pos = sm.get_or_create_position("RELIANCE")
    pos.side = Side.BUY
    pos.quantity = 10
    pos.average_entry = 2500.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_101"

    broker._positions["RELIANCE"] = PositionState(
        symbol="RELIANCE",
        side=Side.BUY,
        quantity=10,
        average_entry=2500.0,
    )

    success, msg = reconciler.execute_flatten_sequence()
    assert success
    assert pos.phase in (PositionPhase.CLOSED, PositionPhase.FLAT)
    assert pos.quantity == 0
    assert pos.exit_order_id != ""


def test_step11_reconcile_broker_sl_fill(tmp_path):
    """When a broker SL order is marked COMPLETE, Step 11 cleanly closes the position."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from apex3.config import load_config
    from apex3.models import Candle, Decision, OrderStatus, OrderType, Product, Quote, Side
    from apex3.strategy.engine import StrategyEngine

    IST = ZoneInfo("Asia/Kolkata")
    cfg = load_config()
    db_file = str(tmp_path / "test_step11_sl.db")
    journal = Journal(db_path=db_file)
    broker = PaperBroker(initial_margin=100000.0)
    engine = StrategyEngine(config=cfg, broker=broker, journal=journal)

    # Setup managed position with protective SL order
    pos = engine.state_machine.get_or_create_position("RELIANCE")
    pos.symbol = "RELIANCE"
    pos.side = Side.BUY
    pos.quantity = 10
    pos.average_entry = 2500.0
    pos.initial_stop = 2480.0
    pos.current_stop = 2480.0
    pos.target_price = 2540.0
    pos.r_value = 20.0
    pos.best_favorable_price = 2510.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_12345"

    # Also place an order with COMPLETE status in broker
    order_req = OrderRequest(
        symbol="RELIANCE",
        side=Side.SELL,
        order_type=OrderType.SL,
        product=Product.MIS,
        quantity=10,
        price=2478.0,
        trigger_price=2480.0,
    )
    order_id = broker.place_order(order_req)
    pos.protective_order_id = order_id
    broker._orders[order_id].status = OrderStatus.COMPLETE

    # Provide benchmark and stock data for Step 1-4 and Step 11
    now = datetime(2026, 4, 15, 11, 0, tzinfo=IST)
    bench_candle = Candle(
        timestamp=now,
        open=24000.0,
        high=24050.0,
        low=23950.0,
        close=24010.0,
        volume=1000,
        is_complete=True,
    )
    stock_candle = Candle(
        timestamp=now,
        open=2480.0,
        high=2485.0,
        low=2475.0,
        close=2478.0,
        volume=5000,
        is_complete=True,
    )
    broker._fixture_candles["NIFTY 50"] = [bench_candle] * 60
    broker._fixture_candles["RELIANCE"] = [stock_candle] * 60
    broker._fixture_quotes["NIFTY 50"] = Quote(
        symbol="NIFTY 50",
        exchange="NSE",
        last_price=24010.0,
        bid=24009.0,
        ask=24011.0,
        volume=100000,
        timestamp=now,
    )
    broker._fixture_quotes["RELIANCE"] = Quote(
        symbol="RELIANCE",
        exchange="NSE",
        last_price=2478.0,
        bid=2477.95,
        ask=2478.05,
        volume=50000,
        timestamp=now,
    )

    decisions = engine.evaluate_step(current_time=now)

    # Verify position is closed cleanly in engine state machine
    assert pos.phase == PositionPhase.CLOSED
    assert pos.quantity == 0

    # Verify decision emitted EXIT with STOP_LOSS_FILLED_AT_BROKER
    exit_decisions = [d for d in decisions if d.decision == Decision.EXIT and d.symbol == "RELIANCE"]
    assert len(exit_decisions) == 1
    assert "STOP_LOSS_FILLED_AT_BROKER" in exit_decisions[0].reason_codes
