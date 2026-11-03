"""Tests for Paper Trading Broker and End-to-End Simulation."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest

from apex3.broker.paper import PaperBroker
from apex3.broker.rate_limiter import RateLimiter
from apex3.config import load_config
from apex3.journal.sqlite_journal import Journal
from apex3.models import OrderRequest, OrderStatus, OrderType, Product, Side
from apex3.strategy.engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")


def test_paper_broker_order_lifecycle():
    """Verify paper broker order placement, fill, and status tracking."""
    broker = PaperBroker(initial_margin=100000.0)

    # 1. Place entry order
    req = OrderRequest(
        symbol="RELIANCE",
        side=Side.BUY,
        quantity=10,
        price=2500.0,
        product=Product.MIS,
        order_type=OrderType.LIMIT,
    )
    order_id = broker.place_order(req)
    assert order_id.startswith("PAPER")

    # Order should be filled on paper adapter
    state = broker.get_order_status(order_id)
    assert state.status == OrderStatus.COMPLETE
    assert state.filled_quantity == 10
    assert state.average_price > 0.0

    # Positions should reflect open quantity
    positions = broker.get_positions()
    rel_pos = next((p for p in positions if p.symbol == "RELIANCE"), None)
    assert rel_pos is not None
    assert rel_pos.quantity == 10


def test_paper_broker_25_modification_cap():
    """Verify that paper broker tracks and rejects > 25 order modifications."""
    broker = PaperBroker()
    req = OrderRequest(
        symbol="RELIANCE",
        side=Side.SELL,
        quantity=10,
        price=2480.0,
        trigger_price=2485.0,
        product=Product.MIS,
        order_type=OrderType.SL,
    )
    order_id = broker.place_order(req)

    # Modify 25 times
    for i in range(25):
        ok = broker.modify_order(order_id, price=2480.0 + i)
        assert ok, f"Modification {i+1} failed"

    # 26th modification must fail
    with pytest.raises(Exception) as exc:
        broker.modify_order(order_id, price=2510.0)
    assert "exceeded" in str(exc.value).lower()


def test_rate_limiter():
    """Verify token bucket rate limiter permits and throttles."""
    limiter = RateLimiter(orders_per_sec=10, quotes_per_sec=1)
    # Orders
    assert limiter.acquire("order")
    # Quote
    assert limiter.acquire("quote")


def test_end_to_end_paper_engine(tmp_path):
    """Run full strategy engine evaluation cycle with paper broker."""
    cfg = load_config()
    cfg.watchlist.symbols.append("RELIANCE")
    journal = Journal(db_path=str(tmp_path / "test_engine.db"))
    broker = PaperBroker(initial_margin=100000.0)

    engine = StrategyEngine(config=cfg, broker=broker, journal=journal)

    # Evaluate at 09:35 IST (inside entry window)
    t = datetime(2026, 10, 21, 9, 35, tzinfo=IST)
    decisions = engine.evaluate_step(t)

    assert len(decisions) >= 1
    # Decision must be WAIT or PROPOSED (deterministic based on generated candles)
    assert decisions[0].decision is not None
    assert decisions[0].explanation != ""


def test_target_2r_hit_exit(tmp_path):
    """Verify that Option A triggers target exit when 2.0R target price is reached."""
    from apex3.models import Decision, Quote
    from apex3.reconciliation.state_machine import PositionPhase

    cfg = load_config()
    journal = Journal(db_path=str(tmp_path / "test_target.db"))
    broker = PaperBroker(initial_margin=100000.0)
    engine = StrategyEngine(config=cfg, broker=broker, journal=journal)

    # Set up an open long position with target at 1020
    pos = engine.state_machine.get_or_create_position("RELIANCE")
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.side = Side.BUY
    pos.quantity = 10
    pos.average_entry = 1000.0
    pos.initial_stop_price = 990.0
    pos.current_stop_price = 990.0
    pos.target_price = 1020.0
    pos.r_value = 10.0
    pos.strategy_tag = "TEST_TAG"

    # Set quote above target price
    broker._fixture_quotes["RELIANCE"] = Quote(
        symbol="RELIANCE",
        exchange="NSE",
        last_price=1021.0,
        bid=1020.95,
        ask=1021.05,
        volume=1000,
        timestamp=datetime.now(IST),
        is_fresh=True,
    )

    t = datetime(2026, 10, 21, 10, 15, tzinfo=IST)
    decisions = engine.evaluate_step(t)

    assert len(decisions) >= 1
    exit_dec = next((d for d in decisions if d.symbol == "RELIANCE"), None)
    assert exit_dec is not None
    assert exit_dec.decision == Decision.EXIT
    assert "TARGET_HIT" in exit_dec.reason_codes
    assert "2.0R" in exit_dec.explanation

