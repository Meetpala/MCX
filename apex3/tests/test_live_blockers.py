"""Comprehensive unit test suite for verifying resolution of all live blockers.

Covers:
1. Partial fill entry handling: cancel unfilled remainder, exact stop size, target recalculated from actual fill.
2. Timeout path: no premature OPEN_PROTECTED; stop verified before protected; emergency exit if unconfirmed.
3. Target exit & EXITING idempotence: stop cancellation confirmed first; no duplicate market exits in subsequent cycles.
4. Tracked one-shot EOD flatten: verifies zero exposure, does not repeat resubmissions.
5. Daily loss breach: forces immediate square-off of open positions.
6. Decoupled live compliance: open positions managed and flattened even when live flags are disabled.
7. Special session: blocked with SPECIAL_SESSION_NO_MIS.
8. Dashboard CSRF protection: rejects cross-origin mutating requests.
9. Restart recovery: reconciles broker orders and restores managed state without false halt.
"""
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo
import pytest

from apex3.config import AppConfig, load_config
from apex3.models import (
    Candle,
    Decision,
    InstrumentInfo,
    MarginInfo,
    OpeningRange,
    OrderMarginEstimate,
    OrderRequest,
    OrderState,
    OrderStatus,
    OrderType,
    PositionState,
    Product,
    Quote,
    Side,
)
from apex3.strategy.regime import MarketRegime
from apex3.reconciliation.reconciler import Reconciler
from apex3.reconciliation.state_machine import PositionPhase, StateMachine
from apex3.strategy.engine import StrategyEngine
from apex3.strategy.signal import SignalEvaluation, GateResult, GateStatus, evaluate_signal

IST = ZoneInfo("Asia/Kolkata")


def test_partial_fill_cancels_remainder_and_sizes_stop_exactly():
    """When an entry partially fills, unfilled remainder is cancelled and SL is placed for exact filled quantity."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_positions.return_value = []
    mock_broker.estimate_order_margin.return_value = OrderMarginEstimate(required_margin=3000.0, leverage_available=4.0)

    now = datetime(2026, 3, 20, 9, 35, tzinfo=IST)
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05)
    mock_broker.get_quote.return_value = Quote(symbol="TRENT", exchange="NSE", last_price=1000.0, bid=999.8, ask=1000.2, volume=10000, timestamp=now)

    placed_orders = []
    def on_place(req):
        ord_id = f"ORD_{len(placed_orders) + 1}"
        placed_orders.append((ord_id, req))
        return ord_id
    mock_broker.place_order.side_effect = on_place

    cancelled_orders = set()
    def on_cancel(ord_id):
        cancelled_orders.add(ord_id)
        return True
    mock_broker.cancel_order.side_effect = on_cancel

    # Entry order for 14 shares partially fills for 6 shares @ 1001.0; once cancelled, confirmed terminal
    def on_status(ord_id):
        if "ORD_1" in ord_id:
            status = OrderStatus.CANCELLED if "ORD_1" in cancelled_orders else OrderStatus.PENDING
            pending_q = 0 if "ORD_1" in cancelled_orders else 8
            return OrderState(
                order_id="ORD_1",
                request=OrderRequest(symbol="TRENT", side=Side.BUY, quantity=14, price=1000.05),
                status=status,
                filled_quantity=6,
                pending_quantity=pending_q,
                average_price=1001.0,
            )
        elif "ORD_2" in ord_id:
            # Protective SL
            return OrderState(
                order_id="ORD_2",
                request=OrderRequest(symbol="TRENT", side=Side.SELL, quantity=6, order_type=OrderType.SL),
                status=OrderStatus.TRIGGER_PENDING,
                pending_quantity=6,
                filled_quantity=0,
            )
        return None
    mock_broker.get_order_status.side_effect = on_status
    mock_broker.get_positions.side_effect = lambda: (
        [PositionState(symbol="TRENT", side=Side.BUY, quantity=6, average_entry=1001.0, product=Product.MIS)]
        if "ORD_1" in cancelled_orders else []
    )

    cfg = AppConfig()
    object.__setattr__(cfg.watchlist, "symbols", ["TRENT"])
    object.__setattr__(cfg.watchlist, "scanner_enabled", False)
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # 20 candles with ATR ~ 2.0 and last candle low=996.0 -> stop distance ~ 7.05 (0.705%), passing Gate 9 [0.55%, 0.90%]
    candles = [Candle(timestamp=now - timedelta(minutes=5 * (20 - i)), open=1000.0, high=1001.0, low=999.0, close=1000.0, volume=10000, is_complete=True) for i in range(20)]
    candles[-1] = Candle(timestamp=now, open=1000.0, high=1001.0, low=996.0, close=1000.0, volume=10000, is_complete=True)
    mock_broker.get_historical.return_value = candles

    fake_sig = SignalEvaluation(
        symbol="TRENT",
        decision=Decision.PROPOSED_LONG,
        signal_price=1000.0,
        side=Side.BUY,
        gates=[GateResult("TEST", GateStatus.PASS, "OK")],
        relative_strength=0.05,
        relative_volume=3.0,
    )

    import apex3.strategy.engine as eng_module
    orig_eval = eng_module.evaluate_signal
    orig_regime = eng_module.evaluate_market_regime
    try:
        eng_module.evaluate_signal = MagicMock(return_value=fake_sig)
        eng_module.evaluate_market_regime = MagicMock(return_value=MagicMock(regime=MarketRegime.BULLISH, is_valid=True))

        engine.evaluate_step(now)

        # 1. Remainder was cancelled
        assert mock_broker.cancel_order.call_count >= 1
        assert mock_broker.cancel_order.call_args[0][0] == "ORD_1"

        # 2. SL placed for exact filled quantity (6, not 14)
        assert len(placed_orders) == 2
        sl_req = placed_orders[1][1]
        assert sl_req.order_type == OrderType.SL
        assert sl_req.quantity == 6

        # 3. Position protected for 6 shares and target recalculated from actual fill 1001.0
        pos = engine.state_machine.positions["TRENT"]
        assert pos.phase == PositionPhase.OPEN_PROTECTED
        assert pos.quantity == 6
        assert pos.average_entry == 1001.0
        assert pos.target_price > 1001.0
    finally:
        eng_module.evaluate_signal = orig_eval
        eng_module.evaluate_market_regime = orig_regime


def test_partial_fill_unconfirmed_cancellation_never_leaves_shares_unprotected():
    """Fix 2: When entry remainder cancellation is unconfirmed, engine does not abort; it places protective stop for filled shares immediately."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_positions.return_value = []
    mock_broker.estimate_order_margin.return_value = OrderMarginEstimate(required_margin=3000.0, leverage_available=4.0)

    now = datetime(2026, 3, 20, 9, 35, tzinfo=IST)
    mock_broker.get_instrument_info.return_value = InstrumentInfo(
        symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05, upper_circuit=1200.0, lower_circuit=800.0, mis_available=True
    )
    mock_broker.get_quote.return_value = Quote(
        symbol="TRENT", exchange="NSE", last_price=1000.0, bid=999.8, ask=1000.2, volume=10000, timestamp=now, is_fresh=True
    )

    placed_orders = []
    def on_place(req):
        ord_id = f"ORD_{len(placed_orders) + 1}"
        placed_orders.append((ord_id, req))
        return ord_id
    mock_broker.place_order.side_effect = on_place

    # Order ORD_1 remains PENDING even after cancel request, but ORD_2 (stop) is TRIGGER_PENDING
    def on_status(ord_id):
        if "ORD_1" in ord_id:
            return OrderState(
                order_id="ORD_1",
                request=OrderRequest(symbol="TRENT", side=Side.BUY, quantity=14, price=1000.05),
                status=OrderStatus.PENDING,
                filled_quantity=6,
                pending_quantity=8,
                average_price=1001.0,
            )
        elif "ORD_2" in ord_id:
            return OrderState(
                order_id="ORD_2",
                request=OrderRequest(symbol="TRENT", side=Side.SELL, quantity=6, order_type=OrderType.SL),
                status=OrderStatus.TRIGGER_PENDING,
                pending_quantity=6,
                filled_quantity=0,
            )
        return None
    mock_broker.get_order_status.side_effect = on_status

    cfg = AppConfig()
    object.__setattr__(cfg.watchlist, "symbols", ["TRENT"])
    object.__setattr__(cfg.watchlist, "scanner_enabled", False)
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    candles = [Candle(timestamp=now - timedelta(minutes=5 * (20 - i)), open=1000.0, high=1001.0, low=999.0, close=1000.0, volume=10000, is_complete=True) for i in range(20)]
    candles[-1] = Candle(timestamp=now, open=1000.0, high=1001.0, low=996.0, close=1000.0, volume=10000, is_complete=True)
    mock_broker.get_historical.return_value = candles

    fake_sig = SignalEvaluation(
        symbol="TRENT",
        decision=Decision.PROPOSED_LONG,
        signal_price=1000.0,
        side=Side.BUY,
        gates=[GateResult("TEST", GateStatus.PASS, "OK")],
        relative_strength=0.05,
        relative_volume=3.0,
    )

    import apex3.strategy.engine as eng_module
    orig_eval = eng_module.evaluate_signal
    orig_regime = eng_module.evaluate_market_regime
    try:
        eng_module.evaluate_signal = MagicMock(return_value=fake_sig)
        eng_module.evaluate_market_regime = MagicMock(return_value=MagicMock(regime=MarketRegime.BULLISH, is_valid=True))

        engine.evaluate_step(now)

        # 1. Remainder cancellation was attempted
        assert mock_broker.cancel_order.call_count >= 1

        # 2. While remainder cancel is still unconfirmed, position transitions to OPEN_PROTECTED because protective stop is confirmed!
        pos = engine.state_machine.positions["TRENT"]
        assert pos.phase == PositionPhase.OPEN_PROTECTED
        assert len(placed_orders) == 2  # Entry order + immediate protective stop for filled shares
        sl_ord_id, sl_req = placed_orders[1]
        assert sl_req.order_type == OrderType.SL
        assert sl_req.quantity == 6
        assert pos.protective_order_id == "ORD_2"

        # 3. Next step: remainder is now confirmed CANCELLED and broker positions confirm 6 shares
        def on_status_confirmed(ord_id):
            if "ORD_1" in ord_id:
                return OrderState(
                    order_id="ORD_1",
                    request=OrderRequest(symbol="TRENT", side=Side.BUY, quantity=14, price=1000.05),
                    status=OrderStatus.CANCELLED,
                    filled_quantity=6,
                    pending_quantity=0,
                    average_price=1001.0,
                )
            elif "ORD_2" in ord_id:
                return OrderState(
                    order_id="ORD_2",
                    request=OrderRequest(symbol="TRENT", side=Side.SELL, quantity=6, order_type=OrderType.SL),
                    status=OrderStatus.TRIGGER_PENDING,
                    pending_quantity=6,
                    filled_quantity=0,
                )
            return None
        mock_broker.get_order_status.side_effect = on_status_confirmed
        mock_broker.get_positions.return_value = [
            PositionState(symbol="TRENT", side=Side.BUY, quantity=6, average_entry=1001.0, product=Product.MIS)
        ]

        engine.evaluate_step(now + timedelta(seconds=1))

        # 4. Now that remainder is confirmed terminal, protective stop IS placed for exact filled quantity (6 shares)
        assert len(placed_orders) == 2
        sl_ord_id, sl_req = placed_orders[1]
        assert sl_req.order_type == OrderType.SL
        assert sl_req.quantity == 6
        assert pos.phase == PositionPhase.OPEN_PROTECTED
        assert pos.quantity == 6
        assert pos.average_entry == 1001.0
    finally:
        eng_module.evaluate_signal = orig_eval
        eng_module.evaluate_market_regime = orig_regime


def test_target_exit_idempotence_prevents_duplicate_exits():
    """At 2R target, stop cancellation is confirmed, exit order is tracked in EXITING, and no duplicate exit is sent."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_positions.return_value = []

    now = datetime(2026, 3, 20, 10, 30, tzinfo=IST)
    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("INFY")
    pos.side = Side.BUY
    pos.quantity = 25
    pos.average_entry = 1500.0
    pos.initial_stop_price = 1490.0
    pos.current_stop_price = 1490.0
    pos.target_price = 1520.0
    pos.r_value = 10.0
    pos.protective_order_id = "SL_RESTING_123"
    pos.phase = PositionPhase.OPEN_PROTECTED

    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="INFY", exchange="NSE", tick_size=0.05, lot_size=1)
    # Price reaches target 1520.0
    mock_broker.get_quote.return_value = Quote(symbol="INFY", exchange="NSE", last_price=1521.0, bid=1520.95, ask=1521.05, volume=50000, timestamp=now, is_fresh=True)

    def on_status_infy(ord_id):
        if ord_id == "SL_RESTING_123":
            if mock_broker.cancel_order.called:
                return OrderState(order_id=ord_id, status=OrderStatus.CANCELLED, pending_quantity=0)
            return OrderState(order_id=ord_id, status=OrderStatus.TRIGGER_PENDING, pending_quantity=25)
        return OrderState(order_id=ord_id, status=OrderStatus.PENDING, pending_quantity=25)
    mock_broker.get_order_status.side_effect = on_status_infy
    mock_broker.place_order.return_value = "EXIT_MKT_999"

    # Cycle 1: Reaches target -> cancels stop -> places exit order -> enters EXITING phase
    decisions1 = engine.evaluate_step(now)
    assert mock_broker.cancel_order.call_count == 1
    assert mock_broker.place_order.call_count == 1
    assert pos.phase == PositionPhase.EXITING
    assert pos.exit_order_id == "EXIT_MKT_999"

    # Cycle 2: 3 seconds later, price is still at target, exit order still pending at broker
    now_later = now + timedelta(seconds=3)
    decisions2 = engine.evaluate_step(now_later)
    # Must NOT place a duplicate market exit order!
    assert mock_broker.place_order.call_count == 1
    assert pos.phase == PositionPhase.EXITING

    # Cycle 3: Broker completes the market exit
    mock_broker.get_order_status.side_effect = lambda ord_id: OrderState(
        order_id=ord_id,
        status=OrderStatus.COMPLETE,
        filled_quantity=25,
        average_price=1521.0,
    )
    decisions3 = engine.evaluate_step(now + timedelta(seconds=6))
    assert pos.phase == PositionPhase.CLOSED
    assert pos.quantity == 0


def test_tracked_one_shot_eod_flatten_does_not_repeat():
    """EOD flatten is executed once, confirms flat with broker, and does not repeatedly cancel/resubmit orders."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True

    # Flatten time at 15:15 IST
    flatten_time = datetime(2026, 3, 20, 15, 15, 5, tzinfo=IST)
    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Position in state machine
    pos = engine.state_machine.get_or_create_position("RELIANCE")
    pos.side = Side.BUY
    pos.quantity = 10
    pos.average_entry = 2500.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_REL_1"

    # Broker has 10 shares open initially
    broker_positions = [PositionState(symbol="RELIANCE", side=Side.BUY, quantity=10, average_entry=2500.0)]
    mock_broker.get_positions.side_effect = lambda: list(broker_positions)
    mock_broker.get_orders.return_value = []

    def on_exit(req):
        broker_positions.clear()  # Broker confirmed flat
        return "EXIT_FLATTEN_ORD"
    mock_broker.place_order.side_effect = on_exit

    # Cycle 1: Flatten window triggered -> executes flatten sequence
    d1 = engine.evaluate_step(flatten_time)
    assert any(d.reason_codes == ["FLATTEN_WINDOW_ACTIVE"] for d in d1)
    assert mock_broker.place_order.call_count == 1
    assert engine._flatten_completed is True
    assert pos.phase in (PositionPhase.CLOSED, PositionPhase.FLAT)

    # Cycle 2: 3 seconds later in flatten window -> returns FLATTEN_COMPLETED without placing orders
    d2 = engine.evaluate_step(flatten_time + timedelta(seconds=3))
    assert any(d.reason_codes == ["FLATTEN_COMPLETED"] for d in d2)
    assert mock_broker.place_order.call_count == 1  # Still 1! No duplicate orders resubmitted!


def test_daily_loss_limit_breach_forces_immediate_flatten():
    """Breaching daily loss limit squares off open positions to make daily loss cap truly hard."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 11, 0, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Holding 50 shares of TCS bought at 3000, now dropped to 2990 (-500 rupees loss, breaches -240 limit)
    pos = engine.state_machine.get_or_create_position("TCS")
    pos.side = Side.BUY
    pos.quantity = 50
    pos.average_entry = 3000.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_TCS"

    broker_positions = [PositionState(symbol="TCS", side=Side.BUY, quantity=50, average_entry=3000.0, current_price=2990.0)]
    mock_broker.get_positions.side_effect = lambda: list(broker_positions)
    mock_broker.get_quote.return_value = Quote(
        symbol="TCS",
        exchange="NSE",
        last_price=2990.0,
        bid=2989.5,
        ask=2990.5,
        volume=5000,
        timestamp=now,
        is_fresh=True,
    )
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_margins.return_value = MarginInfo(equity_available=14500.0, net_value=14500.0)

    def on_flatten(req):
        broker_positions.clear()
        return "FLATTEN_TCS_ORD"
    mock_broker.place_order.side_effect = on_flatten

    decisions = engine.evaluate_step(now)

    # 1. Engine halts with DAILY_LOSS_LIMIT_BREACHED
    halt_decisions = [d for d in decisions if d.decision == Decision.HALT and "DAILY_LOSS_LIMIT_BREACHED" in d.reason_codes]
    assert len(halt_decisions) >= 1
    # 2. Flatten sequence was executed for open positions
    assert mock_broker.place_order.call_count >= 1
    assert pos.phase in (PositionPhase.CLOSED, PositionPhase.FLAT)


def test_decoupled_live_gates_allow_open_position_management():
    """When LIVE_TRADING_ENABLED or STATIC_IP is False, open positions are still managed and flattened."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 11, 0, tzinfo=IST)

    cfg = AppConfig()
    object.__setattr__(cfg, "mode", "live")
    object.__setattr__(cfg.secrets, "live_trading_enabled", False)  # Disabled!
    object.__setattr__(cfg.secrets, "static_ip_configured", False)

    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Existing open position
    pos = engine.state_machine.get_or_create_position("HINDALCO")
    pos.side = Side.BUY
    pos.quantity = 40
    pos.average_entry = 650.0
    pos.initial_stop_price = 645.0
    pos.current_stop_price = 645.0
    pos.target_price = 660.0
    pos.r_value = 5.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_HINDALCO"

    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="HINDALCO", exchange="NSE", tick_size=0.05, lot_size=1)
    # Stop is hit at broker
    mock_broker.get_order_status.return_value = OrderState(
        order_id="SL_HINDALCO",
        status=OrderStatus.COMPLETE,
        filled_quantity=40,
        average_price=645.0,
    )
    mock_broker.get_positions.return_value = []
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)

    decisions = engine.evaluate_step(now)

    # Position is cleanly closed despite live flag disabled!
    assert pos.phase == PositionPhase.CLOSED
    assert pos.quantity == 0
    # And new entries are blocked with LIVE_TRADING_FLAG_DISABLED
    assert any("LIVE_TRADING_FLAG_DISABLED" in d.reason_codes for d in decisions)


def test_special_session_blocks_new_entries():
    """On special session days (e.g. Muhurat Trading), new intraday MIS entries are blocked."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_positions.return_value = []
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)

    # Load real config containing Muhurat Trading special session 2026-11-08
    cfg = load_config()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Date of Muhurat trading: 2026-11-08 (Sunday special session)
    muhurat_now = datetime(2026, 11, 8, 18, 20, tzinfo=IST)
    decisions = engine.evaluate_step(muhurat_now)
    assert any("SPECIAL_SESSION_NO_MIS" in d.reason_codes for d in decisions)


def test_dashboard_csrf_protection_rejects_external_origin():
    """Dashboard control endpoints reject non-loopback Origin/Referer requests to prevent CSRF."""
    from apex3.dashboard.app import create_app
    cfg = AppConfig()
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)
    app = create_app(engine, cfg, mock_journal)
    client = app.test_client()

    # Valid loopback origin
    resp_valid = client.post("/api/pause", json={"pause": True}, headers={"Origin": "http://127.0.0.1:5000"})
    assert resp_valid.status_code == 200

    # Malicious external origin
    resp_blocked = client.post("/api/pause", json={"pause": True}, headers={"Origin": "https://attacker-site.com"})
    assert resp_blocked.status_code == 403
    assert b"Cross-site request rejected" in resp_blocked.data


def test_restart_recovery_reconciles_strategy_orders_without_unknown_positions_halt():
    """Restart recovery restores ManagedPosition from tagged orders without halting on UNKNOWN_MANUAL_POSITIONS."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    sm = StateMachine("APEX3")
    reconciler = Reconciler(mock_broker, sm, mock_journal, session_tag="APEX3")

    mock_broker.get_positions.return_value = [
        PositionState(symbol="INFY", side=Side.BUY, quantity=10, average_entry=1500.0)
    ]
    mock_broker.get_orders.return_value = [
        OrderState(
            order_id="SL_RECOVERED_1",
            request=OrderRequest(symbol="INFY", side=Side.SELL, quantity=10, order_type=OrderType.SL, price=1485.0, trigger_price=1485.0, tag="APEX3_INFY_SL"),
            status=OrderStatus.TRIGGER_PENDING,
            pending_quantity=10,
        )
    ]

    audit = reconciler.audit_startup_state(["INFY"])
    assert audit.is_clean is True
    assert audit.has_unknown_positions is False
    assert "INFY" in sm.positions
    recovered_pos = sm.positions["INFY"]
    assert recovered_pos.phase == PositionPhase.OPEN_PROTECTED
    assert recovered_pos.quantity == 10
    assert recovered_pos.protective_order_id == "SL_RECOVERED_1"
    # Fix 7 verification: initial_stop_price, r_value, and target_price are populated
    assert recovered_pos.initial_stop_price > 0
    assert recovered_pos.r_value > 0
    assert recovered_pos.target_price > 1500.0


def test_broker_tags_strictly_within_20_chars():
    """All broker tags must be strictly <= 20 chars to comply with Zerodha Kite Connect v3."""
    from apex3.models import make_broker_tag
    session_tag = "APEX3"

    tags = [
        make_broker_tag(session_tag, "SUNPHARMA", "E0935"),
        make_broker_tag(session_tag, "SUNPHARMA", "SL"),
        make_broker_tag(session_tag, "SUNPHARMA", "TGT"),
        make_broker_tag(session_tag, "SUNPHARMA", "EMG"),
        make_broker_tag(session_tag, "SUNPHARMA", "FLT"),
        make_broker_tag("APEX3_PROD", "BHARTIARTL", "EMERGENCY_FLATTEN"),
    ]
    for tag in tags:
        assert len(tag) <= 20, f"Tag '{tag}' exceeds 20 characters ({len(tag)})"
        assert tag.startswith(session_tag[:5]), f"Tag '{tag}' does not start with session prefix"


def test_candidate_ranking_evaluates_next_candidate_if_first_fails_gate9():
    """If Candidate #1 fails Gate 9 (stop-width bounds), engine proceeds to evaluate Candidate #2."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_positions.return_value = []
    mock_broker.estimate_order_margin.return_value = OrderMarginEstimate(required_margin=3000.0, leverage_available=4.0)

    now = datetime(2026, 3, 20, 9, 35, tzinfo=IST)
    cfg = AppConfig()
    object.__setattr__(cfg.watchlist, "symbols", ["STOCK_A", "STOCK_B"])
    object.__setattr__(cfg.watchlist, "scanner_enabled", False)
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # STOCK_A has stop width outside bounds (e.g. 2.2% > 0.90%)
    # STOCK_B has stop width inside bounds (~0.70%)
    inst_a = InstrumentInfo(symbol="STOCK_A", exchange="NSE", lot_size=1, tick_size=0.05)
    inst_b = InstrumentInfo(symbol="STOCK_B", exchange="NSE", lot_size=1, tick_size=0.05)
    mock_broker.get_instrument_info.side_effect = lambda sym: inst_a if sym == "STOCK_A" else inst_b

    quote_a = Quote(symbol="STOCK_A", exchange="NSE", last_price=1000.0, bid=999.8, ask=1000.2, volume=10000, timestamp=now)
    quote_b = Quote(symbol="STOCK_B", exchange="NSE", last_price=500.0, bid=499.9, ask=500.1, volume=10000, timestamp=now)
    mock_broker.get_quote.side_effect = lambda sym: quote_a if sym == "STOCK_A" else quote_b

    # STOCK_A swing low 970 (wide stop ~3%), STOCK_B swing low 497 (stop ~0.7%)
    candles_a = [Candle(timestamp=now - timedelta(minutes=5 * (20 - i)), open=1000.0, high=1005.0, low=970.0, close=1000.0, volume=10000, is_complete=True) for i in range(20)]
    candles_b = [Candle(timestamp=now - timedelta(minutes=5 * (20 - i)), open=500.0, high=500.5, low=499.5, close=500.0, volume=10000, is_complete=True) for i in range(20)]
    candles_b[-1] = Candle(timestamp=now, open=500.0, high=500.5, low=498.0, close=500.0, volume=10000, is_complete=True)
    mock_broker.get_historical.side_effect = lambda sym, *args, **kwargs: candles_a if sym == "STOCK_A" else candles_b

    sig_a = SignalEvaluation(symbol="STOCK_A", decision=Decision.PROPOSED_LONG, signal_price=1000.0, side=Side.BUY, gates=[], relative_strength=0.08, relative_volume=4.0)
    sig_b = SignalEvaluation(symbol="STOCK_B", decision=Decision.PROPOSED_LONG, signal_price=500.0, side=Side.BUY, gates=[], relative_strength=0.05, relative_volume=3.0)

    import apex3.strategy.engine as eng_module
    orig_eval = eng_module.evaluate_signal
    orig_regime = eng_module.evaluate_market_regime
    try:
        eng_module.evaluate_signal = lambda symbol, **kwargs: sig_a if symbol == "STOCK_A" else sig_b
        eng_module.evaluate_market_regime = MagicMock(return_value=MagicMock(regime=MarketRegime.BULLISH, is_valid=True))

        mock_broker.place_order.return_value = "ORD_B_1"
        mock_broker.get_order_status.return_value = OrderState(
            order_id="ORD_B_1",
            request=OrderRequest(symbol="STOCK_B", side=Side.BUY, quantity=20, price=500.05),
            status=OrderStatus.PENDING,
            filled_quantity=0,
            pending_quantity=20,
        )

        decisions = engine.evaluate_step(now)
        # STOCK_A is rejected on STOP_WIDTH_OUT_OF_BOUNDS
        # But STOCK_B is successfully evaluated and proposed!
        submitted = [d for d in decisions if d.decision == Decision.PROPOSED_LONG and d.symbol == "STOCK_B"]
        assert len(submitted) == 1, "Candidate #2 (STOCK_B) should be proposed after Candidate #1 failed Gate 9"
    finally:
        eng_module.evaluate_signal = orig_eval
        eng_module.evaluate_market_regime = orig_regime


def test_step11_fee_lock_no_unbound_local_error_when_modify_fails():
    """In Step 11, when fee-lock triggers but broker modify_order fails, no UnboundLocalError is raised."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 11, 0, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Position in profit >= +1.60R
    pos = engine.state_machine.get_or_create_position("HINDALCO")
    pos.side = Side.BUY
    pos.quantity = 40
    pos.average_entry = 650.0
    pos.initial_stop_price = 645.0
    pos.current_stop_price = 645.0
    pos.target_price = 660.0
    pos.r_value = 5.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_HINDALCO_1"

    # Current price 658.5 -> +1.70R
    mock_broker.get_quote.return_value = Quote(
        symbol="HINDALCO",
        exchange="NSE",
        last_price=658.5,
        bid=658.4,
        ask=658.6,
        volume=50000,
        timestamp=now,
        is_fresh=True,
    )
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="HINDALCO", exchange="NSE", lot_size=1, tick_size=0.05)
    mock_broker.get_order_status.return_value = OrderState(
        order_id="SL_HINDALCO_1",
        status=OrderStatus.TRIGGER_PENDING,
        pending_quantity=40,
    )
    # modify_order fails!
    mock_broker.modify_order.return_value = False

    # Must execute cleanly without UnboundLocalError
    decisions = engine.evaluate_step(now)
    assert len(decisions) >= 1
    assert decisions[0].decision == Decision.MANAGE
    assert pos.current_stop_price == 645.0  # Internal stop not moved because modification unconfirmed


def test_reconciler_step3_does_not_swallow_stop_loss_event():
    """When broker position drops to quantity==0 because protective SL filled, Step 3 does not wipe it before Step 11 logs STOP_LOSS_HIT."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 0, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Position in OPEN_PROTECTED state with resting stop order
    pos = engine.state_machine.get_or_create_position("TRENT")
    pos.side = Side.BUY
    pos.quantity = 20
    pos.average_entry = 1000.0
    pos.initial_stop_price = 993.0
    pos.current_stop_price = 993.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_TRENT_1"

    # Broker position has dropped to 0 (stop filled at Zerodha)
    mock_broker.get_positions.return_value = [PositionState(symbol="TRENT", quantity=0)]
    mock_broker.get_orders.return_value = []
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05)
    mock_broker.get_quote.return_value = Quote(symbol="TRENT", exchange="NSE", last_price=992.5, bid=992.4, ask=992.6, volume=20000, timestamp=now, is_fresh=True)

    # Stop order filled with COMPLETE status
    mock_broker.get_order_status.return_value = OrderState(
        order_id="SL_TRENT_1",
        status=OrderStatus.COMPLETE,
        filled_quantity=20,
        average_price=992.8,
    )

    decisions = engine.evaluate_step(now)

    # Must NOT be swallowed by Step 3; Step 11 must process the fill cleanly
    sl_decisions = [d for d in decisions if d.decision == Decision.EXIT and "STOP_LOSS_FILLED_AT_BROKER" in d.reason_codes]
    assert len(sl_decisions) == 1, "Step 11 should return EXIT with STOP_LOSS_FILLED_AT_BROKER"

    # Journal must log STOP_LOSS_HIT event with fill_price
    sl_hit_logs = [call for call in mock_journal.log_event.call_args_list if call.kwargs.get("event_type") == "STOP_LOSS_HIT"]
    assert len(sl_hit_logs) == 1, "Journal must log STOP_LOSS_HIT event"
    assert sl_hit_logs[0].kwargs.get("data", {}).get("fill_price") == 992.8

    # Position is now cleanly closed and flat
    assert pos.phase == PositionPhase.CLOSED
    assert pos.quantity == 0


def test_target_and_gap_exit_wait_for_stop_cancellation_and_deduct_partial_fills():
    """Target exit and Gap exit do not place market exit while SL is still TRIGGER_PENDING/OPEN, and Gap exit deducts partial SL fills."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 15, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # --- Part A: Target Exit with cancellation in-flight (TRIGGER_PENDING) ---
    pos_a = engine.state_machine.get_or_create_position("HINDALCO")
    pos_a.side = Side.BUY
    pos_a.quantity = 40
    pos_a.average_entry = 650.0
    pos_a.initial_stop_price = 645.0
    pos_a.current_stop_price = 645.0
    pos_a.target_price = 660.0
    pos_a.phase = PositionPhase.OPEN_PROTECTED
    pos_a.protective_order_id = "SL_HIND_1"

    mock_broker.get_positions.return_value = [PositionState(symbol="HINDALCO", quantity=40, average_entry=650.0)]
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="HINDALCO", exchange="NSE", lot_size=1, tick_size=0.05)
    mock_broker.get_quote.return_value = Quote(symbol="HINDALCO", exchange="NSE", last_price=662.0, bid=661.9, ask=662.1, volume=50000, timestamp=now, is_fresh=True)
    # SL cancellation failed/unconfirmed (cancel_order returns False, order still TRIGGER_PENDING)
    mock_broker.get_order_status.return_value = OrderState(order_id="SL_HIND_1", status=OrderStatus.TRIGGER_PENDING, pending_quantity=40)
    mock_broker.cancel_order.return_value = False

    decisions = engine.evaluate_step(now)
    # Market exit order must NOT have been placed while SL cancellation unconfirmed and TRIGGER_PENDING
    mock_broker.place_order.assert_not_called()
    assert pos_a.phase == PositionPhase.OPEN_PROTECTED
    assert pos_a.quantity == 40

    # --- Part B: Gap Exit with partial fill on SL ---
    engine_b = StrategyEngine(cfg, mock_broker, mock_journal)
    pos_b = engine_b.state_machine.get_or_create_position("TATASTEEL")
    pos_b.side = Side.BUY
    pos_b.quantity = 50
    pos_b.average_entry = 155.0
    pos_b.initial_stop_price = 150.0
    pos_b.current_stop_price = 150.0
    pos_b.phase = PositionPhase.OPEN_PROTECTED
    pos_b.protective_order_id = "SL_TATA_1"

    mock_broker.reset_mock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.cancel_order.return_value = True
    mock_broker.get_positions.return_value = [PositionState(symbol="TATASTEEL", quantity=50, average_entry=155.0)]
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TATASTEEL", exchange="NSE", lot_size=1, tick_size=0.05)
    # Market gapped down to 140.0 through limit price 150.0
    mock_broker.get_quote.return_value = Quote(symbol="TATASTEEL", exchange="NSE", last_price=140.0, bid=139.9, ask=140.1, volume=50000, timestamp=now, is_fresh=True)

    placed_orders = []
    def on_place(req):
        placed_orders.append(req)
        return "EXIT_GAP_1"
    mock_broker.place_order.side_effect = on_place

    def sl_order_status(ord_id):
        if ord_id == "SL_TATA_1":
            if mock_broker.cancel_order.called:
                # After cancel, broker reports 12 shares were filled, remaining cancelled
                return OrderState(order_id="SL_TATA_1", status=OrderStatus.CANCELLED, filled_quantity=12, pending_quantity=0)
            # Before cancel, order triggered and resting OPEN on book
            return OrderState(
                order_id="SL_TATA_1",
                request=OrderRequest(symbol="TATASTEEL", price=150.0, trigger_price=150.0),
                status=OrderStatus.OPEN,
                pending_quantity=50,
            )
        return None
    mock_broker.get_order_status.side_effect = sl_order_status

    decisions_b = engine_b.evaluate_step(now)
    # Gap exit should be dispatched for remaining quantity strictly: 50 - 12 = 38 shares
    assert len(placed_orders) == 1, "Should place emergency gap exit order"
    assert placed_orders[0].quantity == 38, f"Emergency exit quantity should be 38 (50 - 12), got {placed_orders[0].quantity}"
    assert placed_orders[0].order_type in (OrderType.LIMIT, OrderType.MARKET)
    gap_decisions = [d for d in decisions_b if "STOP_TRIGGERED_UNFILLED_GAP_EXIT" in d.reason_codes]
    assert len(gap_decisions) == 1


def test_signal_fails_closed_on_missing_quote_timestamp():
    """evaluate_signal fails closed with Decision.WAIT and ['STALE_QUOTE', 'MISSING_QUOTE_TIMESTAMP'] when quote.timestamp is None."""
    now = datetime(2026, 3, 20, 10, 0, tzinfo=IST)
    cfg = AppConfig()

    inst = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05)
    quote = Quote(
        symbol="TRENT",
        exchange="NSE",
        last_price=1000.0,
        bid=999.8,
        ask=1000.2,
        volume=10000,
        timestamp=None,  # Missing timestamp!
        is_fresh=True,
    )

    candles = [
        Candle(
            timestamp=now - timedelta(minutes=5 * (20 - i)),
            open=1000.0,
            high=1002.0,
            low=998.0,
            close=1000.0,
            volume=50000,
            is_complete=True,
        )
        for i in range(20)
    ]
    or_range = OpeningRange(
        symbol="TRENT",
        high=1005.0,
        low=995.0,
        is_complete=True,
        candle_count=3,
    )

    eval_res = evaluate_signal(
        symbol="TRENT",
        stock_candles_5m=candles,
        benchmark_candles_5m=candles,
        opening_range=or_range,
        market_regime=MarketRegime.BULLISH,
        instrument=inst,
        quote=quote,
        config=cfg.signal,
        is_entry_window=True,
    )

    assert eval_res.decision == Decision.WAIT
    assert "STALE_QUOTE" in eval_res.reason_codes
    assert "MISSING_QUOTE_TIMESTAMP" in eval_res.reason_codes
    missing_gates = [g for g in eval_res.gates if g.status == GateStatus.FAIL and g.data.get("code") == "MISSING_QUOTE_TIMESTAMP"]
    assert len(missing_gates) == 1


def test_exiting_phase_rejected_order_clears_and_auto_retries():
    """In EXITING phase, a REJECTED exit order is not swallowed by hasattr(request); it clears exit_order_id and retries immediately."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 15, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("RELIANCE")
    pos.side = Side.BUY
    pos.quantity = 25
    pos.average_entry = 2500.0
    pos.initial_stop_price = 2480.0
    pos.current_stop_price = 2480.0
    pos.phase = PositionPhase.EXITING
    pos.exit_order_id = "EXIT_REJ_1"

    mock_broker.get_positions.return_value = [PositionState(symbol="RELIANCE", quantity=25, average_entry=2500.0)]
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="RELIANCE", exchange="NSE", lot_size=1, tick_size=0.05)
    mock_broker.get_quote.return_value = Quote(symbol="RELIANCE", exchange="NSE", last_price=2520.0, bid=2519.8, ask=2520.2, volume=50000, timestamp=now, is_fresh=True)

    # Broker returns REJECTED with request.quantity > 0 and filled_quantity = 0
    rej_order = OrderState(
        order_id="EXIT_REJ_1",
        request=OrderRequest(symbol="RELIANCE", side=Side.SELL, quantity=25),
        status=OrderStatus.REJECTED,
        filled_quantity=0,
        pending_quantity=25,
    )
    mock_broker.get_order_status.return_value = rej_order

    placed_orders = []
    def on_place(req):
        ord_id = f"RETRY_EXT_{len(placed_orders) + 1}"
        placed_orders.append((ord_id, req))
        return ord_id
    mock_broker.place_order.side_effect = on_place

    engine.evaluate_step(now)

    # 1. New retry order must have been dispatched on this cycle
    assert len(placed_orders) == 1, "Should immediately dispatch retry market exit order"
    retry_id, retry_req = placed_orders[0]
    assert retry_req.order_type in (OrderType.LIMIT, OrderType.MARKET)
    assert retry_req.quantity == 25
    assert retry_req.side == Side.SELL

    # 2. Position's exit_order_id must be updated to the new retry ID
    assert pos.exit_order_id == retry_id
    assert pos.phase == PositionPhase.EXITING


def test_step11_detects_cancelled_protective_stop_and_liquidates_immediately():
    """In Step 11.0, a CANCELLED/REJECTED protective stop marks pos OPEN_UNPROTECTED and immediately fires emergency market liquidation."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 20, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("JSWSTEEL")
    pos.side = Side.BUY
    pos.quantity = 30
    pos.average_entry = 800.0
    pos.initial_stop_price = 794.0
    pos.current_stop_price = 794.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_JSW_1"

    mock_broker.get_positions.return_value = [PositionState(symbol="JSWSTEEL", quantity=30, average_entry=800.0)]
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="JSWSTEEL", exchange="NSE", lot_size=1, tick_size=0.05)
    mock_broker.get_quote.return_value = Quote(symbol="JSWSTEEL", exchange="NSE", last_price=805.0, bid=804.8, ask=805.2, volume=30000, timestamp=now, is_fresh=True)

    # Stop order was cancelled externally
    cancelled_sl = OrderState(
        order_id="SL_JSW_1",
        request=OrderRequest(symbol="JSWSTEEL", side=Side.SELL, quantity=30, order_type=OrderType.SL),
        status=OrderStatus.CANCELLED,
        filled_quantity=0,
        pending_quantity=0,
    )
    mock_broker.get_order_status.return_value = cancelled_sl

    placed_orders = []
    def on_place(req):
        placed_orders.append(req)
        return "EMER_FLT_1"
    mock_broker.place_order.side_effect = on_place

    engine.evaluate_step(now)

    # Emergency liquidation order must have been placed
    assert len(placed_orders) == 1, "Should fire emergency market liquidation"
    assert placed_orders[0].order_type in (OrderType.LIMIT, OrderType.MARKET)
    assert placed_orders[0].quantity == 30
    assert placed_orders[0].side == Side.SELL
    assert pos.phase == PositionPhase.EXITING
    assert pos.exit_order_id == "EMER_FLT_1"


def test_daily_loss_breach_flattens_even_with_resting_unfilled_entering_order():
    """When daily loss is breached, flatten sequence executes even if an unfilled order is in ENTERING phase (quantity==0)."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 30, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Position in ENTERING with quantity == 0
    pos = engine.state_machine.get_or_create_position("TRENT")
    pos.side = Side.BUY
    pos.quantity = 0
    pos.entry_order_id = "ORD_ENT_1"
    pos.phase = PositionPhase.ENTERING

    mock_broker.get_positions.return_value = []
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)

    import apex3.strategy.engine as eng_module
    orig_eval_risk = eng_module.evaluate_portfolio_risk
    try:
        mock_risk = MagicMock(is_daily_loss_breached=True, is_hard_loss_breached=True, gate=MagicMock(reason="Daily loss limit hit"))
        eng_module.evaluate_portfolio_risk = MagicMock(return_value=mock_risk)
        mock_reconciler_flatten = MagicMock(return_value=(True, "Flattened"))
        engine.reconciler.execute_flatten_sequence = mock_reconciler_flatten

        decisions = engine.evaluate_step(now)

        # execute_flatten_sequence must have been called
        assert mock_reconciler_flatten.called, "execute_flatten_sequence should be called for ENTERING phase"
        halt_decisions = [d for d in decisions if d.decision == Decision.HALT and "DAILY_LOSS_LIMIT_BREACHED" in d.reason_codes]
        assert len(halt_decisions) == 1
    finally:
        eng_module.evaluate_portfolio_risk = orig_eval_risk


def test_risk_positions_populates_protective_stop_and_normalizes_short():
    """risk_positions correctly normalizes signed short positions and populates protective_stop_price from managed position."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 30, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Short position registered in state machine with current_stop_price
    pos = engine.state_machine.get_or_create_position("HINDALCO")
    pos.side = Side.SELL
    pos.quantity = 15
    pos.average_entry = 600.0
    pos.current_stop_price = 612.0
    pos.initial_stop_price = 612.0
    pos.phase = PositionPhase.OPEN_PROTECTED

    # Broker returns signed short quantity (-15) with protective_stop_price = 0.0
    mock_broker.get_positions.return_value = [
        PositionState(
            symbol="HINDALCO",
            side=Side.SELL,
            quantity=-15,
            average_entry=600.0,
            protective_stop_price=0.0,
        )
    ]
    mock_broker.get_quote.return_value = Quote(symbol="HINDALCO", exchange="NSE", last_price=595.0, bid=594.8, ask=595.2, volume=10000, timestamp=now, is_fresh=True)

    import apex3.strategy.engine as eng_module
    orig_eval_risk = eng_module.evaluate_portfolio_risk
    captured_positions = []
    def capture_risk(open_positions, **kwargs):
        captured_positions.extend(open_positions)
        return MagicMock(is_daily_loss_breached=False, is_hard_loss_breached=False, gate=MagicMock(reason="OK"))

    try:
        eng_module.evaluate_portfolio_risk = MagicMock(side_effect=capture_risk)
        engine.evaluate_step(now)

        assert len(captured_positions) == 1
        risk_pos = captured_positions[0]
        assert risk_pos.quantity == 15, "Quantity must be absolute positive"
        assert risk_pos.side == Side.SELL
        assert risk_pos.protective_stop_price == 612.0, "protective_stop_price must be populated from managed_pos"
        assert risk_pos.unrealized_pnl == (600.0 - 595.0) * 15  # Short profit: (600 - 595) * 15 = +75
    finally:
        eng_module.evaluate_portfolio_risk = orig_eval_risk


def test_reconciler_handles_negative_short_quantities():
    """reconciler.audit_startup_state and execute_flatten_sequence handle negative short quantities from broker."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    sm = StateMachine("TEST_SESSION")
    reconciler = Reconciler(mock_broker, sm, mock_journal, "TEST_SESSION")

    # 1. Startup audit with short position (quantity = -20)
    broker_positions = [
        PositionState(symbol="TATASTEEL", quantity=-20, average_entry=150.0, protective_stop_price=153.0)
    ]
    mock_broker.get_positions.return_value = broker_positions
    mock_broker.get_orders.return_value = [
        OrderState(
            order_id="SL_TATA_1",
            request=OrderRequest(symbol="TATASTEEL", side=Side.BUY, quantity=20, order_type=OrderType.SL, tag="TEST_SESSION_TATA_SL"),
            status=OrderStatus.TRIGGER_PENDING,
            pending_quantity=20,
        )
    ]

    status = reconciler.audit_startup_state(["TATASTEEL"])
    assert status.is_clean
    assert "TATASTEEL" in sm.positions
    managed = sm.positions["TATASTEEL"]
    assert managed.quantity == 20
    assert managed.side == Side.SELL
    assert managed.phase == PositionPhase.OPEN_PROTECTED

    # 2. Flatten sequence dispatches BUY exit for short position
    placed_orders = []
    def on_place(req):
        placed_orders.append(req)
        return "FLT_ORD_1"
    mock_broker.place_order.side_effect = on_place

    reconciler.execute_flatten_sequence(timeout_sec=0.1)
    assert len(placed_orders) == 1
    assert placed_orders[0].side == Side.BUY, "Exit for short position must be BUY"
    assert placed_orders[0].quantity == 20


def test_sizer_accepts_broker_leverage_within_float_tolerance():
    """size_position accepts broker leverage within float tolerance (e.g. 3.999x with min_accepted_leverage=4.0)."""
    from apex3.strategy.sizer import size_position
    from apex3.config import LimitsConfig, RiskConfig

    limits = LimitsConfig(max_symbols=2, max_leverage=4.0, min_accepted_leverage=4.0)
    risk = RiskConfig()
    margin = MarginInfo(equity_available=15000.0, net_value=15000.0)
    inst = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05)
    # Estimate has 3.999x leverage
    est = OrderMarginEstimate(required_margin=3000.0, leverage_available=3.999)

    res = size_position(
        symbol="TRENT",
        side=Side.BUY,
        entry_price=1000.0,
        stop_price=993.0,
        num_qualified_symbols=1,
        margin_info=margin,
        instrument=inst,
        limits_config=limits,
        risk_config=risk,
        order_margin_estimate=est,
    )
    assert res.is_valid, f"Sizing should pass with 3.999x leverage, got {res.rejection_code}: {res.rejection_reason}"
    assert res.quantity > 0


def test_flatten_reports_incomplete_if_strategy_order_remains_pending():
    """P0.2: Flatten reports incomplete and logs EMERGENCY if a strategy order remains unconfirmed/pending."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    sm = StateMachine("APEX3")
    reconciler = Reconciler(mock_broker, sm, mock_journal, "APEX3")

    pos = sm.get_or_create_position("INFY")
    pos.side = Side.BUY
    pos.quantity = 10
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_INFY_1"

    # Broker has 10 shares open initially
    broker_positions = [PositionState(symbol="INFY", side=Side.BUY, quantity=10, average_entry=1500.0, product=Product.MIS)]
    mock_broker.get_positions.side_effect = lambda: list(broker_positions)
    mock_broker.get_orders.return_value = [
        OrderState(
            order_id="SL_INFY_1",
            request=OrderRequest(symbol="INFY", side=Side.SELL, quantity=10, tag="APEX3_INFY_SL"),
            status=OrderStatus.TRIGGER_PENDING,
            pending_quantity=10,
        )
    ]

    # When exit order placed, position clears at broker
    def on_exit(req):
        broker_positions.clear()
        return "EXIT_INFY_ORD"
    mock_broker.place_order.side_effect = on_exit

    # But SL order cancellation never confirms (remains TRIGGER_PENDING)
    def on_status(oid):
        if oid == "SL_INFY_1":
            return OrderState(order_id="SL_INFY_1", status=OrderStatus.TRIGGER_PENDING, pending_quantity=10)
        return OrderState(order_id=oid, status=OrderStatus.COMPLETE, filled_quantity=10)
    mock_broker.get_order_status.side_effect = on_status

    success, msg = reconciler.execute_flatten_sequence(timeout_sec=0.2)
    assert success is False
    assert "Residual open exposure" in msg or "unconfirmed" in msg.lower()
    assert "SL_INFY_1" in msg
    # Verify emergency warning logged
    mock_journal.log_warning.assert_called()
    assert any(call.kwargs.get("severity") == "EMERGENCY" for call in mock_journal.log_warning.call_args_list)


def test_flatten_ignores_non_mis_delivery_holdings():
    """P1.4 / P0.2: Flatten sequence must never liquidate non-MIS (CNC/NRML) delivery holdings."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    sm = StateMachine("APEX3")
    reconciler = Reconciler(mock_broker, sm, mock_journal, "APEX3")

    # Long-term CNC delivery holding in account, but no MIS positions
    cnc_holding = PositionState(symbol="RELIANCE", side=Side.BUY, quantity=50, average_entry=2000.0, product=Product.CNC)
    mock_broker.get_positions.return_value = [cnc_holding]
    mock_broker.get_orders.return_value = []
    mock_broker.get_order_status.return_value = OrderState(status=OrderStatus.COMPLETE)

    success, msg = reconciler.execute_flatten_sequence(timeout_sec=0.1)
    assert success is True
    # Crucially: place_order was never called to dump CNC holdings!
    assert mock_broker.place_order.call_count == 0


def test_reconciler_startup_audit_ignores_cnc_positions():
    """P1.4: Startup audit must not crash with KeyError or flag CNC delivery holdings as unauthorized MIS."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    sm = StateMachine("APEX3")
    reconciler = Reconciler(mock_broker, sm, mock_journal, "APEX3")

    cnc_pos = PositionState(symbol="TCS", side=Side.BUY, quantity=100, average_entry=3500.0, product=Product.CNC)
    mock_broker.get_positions.return_value = [cnc_pos]
    mock_broker.get_orders.return_value = []

    status = reconciler.audit_startup_state(["INFY", "TRENT"])
    assert status.is_clean is True
    assert "TCS" not in sm.positions


def test_exiting_phase_unknown_status_blocks_new_entries_and_defers():
    """P0.3: When exit order status is UNKNOWN, engine treats it as reconciliation failure and defers duplicate exit."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 30, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("INFY")
    pos.side = Side.BUY
    pos.quantity = 25
    pos.phase = PositionPhase.EXITING
    pos.exit_order_id = "EXIT_ORD_1"

    # Status returns UNKNOWN
    mock_broker.get_order_status.return_value = OrderState(order_id="EXIT_ORD_1", status=OrderStatus.UNKNOWN)
    mock_broker.get_positions.return_value = [
        PositionState(symbol="INFY", side=Side.BUY, quantity=25, product=Product.MIS)
    ]
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []

    decisions = engine.evaluate_step(now)
    # Must NOT place a duplicate exit order while status is unknown!
    assert mock_broker.place_order.call_count == 0
    assert pos.phase == PositionPhase.EXITING
    # Emergency warning logged
    assert any(call.kwargs.get("severity") == "EMERGENCY" for call in mock_journal.log_warning.call_args_list)


def test_step11_unknown_stop_order_status_defers_modification():
    """P0.3: When protective stop status returns UNKNOWN, engine defers modification and alerts."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 30, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("TRENT")
    pos.side = Side.BUY
    pos.quantity = 10
    pos.average_entry = 1000.0
    pos.current_stop_price = 990.0
    pos.initial_stop_price = 990.0
    pos.target_price = 1020.0
    pos.r_value = 10.0
    pos.protective_order_id = "SL_TRENT_1"
    pos.phase = PositionPhase.OPEN_PROTECTED

    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05)
    mock_broker.get_quote.return_value = Quote(symbol="TRENT", exchange="NSE", last_price=1010.5, bid=1010.4, ask=1010.6, volume=5000, timestamp=now, is_fresh=True)
    mock_broker.get_positions.return_value = [
        PositionState(symbol="TRENT", side=Side.BUY, quantity=10, product=Product.MIS)
    ]
    mock_broker.get_order_status.return_value = OrderState(order_id="SL_TRENT_1", status=OrderStatus.UNKNOWN)
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []

    engine.evaluate_step(now)
    # Stop modification must be deferred when stop status is unknown
    assert mock_broker.modify_order.call_count == 0
    assert any(call.kwargs.get("severity") == "EMERGENCY" for call in mock_journal.log_warning.call_args_list)


def test_signal_safe_default_when_circuit_limits_zero():
    """Fix 1: Gate 8 safely defaults to 100.0% circuit distance and passes when instrument circuit limits are 0, and rejects when near circuit."""
    from apex3.strategy.signal import evaluate_signal

    now = datetime(2026, 3, 20, 9, 35, tzinfo=IST)
    inst = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05, upper_circuit=0.0, lower_circuit=0.0, mis_available=True)
    quote = Quote(symbol="TRENT", exchange="NSE", last_price=1000.0, bid=999.8, ask=1000.2, volume=50000, timestamp=now, is_fresh=True)

    # 19 base candles with 10k volume, then 20th candle breaks out above OR high (995) with 50k volume (5x)
    stock_candles = [Candle(timestamp=now - timedelta(minutes=5 * (20 - i)), open=991.0, high=994.0, low=990.0, close=992.0, volume=10000, is_complete=True) for i in range(19)]
    stock_candles.append(Candle(timestamp=now, open=996.0, high=1001.0, low=995.5, close=1000.0, volume=50000, is_complete=True))
    bench_candles = [Candle(timestamp=now - timedelta(minutes=5 * (20 - i)), open=10000.0, high=10005.0, low=9995.0, close=10000.0, volume=10000, is_complete=True) for i in range(20)]
    or_range = OpeningRange(symbol="TRENT", high=995.0, low=990.0, is_complete=True, candle_count=3)

    cfg = AppConfig()
    sig = evaluate_signal(
        symbol="TRENT",
        stock_candles_5m=stock_candles,
        benchmark_candles_5m=bench_candles,
        opening_range=or_range,
        market_regime=MarketRegime.BULLISH,
        instrument=inst,
        quote=quote,
        config=cfg.signal,
        is_entry_window=True,
    )
    # 0.0 circuit limits should safely pass Gate 8 with 100% distance
    assert sig.decision == Decision.PROPOSED_LONG
    assert any(g.name == "CIRCUIT_DISTANCE" and g.status == GateStatus.PASS for g in sig.gates)

    # But when upper circuit is tight (e.g. 1005.0 vs signal_price 1000.0, dist = 0.5% < min 2.0%), it fails closed
    inst_tight = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05, upper_circuit=1005.0, lower_circuit=800.0, mis_available=True)
    sig_tight = evaluate_signal(
        symbol="TRENT",
        stock_candles_5m=stock_candles,
        benchmark_candles_5m=bench_candles,
        opening_range=or_range,
        market_regime=MarketRegime.BULLISH,
        instrument=inst_tight,
        quote=quote,
        config=cfg.signal,
        is_entry_window=True,
    )
    assert sig_tight.decision == Decision.WAIT
    assert "TOO_CLOSE_TO_CIRCUIT" in sig_tight.reason_codes
    assert any(g.name == "CIRCUIT_DISTANCE" and g.status == GateStatus.FAIL for g in sig_tight.gates)


def test_signal_fails_closed_when_mis_unavailable():
    """P1.3: Signal rejects immediately when mis_available is False."""
    from apex3.strategy.signal import evaluate_signal

    now = datetime(2026, 3, 20, 9, 35, tzinfo=IST)
    inst = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05, upper_circuit=1200.0, lower_circuit=800.0, mis_available=False)
    quote = Quote(symbol="TRENT", exchange="NSE", last_price=1000.0, bid=999.8, ask=1000.2, volume=10000, timestamp=now, is_fresh=True)
    candles = [Candle(timestamp=now - timedelta(minutes=5 * (20 - i)), open=1000.0, high=1001.0, low=999.0, close=1000.0, volume=10000, is_complete=True) for i in range(20)]
    or_range = OpeningRange(symbol="TRENT", high=995.0, low=990.0, is_complete=True, candle_count=3)

    cfg = AppConfig()
    sig = evaluate_signal(
        symbol="TRENT",
        stock_candles_5m=candles,
        benchmark_candles_5m=candles,
        opening_range=or_range,
        market_regime=MarketRegime.BULLISH,
        instrument=inst,
        quote=quote,
        config=cfg.signal,
        is_entry_window=True,
    )
    assert sig.decision == Decision.WAIT
    assert "MIS_NOT_AVAILABLE" in sig.reason_codes
    assert any(g.name == "PRODUCT_ELIGIBILITY" and g.status == GateStatus.FAIL for g in sig.gates)


def test_calendar_freshness_validation():
    """P1.6: 2026-01-15 is included in holidays, and validate_calendar_freshness guards against stale calendars."""
    from apex3.strategy.schedule import SessionScheduler
    from apex3.config import load_config
    import os

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    assert "2026-01-15" in cfg.schedule.holidays

    sched = SessionScheduler(cfg.schedule)
    is_valid, msg = sched.validate_calendar_freshness(2026)
    assert is_valid is True

    # 2027 calendar is unconfigured and must fail
    is_valid_2027, msg_2027 = sched.validate_calendar_freshness(2027)
    assert is_valid_2027 is False
    assert "No exchange holidays configured for year 2027" in msg_2027


def test_audit_startup_state_cancels_orphaned_unfilled_entry():
    """Fix 1: In audit_startup_state, orphaned unfilled entry order is cancelled and logged."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    sm = StateMachine("APEX3")
    reconciler = Reconciler(mock_broker, sm, mock_journal, session_tag="APEX3")

    mock_broker.get_positions.return_value = []
    orphan_order = OrderState(
        order_id="ENTRY_ORPHAN_1",
        request=OrderRequest(
            symbol="INFY",
            side=Side.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=1500.0,
            tag="APEX3_INFY_E0935",
        ),
        status=OrderStatus.OPEN,
        filled_quantity=0,
    )
    mock_broker.get_orders.return_value = [orphan_order]

    status = reconciler.audit_startup_state(["INFY"])
    assert status.is_clean is True
    # Order was cancelled immediately
    mock_broker.cancel_order.assert_called_once_with("ENTRY_ORPHAN_1")
    # Position was NOT left in ENTERING
    assert "INFY" not in sm.positions
    # Event logged to journal
    assert any(
        call.kwargs.get("event_type") == "CANCELLED_ORPHAN_ENTRY_ON_STARTUP"
        for call in mock_journal.log_event.call_args_list
    )


def test_entry_slippage_exceeded_live_gap():
    """Fix 2: Reject entry if live ask/bid has gapped past cap_price."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 9, 35, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_positions.return_value = []
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []

    inst = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05, upper_circuit=1500.0, lower_circuit=800.0, mis_available=True)
    mock_broker.get_instrument_info.return_value = inst
    engine.set_manual_symbols(["TRENT"])
    engine.opening_ranges["TRENT"] = OpeningRange(symbol="TRENT", high=1000.0, low=990.0, is_complete=True, candle_count=3)

    # Quote where ask has gapped above cap_price
    quote = Quote(symbol="TRENT", exchange="NSE", last_price=1005.0, bid=1004.8, ask=1005.0, volume=50000, timestamp=now, is_fresh=True)
    mock_broker.get_quote.return_value = quote

    candles_5m = [
        Candle(timestamp=now - timedelta(minutes=5 * (25 - i)), open=990.0, high=1000.0, low=985.0, close=995.0, volume=10000, is_complete=True)
        for i in range(25)
    ]
    mock_broker.get_historical.return_value = candles_5m

    with patch("apex3.strategy.engine.evaluate_signal") as mock_eval_sig, \
         patch("apex3.strategy.engine.evaluate_market_regime") as mock_regime:
        mock_regime.return_value = MagicMock(regime=MarketRegime.BULLISH)
        mock_eval_sig.return_value = SignalEvaluation(
            symbol="TRENT",
            decision=Decision.PROPOSED_LONG,
            side=Side.BUY,
            gates=[],
            reason_codes=["BREAKOUT_CONFIRMED"],
            explanation="Test breakout",
            signal_price=1000.0,
        )

        decisions = engine.evaluate_step(now)
        wait_decisions = [d for d in decisions if "ENTRY_SLIPPAGE_EXCEEDED" in d.reason_codes]
        assert len(wait_decisions) >= 1
        assert wait_decisions[0].decision == Decision.WAIT


def test_step11_fee_lock_confirmed_when_order_status_open():
    """Fix 3: When stop modification is OPEN (market touched limit), stop is confirmed moved."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 11, 0, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("INFY")
    pos.side = Side.BUY
    pos.quantity = 10
    pos.average_entry = 1000.0
    pos.initial_stop_price = 990.0
    pos.current_stop_price = 990.0
    pos.r_value = 10.0
    pos.target_price = 1020.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_ORD_1"
    pos.fee_lock_applied = False

    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_positions.return_value = [
        PositionState(symbol="INFY", side=Side.BUY, quantity=10, average_entry=1000.0, product=Product.MIS)
    ]
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    inst = InstrumentInfo(symbol="INFY", exchange="NSE", lot_size=1, tick_size=0.05, upper_circuit=1200.0, lower_circuit=800.0, mis_available=True)
    mock_broker.get_instrument_info.return_value = inst
    quote = Quote(symbol="INFY", exchange="NSE", last_price=1016.50, bid=1016.4, ask=1016.6, volume=10000, timestamp=now, is_fresh=True)
    mock_broker.get_quote.return_value = quote

    mock_broker.modify_order.return_value = True
    mock_broker.get_order_status.return_value = OrderState(order_id="SL_ORD_1", status=OrderStatus.OPEN)

    decisions = engine.evaluate_step(now)
    assert pos.fee_lock_applied is True
    assert pos.current_stop_price > 990.0
    assert pos.stop_modification_count == 1


def test_exiting_phase_cancelled_exit_with_zero_quantity_closes_position():
    """Fix 4: When exit order was cancelled/rejected but 100% filled, phase is CLOSED."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 30, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("INFY")
    pos.side = Side.BUY
    pos.quantity = 25
    pos.phase = PositionPhase.EXITING
    pos.exit_order_id = "EXIT_ORD_1"

    # Exit order was cancelled after 25 shares filled
    mock_broker.get_order_status.return_value = OrderState(
        order_id="EXIT_ORD_1",
        status=OrderStatus.CANCELLED,
        filled_quantity=25,
        pending_quantity=0,
    )
    mock_broker.get_positions.return_value = []
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []

    decisions = engine.evaluate_step(now)
    assert pos.phase == PositionPhase.CLOSED
    assert pos.quantity == 0
    assert pos.exit_order_id == ""


def test_hybrid_preflight_fails_closed_if_angel_or_zerodha_invalid():
    """P0.5: Live preflight requires both Angel and Zerodha live sessions to be valid."""
    from apex3.broker.hybrid_adapter import HybridBrokerAdapter

    mock_data = MagicMock()
    mock_exec = MagicMock()
    adapter = HybridBrokerAdapter(data_provider=mock_data, execution_broker=mock_exec)

    # 1. Angel valid, Zerodha invalid -> False
    mock_data.is_session_valid.return_value = True
    mock_exec.is_session_valid.return_value = False
    assert adapter.is_session_valid() is False

    # 2. Angel invalid, Zerodha valid -> False
    mock_data.is_session_valid.return_value = False
    mock_exec.is_session_valid.return_value = True
    assert adapter.is_session_valid() is False

    # 3. Both valid -> True
    mock_data.is_session_valid.return_value = True
    mock_exec.is_session_valid.return_value = True
    assert adapter.is_session_valid() is True


def test_angel_quote_marketdata_full_depth_and_exchange_feed_timestamp():
    """P0.1: Angel provider parses FULL mode market depth and exchange feed timestamp."""
    from apex3.broker.angel_data_provider import AngelDataProvider

    provider = AngelDataProvider(api_key="", client_code="", pin="", totp_secret="")
    mock_smart = MagicMock()
    provider._smart_api = mock_smart
    provider._auth_token = "test_jwt"
    provider._token_cache["TRENT"] = "1964"

    now_feed = datetime.now(IST)
    mock_smart.getMarketData.return_value = {
        "status": True,
        "data": {
            "fetched": [
                {
                    "tradingSymbol": "TRENT-EQ",
                    "ltp": 1000.50,
                    "tradeVolume": 50000,
                    "exchFeedTime": now_feed.strftime("%d-%b-%Y %H:%M:%S"),
                    "upperCircuit": 1100.0,
                    "lowerCircuit": 900.0,
                    "depth": {
                        "buy": [{"price": 1000.40, "quantity": 100, "orders": 5}],
                        "sell": [{"price": 1000.60, "quantity": 150, "orders": 3}],
                    },
                }
            ]
        },
    }

    quote = provider.get_quote("TRENT")
    assert quote.last_price == 1000.50
    assert quote.bid == 1000.40
    assert quote.ask == 1000.60
    assert quote.is_fresh is True
    assert quote.timestamp is not None
    assert quote.timestamp.hour == now_feed.hour and quote.timestamp.minute == now_feed.minute

    # Check instrument info cached circuit limits
    inst = provider.get_instrument_info("TRENT")
    assert inst.upper_circuit == 1100.0
    assert inst.lower_circuit == 900.0


def test_angel_quote_missing_or_inverted_depth_fails_freshness():
    """P0.1: Missing or inverted bid/ask depth marks quote as not fresh."""
    from apex3.broker.angel_data_provider import AngelDataProvider

    provider = AngelDataProvider(api_key="", client_code="", pin="", totp_secret="")
    mock_smart = MagicMock()
    provider._smart_api = mock_smart
    provider._auth_token = "test_jwt"
    provider._token_cache["TRENT"] = "1964"

    # Inverted depth: ask < bid
    mock_smart.getMarketData.return_value = {
        "status": True,
        "data": {
            "fetched": [
                {
                    "tradingSymbol": "TRENT-EQ",
                    "ltp": 1000.50,
                    "tradeVolume": 50000,
                    "exchFeedTime": "27-Oct-2026 10:15:30",
                    "depth": {
                        "buy": [{"price": 1001.0, "quantity": 100, "orders": 5}],
                        "sell": [{"price": 999.0, "quantity": 150, "orders": 3}],
                    },
                }
            ]
        },
    }

    quote = provider.get_quote("TRENT")
    assert quote.is_fresh is False


def test_target_exit_defers_when_stop_cancel_unconfirmed():
    """P0.2: Target exit defers sending exit order when stop cancellation is unconfirmed."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    now = datetime(2026, 3, 20, 10, 0, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("INFY")
    pos.side = Side.BUY
    pos.quantity = 10
    pos.average_entry = 1000.0
    pos.target_price = 1020.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_ORD_1"

    # Quote reaches target price
    mock_broker.get_quote.return_value = Quote(
        symbol="INFY", exchange="NSE", last_price=1021.0, bid=1020.9, ask=1021.1, volume=10000, timestamp=now, is_fresh=True
    )
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="INFY", exchange="NSE", tick_size=0.05)

    # Broker cancel returns True, but order status is still TRIGGER_PENDING (unconfirmed terminal)
    mock_broker.cancel_order.return_value = True
    mock_broker.get_order_status.return_value = OrderState(
        order_id="SL_ORD_1",
        status=OrderStatus.TRIGGER_PENDING,
        pending_quantity=10,
        filled_quantity=0,
    )

    placed_orders = []
    mock_broker.place_order.side_effect = lambda req: placed_orders.append(req) or "EXIT_ORD_1"

    decisions = engine.evaluate_step(now)

    # Cancel was requested
    assert mock_broker.cancel_order.call_count >= 1
    # Exit order was NOT placed because stop is still TRIGGER_PENDING!
    assert len(placed_orders) == 0
    assert pos.phase == PositionPhase.OPEN_PROTECTED


def test_target_exit_handles_stop_loss_hit_if_complete_during_cancel():
    """P0.2: If protective stop completes at exchange during cancel request, handle as STOP_LOSS_HIT without sending opposite exit."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    now = datetime(2026, 3, 20, 10, 0, tzinfo=IST)

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    pos = engine.state_machine.get_or_create_position("INFY")
    pos.side = Side.BUY
    pos.quantity = 10
    pos.average_entry = 1000.0
    pos.target_price = 1020.0
    pos.phase = PositionPhase.OPEN_PROTECTED
    pos.protective_order_id = "SL_ORD_1"

    mock_broker.get_quote.return_value = Quote(
        symbol="INFY", exchange="NSE", last_price=1021.0, bid=1020.9, ask=1021.1, volume=10000, timestamp=now, is_fresh=True
    )
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="INFY", exchange="NSE", tick_size=0.05)

    # Broker cancel sent, but stop filled (COMPLETE)
    mock_broker.cancel_order.return_value = True
    mock_broker.get_order_status.return_value = OrderState(
        order_id="SL_ORD_1",
        status=OrderStatus.COMPLETE,
        pending_quantity=0,
        filled_quantity=10,
        average_price=990.0,
    )

    placed_orders = []
    mock_broker.place_order.side_effect = lambda req: placed_orders.append(req) or "EXIT_ORD_1"

    decisions = engine.evaluate_step(now)

    # Never placed opposite exit order!
    assert len(placed_orders) == 0
    assert pos.phase == PositionPhase.CLOSED
    assert pos.quantity == 0


def test_flatten_fails_closed_when_get_orders_raises():
    """P0.4: Flatten fails closed and returns False if get_orders raises an exception."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    sm = StateMachine("APEX3")
    rec = Reconciler(broker=mock_broker, state_machine=sm, journal=mock_journal, session_tag="APEX3")

    mock_broker.get_orders.side_effect = RuntimeError("Broker connection timeout")

    ok, msg = rec.execute_flatten_sequence(timeout_sec=1.0)
    assert ok is False
    assert "CRITICAL: Could not retrieve orders during flatten" in msg
    mock_journal.log_warning.assert_called()


def test_exit_order_builder_raises_value_error_instead_of_100_dummy_price():
    """P1.1: Exit order builder raises ValueError if no valid quote price is available."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Quote missing / zero
    mock_broker.get_quote.return_value = Quote(
        symbol="INFY", exchange="NSE", last_price=0.0, bid=0.0, ask=0.0, volume=0, timestamp=datetime.now(IST)
    )
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="INFY", exchange="NSE", tick_size=0.05)

    with pytest.raises(ValueError, match="valid price quote is required"):
        engine._create_exit_order_request("INFY", Side.SELL, 10, "FLT", current_price=0.0)


def test_rate_limiter_wired_in_kite_adapter():
    """P1.3: RateLimiter tokens are acquired on order, quote, and historical requests."""
    from apex3.broker.kite_adapter import KiteAdapter
    from apex3.broker.rate_limiter import RateLimiter

    limiter = RateLimiter(rates={"orders": 10.0, "quotes": 10.0, "historical": 10.0})
    adapter = KiteAdapter(api_key="test_key", access_token="test_token", live_enabled=False, rate_limiter=limiter)
    adapter._kite = MagicMock()
    adapter._kite.quote.return_value = {
        "NSE:INFY": {"last_price": 1000.0, "depth": {"buy": [{"price": 999.8}], "sell": [{"price": 1000.2}]}}
    }

    # Calling get_quote should acquire a quote token
    with patch.object(limiter, "acquire", wraps=limiter.acquire) as mock_acq:
        q = adapter.get_quote("INFY")
        assert mock_acq.call_count >= 1
        assert mock_acq.call_args[0][0] == "quotes"


def test_unknown_kite_order_type_and_product_mapping():
    """P1.4: Unrecognized Kite order types and products map to UNKNOWN instead of LIMIT/MIS."""
    from apex3.broker.kite_adapter import _map_kite_order_type, KiteAdapter

    # Order type
    assert _map_kite_order_type("ICEBERG") == OrderType.UNKNOWN
    assert _map_kite_order_type("LIMIT") == OrderType.LIMIT

    # Product
    adapter = KiteAdapter(api_key="k", access_token="t", live_enabled=False)
    adapter._kite = MagicMock()
    adapter._kite.positions.return_value = {
        "net": [
            {"tradingsymbol": "INFY", "quantity": 10, "average_price": 1000.0, "product": "CO"},
            {"tradingsymbol": "TCS", "quantity": 5, "average_price": 3000.0, "product": "MIS"},
        ]
    }
    pos_list = adapter.get_positions()
    infy_pos = next(p for p in pos_list if p.symbol == "INFY")
    tcs_pos = next(p for p in pos_list if p.symbol == "TCS")
    assert infy_pos.product == Product.UNKNOWN
    assert tcs_pos.product == Product.MIS


def test_index_freshness_passes_without_exchange_depth():
    """P0: Indices (NIFTY 50) have 0 bid/ask depth on exchange; valid feed time + LTP must yield is_fresh=True."""
    from apex3.broker.angel_data_provider import AngelDataProvider
    provider = AngelDataProvider(api_key="", client_code="", pin="", totp_secret="")
    now_feed = datetime.now(IST)

    item = {
        "tradingSymbol": "Nifty 50",
        "symbolToken": "99926000",
        "ltp": 24500.25,
        "tradeVolume": 0,
        "exchFeedTime": now_feed.strftime("%d-%b-%Y %H:%M:%S"),
        "depth": {"buy": [], "sell": []},
    }

    quote = provider._parse_market_item(item, "NIFTY 50", "NSE")
    assert quote.last_price == 24500.25
    assert quote.bid == 0.0
    assert quote.ask == 0.0
    assert quote.is_fresh is True


def test_angel_quote_fails_closed_when_feed_time_missing_or_stale():
    """P1: Angel provider marks quote is_fresh=False if exchFeedTime is missing or older than 2.0s."""
    from apex3.broker.angel_data_provider import AngelDataProvider
    provider = AngelDataProvider(api_key="", client_code="", pin="", totp_secret="")

    # Missing feed timestamp
    item_missing_ts = {
        "tradingSymbol": "INFY-EQ",
        "symbolToken": "1594",
        "ltp": 1800.0,
        "depth": {
            "buy": [{"price": 1799.9, "quantity": 100}],
            "sell": [{"price": 1800.1, "quantity": 100}],
        },
    }
    q1 = provider._parse_market_item(item_missing_ts, "INFY", "NSE")
    assert q1.is_fresh is False

    # Stale timestamp (20 seconds ago, exceeding 15.0s threshold)
    old_time = datetime.now(IST) - timedelta(seconds=20)
    item_stale_ts = {
        "tradingSymbol": "INFY-EQ",
        "symbolToken": "1594",
        "ltp": 1800.0,
        "exchFeedTime": old_time.strftime("%d-%b-%Y %H:%M:%S"),
        "depth": {
            "buy": [{"price": 1799.9, "quantity": 100}],
            "sell": [{"price": 1800.1, "quantity": 100}],
        },
    }
    q2 = provider._parse_market_item(item_stale_ts, "INFY", "NSE")
    assert q2.is_fresh is False


def test_batch_quotes_snapshot_used_in_scan_and_propose():
    """P1: Engine evaluate_step calls broker.get_quotes_batch() for all watchlist symbols in one call."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_positions.return_value = []
    cfg = AppConfig()
    object.__setattr__(cfg.watchlist, "symbols", ["RELIANCE", "INFY"])
    object.__setattr__(cfg.watchlist, "scanner_enabled", False)
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    engine.opening_ranges["RELIANCE"] = OpeningRange(symbol="RELIANCE", is_complete=True, high=2500.0, low=2480.0, candle_count=3)
    engine.opening_ranges["INFY"] = OpeningRange(symbol="INFY", is_complete=True, high=1800.0, low=1780.0, candle_count=3)

    now = datetime(2026, 3, 23, 9, 35, 10, tzinfo=IST)
    bench_q = Quote(symbol="NIFTY 50", exchange="NSE", last_price=24500.0, bid=0.0, ask=0.0, volume=0, timestamp=now, is_fresh=True)
    rel_q = Quote(symbol="RELIANCE", exchange="NSE", last_price=2500.0, bid=2499.8, ask=2500.2, volume=1000, timestamp=now, is_fresh=True)
    infy_q = Quote(symbol="INFY", exchange="NSE", last_price=1800.0, bid=1799.8, ask=1800.2, volume=1000, timestamp=now, is_fresh=True)

    batch_map = {"NIFTY 50": bench_q, "RELIANCE": rel_q, "INFY": infy_q}
    mock_broker.get_quotes_batch.return_value = batch_map
    mock_broker.get_historical.return_value = [
        Candle(timestamp=now - timedelta(minutes=5 * (15 - i)), open=100.0, high=101.0, low=99.0, close=100.0, volume=1000, is_complete=True)
        for i in range(15)
    ]
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="RELIANCE", tick_size=0.05, lot_size=1)

    import apex3.strategy.engine as eng_module
    orig_eval_sig = eng_module.evaluate_signal
    orig_regime = eng_module.evaluate_market_regime
    try:
        eng_module.evaluate_market_regime = MagicMock(return_value=MagicMock(regime=MarketRegime.BULLISH, is_valid=True, gate=GateResult("REGIME", GateStatus.PASS, "OK")))
        eng_module.evaluate_signal = MagicMock(return_value=SignalEvaluation(
            symbol="RELIANCE",
            decision=Decision.WAIT,
            signal_price=2500.0,
            side=Side.BUY,
            gates=[GateResult("TEST", GateStatus.FAIL, "WAIT")],
        ))

        engine.evaluate_step(now)

        # Verify get_quotes_batch was called with benchmark + watchlist
        assert mock_broker.get_quotes_batch.call_count >= 1
        call_symbols = mock_broker.get_quotes_batch.call_args[0][0]
        assert "NIFTY 50" in call_symbols
        assert "RELIANCE" in call_symbols
        assert "INFY" in call_symbols
    finally:
        eng_module.evaluate_signal = orig_eval_sig
        eng_module.evaluate_market_regime = orig_regime


def test_hybrid_adapter_queries_zerodha_for_mis_and_tick_size():
    """P1: HybridBrokerAdapter consults execution_broker (Zerodha) first for MIS eligibility, tick size, and lot size."""
    from apex3.broker.hybrid_adapter import HybridBrokerAdapter
    mock_data = MagicMock()
    mock_exec = MagicMock()
    adapter = HybridBrokerAdapter(data_provider=mock_data, execution_broker=mock_exec)

    # Zerodha returns verified instrument info
    mock_exec.get_instrument_info.return_value = InstrumentInfo(
        symbol="TRENT",
        exchange="NSE",
        tick_size=0.05,
        lot_size=1,
        mis_available=True,
        instrument_token="12345",
    )
    # Angel returns token and circuit limits
    mock_data.resolve_token.return_value = "1964"
    mock_data.get_instrument_info.return_value = InstrumentInfo(
        symbol="TRENT",
        exchange="NSE",
        upper_circuit=1100.0,
        lower_circuit=900.0,
    )

    info = adapter.get_instrument_info("TRENT")
    assert info.mis_available is True
    assert info.tick_size == 0.05
    assert info.instrument_token == "12345"
    assert info.upper_circuit == 1100.0
    assert info.lower_circuit == 900.0
    assert info.exchange_token == "1964"

    # When Zerodha fails, fail safe with mis_available=False
    mock_exec.get_instrument_info.side_effect = RuntimeError("Zerodha down")
    info_fail = adapter.get_instrument_info("TRENT")
    assert info_fail.mis_available is False


def test_partial_fill_late_fill_modifies_stop_quantity():
    """P0: If additional shares fill during cancel polling, stop order quantity is modified via modify_order."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TRENT", tick_size=0.05, lot_size=1)

    pos = engine.state_machine.get_or_create_position("TRENT")
    pos.symbol = "TRENT"
    pos.phase = PositionPhase.ENTERING
    pos.entry_order_id = "ENTRY_TRENT"
    pos.side = Side.BUY
    pos.initial_stop_price = 990.0
    pos.protective_order_id = "SL_TRENT_1"

    # First poll: partial fill 5 shares
    def on_status(oid):
        if oid == "ENTRY_TRENT":
            return OrderState(
                order_id="ENTRY_TRENT",
                status=OrderStatus.OPEN,
                filled_quantity=5,
                average_price=1000.0,
            )
        elif oid == "SL_TRENT_1":
            return OrderState(
                order_id="SL_TRENT_1",
                request=OrderRequest(symbol="TRENT", side=Side.SELL, quantity=5, order_type=OrderType.SL),
                status=OrderStatus.TRIGGER_PENDING,
                pending_quantity=5,
            )
        return None
    mock_broker.get_order_status.side_effect = on_status

    # Terminal state polling during cancellation: late fill brings total to 8 shares before CANCELLED
    engine._confirm_order_terminal = MagicMock(return_value=OrderState(
        order_id="ENTRY_TRENT",
        status=OrderStatus.CANCELLED,
        filled_quantity=8,
        average_price=1000.0,
    ))
    mock_broker.get_positions.return_value = [
        PositionState(symbol="TRENT", side=Side.BUY, quantity=8, average_entry=1000.0, product=Product.MIS)
    ]
    mock_broker.modify_order.return_value = True

    now = datetime(2026, 3, 23, 9, 36, 0, tzinfo=IST)
    engine._evaluate_step_locked(now)

    # Verify modify_order was called to update stop from 5 to 8 shares
    assert mock_broker.modify_order.call_count >= 1
    call_args = mock_broker.modify_order.call_args
    assert call_args.kwargs.get("quantity") == 8 or (len(call_args.args) > 1 and call_args.args[1] == 8)
    assert pos.quantity == 8


def test_angel_candle_caching_and_timeout_fallback():
    """Verify AngelDataProvider caches candles and falls back to cache on rate-limit timeout."""
    from apex3.broker.angel_data_provider import AngelDataProvider
    from apex3.broker.rate_limiter import RateLimiter

    limiter = RateLimiter(rates={"historical": 10.0})
    provider = AngelDataProvider(rate_limiter=limiter)
    provider._smart_api = MagicMock()
    mock_candles = [
        ["2026-10-27T09:15:00Z", 100.0, 105.0, 99.0, 104.0, 1000],
        ["2026-10-27T09:20:00Z", 104.0, 106.0, 103.0, 105.0, 1200],
    ]
    provider._smart_api.getCandleData.return_value = {"status": True, "data": mock_candles}

    # First fetch: hits API and caches
    c1 = provider.get_historical("RELIANCE", "5minute")
    assert len(c1) == 2
    assert provider._smart_api.getCandleData.call_count == 1

    # Second fetch within TTL: served from cache without calling getCandleData
    c2 = provider.get_historical("RELIANCE", "5minute")
    assert len(c2) == 2
    assert provider._smart_api.getCandleData.call_count == 1

    # When rate limiter times out, fallback serves cached candles instead of empty list
    with patch.object(limiter, "acquire", return_value=False):
        c3 = provider.get_historical("RELIANCE", "5minute")
        assert len(c3) == 2  # Not empty!


def test_kite_quote_rate_limiting_and_timeout_error():
    """Verify KiteAdapter raises RateLimitTimeoutError on timeout and rate-limits quotes."""
    from apex3.broker.kite_adapter import KiteAdapter
    from apex3.broker.rate_limiter import RateLimiter, RateLimitTimeoutError

    limiter = RateLimiter(rates={"quotes": 1.0, "orders": 1.0})
    adapter = KiteAdapter(api_key="k", access_token="t", live_enabled=False, rate_limiter=limiter)
    adapter._kite = MagicMock()
    adapter._kite.quote.return_value = {
        "NSE:TCS": {"last_price": 3500.0, "upper_circuit_limit": 3850.0, "lower_circuit_limit": 3150.0}
    }
    adapter._kite.instruments.return_value = [
        {"tradingsymbol": "TCS", "segment": "NSE", "instrument_type": "EQ", "tick_size": 0.05, "lot_size": 1}
    ]

    # With rate limiter timing out, calling _acquire_rate_limit raises RateLimitTimeoutError
    with patch.object(limiter, "acquire", return_value=False):
        with pytest.raises(RateLimitTimeoutError):
            adapter._acquire_rate_limit("quotes")

    # get_instrument_info acquires quotes rate limit and caches result
    with patch.object(limiter, "acquire", wraps=limiter.acquire) as mock_acq:
        info1 = adapter.get_instrument_info("TCS")
        assert mock_acq.call_count >= 1
        assert info1.symbol == "TCS"
        assert info1.upper_circuit == 3850.0

        # Second call served from cache without extra API calls
        info2 = adapter.get_instrument_info("TCS")
        assert info2 is info1


def test_kite_mis_margin_fails_closed_on_error():
    """Verify KiteAdapter marks MIS unavailable when order_margins returns an error or failure."""
    from apex3.broker.kite_adapter import KiteAdapter

    adapter = KiteAdapter(api_key="k", access_token="t", live_enabled=False)
    adapter._kite = MagicMock()
    adapter._kite.quote.return_value = {"NSE:TEST": {"last_price": 500.0}}
    adapter._kite.instruments.return_value = [
        {"tradingsymbol": "TEST", "segment": "NSE", "instrument_type": "EQ", "tick_size": 0.05, "lot_size": 1}
    ]
    # Margin check returns error or missing margin
    adapter._kite.order_margins.return_value = [{"error": "MIS blocked for this stock"}]

    info = adapter.get_instrument_info("TEST")
    assert info.mis_available is False


def test_flatten_fails_closed_on_unconfirmed_order_states():
    """Verify flatten refuses to place duplicate exit when existing exit status fails or stop status is None."""
    from apex3.reconciliation.reconciler import Reconciler
    from apex3.reconciliation.state_machine import StateMachine

    mock_broker = MagicMock()
    sm = StateMachine()
    journal = MagicMock()
    cfg = AppConfig()
    reconciler = Reconciler(mock_broker, sm, journal, "TEST", config=cfg)

    # 1. Position in EXITING with existing exit_order_id, but broker status call raises Exception
    pos = sm.get_or_create_position("INFY")
    pos.phase = PositionPhase.EXITING
    pos.side = Side.BUY
    pos.quantity = 10
    pos.exit_order_id = "EX_INFY_1"

    mock_broker.get_positions.return_value = [
        PositionState(symbol="INFY", side=Side.BUY, quantity=10, product=Product.MIS, average_entry=1500.0)
    ]
    mock_broker.get_order_status.side_effect = RuntimeError("Broker connection timeout")

    # Flatten must fail closed: do NOT submit duplicate exit order!
    success, msg = reconciler.execute_flatten_sequence()
    assert mock_broker.place_order.call_count == 0

    # 2. Position with protective stop where broker status returns None (missing status)
    sm.positions.clear()
    pos2 = sm.get_or_create_position("TCS")
    pos2.phase = PositionPhase.OPEN_PROTECTED
    pos2.side = Side.BUY
    pos2.quantity = 5
    pos2.protective_order_id = "SL_TCS_1"

    mock_broker.get_positions.return_value = [
        PositionState(symbol="TCS", side=Side.BUY, quantity=5, product=Product.MIS, average_entry=3500.0)
    ]
    mock_broker.get_order_status.side_effect = None
    mock_broker.get_order_status.return_value = None  # None = missing status
    reconciler._confirm_order_terminal = MagicMock(return_value=None)

    success2, msg2 = reconciler.execute_flatten_sequence()
    # Must refuse to submit exit order because stop state is unconfirmed
    assert mock_broker.place_order.call_count == 0
    assert "unconfirmed" in msg2.lower()


def test_exit_pricing_rejects_stale_or_missing_quote():
    """Verify exit pricing helper strictly requires fresh quotes with top-of-book depth."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Stale quote: is_fresh = False
    mock_broker.get_quote.return_value = Quote(
        symbol="TCS",
        exchange="NSE",
        last_price=3500.0,
        bid=3499.0,
        ask=3501.0,
        volume=1000,
        timestamp=datetime.now(IST),
        is_fresh=False,  # STALE
    )
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TCS", exchange="NSE", tick_size=0.05)

    with pytest.raises(ValueError, match="valid price quote is required"):
        engine._create_exit_order_request("TCS", Side.SELL, 5, "EXIT")

    # Stale quote: age > 15 seconds
    mock_broker.get_quote.return_value = Quote(
        symbol="TCS",
        exchange="NSE",
        last_price=3500.0,
        bid=3499.0,
        ask=3501.0,
        volume=1000,
        timestamp=datetime.now(IST) - timedelta(seconds=20),  # OLD
        is_fresh=True,
    )
    with pytest.raises(ValueError, match="valid price quote is required"):
        engine._create_exit_order_request("TCS", Side.SELL, 5, "EXIT")


def test_dashboard_status_requires_auth_and_serves_cached_regime():
    """Verify /api/status requires operator authorization when auth_required=True and uses cached regime."""
    from apex3.dashboard.app import create_app
    from apex3.strategy.regime import RegimeEvaluation, MarketRegime
    from apex3.models import GateResult

    from apex3.config import DashboardConfig
    cfg = AppConfig(dashboard=DashboardConfig(auth_required=True, api_token="secure_test_token"))

    mock_engine = MagicMock()
    mock_engine.scheduler.now_ist.return_value = datetime(2026, 10, 27, 10, 0, tzinfo=IST)
    mock_engine.scheduler.get_state_display.return_value = {"state": "ENTRY_WINDOW"}
    mock_engine.broker.get_margins.return_value = MagicMock(equity_available=15000.0, equity_used=0.0, collateral=0.0, net_value=15000.0)
    mock_engine.broker.get_orders.return_value = []
    mock_engine.state_machine.positions = {}
    mock_engine.is_paused = False
    mock_engine.pause_reason = ""
    mock_engine.get_active_watchlist.return_value = ["INFY", "TCS"]
    mock_engine.last_decisions = []
    mock_engine.journal = MagicMock()
    mock_engine.journal.get_orders.return_value = []
    mock_engine.journal.get_events.return_value = []

    # Engine has cached regime
    cached_regime = RegimeEvaluation(
        regime=MarketRegime.BULLISH,
        gate=GateResult(name="REGIME", status=GateStatus.PASS, reason="Bullish trend verified"),
        ema_slope_val=0.005,
        adx_val=26.5,
    )
    mock_engine.latest_regime_eval = cached_regime

    app = create_app(mock_engine, cfg, mock_engine.journal)
    client = app.test_client()

    # 1. Unauthenticated request to /api/status returns 401
    res_unauth = client.get("/api/status")
    assert res_unauth.status_code == 401

    # 2. Authenticated request with X-Operator-Token returns 200 and uses cached regime
    res_auth = client.get("/api/status", headers={"X-Operator-Token": "secure_test_token"})
    assert res_auth.status_code == 200
    data = res_auth.get_json()
    assert data["regime"]["state"] == "BULLISH"
    # Broker get_historical was NOT called by dashboard!
    assert mock_engine.broker.get_historical.call_count == 0


def test_late_start_at_0950_reconstructs_opening_range_and_trades():
    """Verify that starting late at 09:50 IST reconstructs Opening Range from 09:15-09:30 bars and trades breakout."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_positions.return_value = []
    mock_broker.estimate_order_margin.return_value = OrderMarginEstimate(required_margin=3000.0, leverage_available=4.0)

    now = datetime(2026, 3, 20, 9, 50, tzinfo=IST)
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05)
    trent_quote = Quote(symbol="TRENT", exchange="NSE", last_price=1000.0, bid=999.8, ask=1000.2, volume=10000, timestamp=now, is_fresh=True)
    nifty_quote = Quote(symbol="NIFTY 50", exchange="NSE", last_price=24500.0, bid=24498.0, ask=24502.0, volume=50000, timestamp=now, is_fresh=True)
    mock_broker.get_quote.return_value = trent_quote
    mock_broker.get_quotes_batch.return_value = {"TRENT": trent_quote, "NIFTY 50": nifty_quote}

    placed_orders = []
    def on_place(req):
        ord_id = f"ORD_{len(placed_orders) + 1}"
        placed_orders.append((ord_id, req))
        return ord_id
    mock_broker.place_order.side_effect = on_place

    def on_status(ord_id):
        if "ORD_1" in ord_id:
            # Immediate entry fill @ 1000.0
            return OrderState(
                order_id="ORD_1",
                request=OrderRequest(symbol="TRENT", side=Side.BUY, quantity=10, price=1000.05),
                status=OrderStatus.COMPLETE,
                filled_quantity=10,
                pending_quantity=0,
                average_price=1000.0,
            )
        elif "ORD_2" in ord_id:
            # Protective SL
            return OrderState(
                order_id="ORD_2",
                request=OrderRequest(symbol="TRENT", side=Side.SELL, quantity=10, order_type=OrderType.SL),
                status=OrderStatus.TRIGGER_PENDING,
                pending_quantity=10,
                filled_quantity=0,
            )
        return None
    mock_broker.get_order_status.side_effect = on_status

    cfg = AppConfig()
    object.__setattr__(cfg.watchlist, "symbols", ["TRENT"])
    object.__setattr__(cfg.watchlist, "scanner_enabled", False)
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    # Verify opening ranges starts completely empty on cold / late start
    assert len(engine.opening_ranges) == 0

    # Build 5m candles spanning 09:15 to 09:50 IST
    # 09:15, 09:20, 09:25 are the 3 OR bars (High=995.0, Low=985.0)
    c_0915 = Candle(timestamp=datetime(2026, 3, 20, 9, 15, tzinfo=IST), open=992.0, high=993.0, low=985.0, close=992.0, volume=5000, is_complete=True)
    c_0920 = Candle(timestamp=datetime(2026, 3, 20, 9, 20, tzinfo=IST), open=992.0, high=994.0, low=990.0, close=993.0, volume=6000, is_complete=True)
    c_0921 = Candle(timestamp=datetime(2026, 3, 20, 9, 25, tzinfo=IST), open=994.0, high=995.0, low=996.0, close=995.0, volume=7000, is_complete=True)
    c_0930 = Candle(timestamp=datetime(2026, 3, 20, 9, 30, tzinfo=IST), open=995.0, high=997.0, low=996.0, close=996.0, volume=4000, is_complete=True)
    c_0935 = Candle(timestamp=datetime(2026, 3, 20, 9, 35, tzinfo=IST), open=996.0, high=998.0, low=996.0, close=997.0, volume=4000, is_complete=True)
    c_0940 = Candle(timestamp=datetime(2026, 3, 20, 9, 40, tzinfo=IST), open=997.0, high=999.0, low=996.0, close=998.0, volume=4000, is_complete=True)
    c_0945 = Candle(timestamp=datetime(2026, 3, 20, 9, 45, tzinfo=IST), open=998.0, high=1001.0, low=996.0, close=1000.0, volume=15000, is_complete=True)

    # 20 prior bars around 1000.0 with high-low span ~ 2.0
    prior_bars = [
        Candle(timestamp=datetime(2026, 3, 19, 14, 0, tzinfo=IST) + timedelta(minutes=5 * i), open=1000.0, high=1001.0, low=999.0, close=1000.0, volume=5000, is_complete=True)
        for i in range(20)
    ]
    trent_candles = prior_bars + [c_0915, c_0920, c_0921, c_0930, c_0935, c_0940, c_0945]
    mock_broker.get_historical.return_value = trent_candles

    fake_sig = SignalEvaluation(
        symbol="TRENT",
        decision=Decision.PROPOSED_LONG,
        signal_price=1000.0,
        side=Side.BUY,
        gates=[GateResult("TEST", GateStatus.PASS, "OK")],
        relative_strength=0.05,
        relative_volume=3.0,
    )

    import apex3.strategy.engine as eng_module
    orig_eval = eng_module.evaluate_signal
    orig_regime = eng_module.evaluate_market_regime
    try:
        eng_module.evaluate_signal = MagicMock(return_value=fake_sig)
        eng_module.evaluate_market_regime = MagicMock(return_value=MagicMock(regime=MarketRegime.BULLISH, is_valid=True))

        engine.evaluate_step(now)

        # 1. Opening Range reconstructed automatically from 09:15-09:30 bars
        assert "TRENT" in engine.opening_ranges
        or_res = engine.opening_ranges["TRENT"]
        assert or_res.is_complete is True
        assert or_res.high == 995.0
        assert or_res.low == 985.0
        assert or_res.candle_count == 3

        # 2. Both entry order and protective SL order were placed on cycle at 09:50
        assert len(placed_orders) == 2
        entry_ord = placed_orders[0][1]
        sl_ord = placed_orders[1][1]

        assert entry_ord.symbol == "TRENT"
        assert entry_ord.side == Side.BUY
        assert sl_ord.symbol == "TRENT"
        assert sl_ord.side == Side.SELL
        assert sl_ord.order_type == OrderType.SL

        # 3. Position phase is OPEN_PROTECTED
        pos = engine.state_machine.positions["TRENT"]
        assert pos.phase == PositionPhase.OPEN_PROTECTED
        assert pos.quantity == 10
        assert pos.average_entry == 1000.0
    finally:
        eng_module.evaluate_signal = orig_eval
        eng_module.evaluate_market_regime = orig_regime


def test_jiofin_token_is_18143():
    """Verify JIOFIN NSE token in AngelDataProvider is 18143 (not outdated 18147)."""
    from apex3.broker.angel_data_provider import DEFAULT_ANGEL_TOKENS
    assert DEFAULT_ANGEL_TOKENS.get("JIOFIN") == "18143"


def test_signal_config_max_quote_age_default_15s():
    """Verify SignalConfig.max_quote_age_sec and ScheduleConfig.stale_threshold_sec default to 15s."""
    cfg = load_config()
    assert cfg.signal.max_quote_age_sec == 15.0
    assert cfg.schedule.stale_threshold_sec == 15


def test_kite_adapter_quote_permission_denied_skips_subsequent_quotes():
    """When Kite quote() raises 'Insufficient permission', KiteAdapter caches flag and skips redundant quote() calls."""
    from apex3.broker.kite_adapter import KiteAdapter
    adapter = KiteAdapter(api_key="test_key", access_token="test_tok", live_enabled=False)
    mock_kite = MagicMock()
    adapter._kite = mock_kite
    adapter._instruments_cache = {
        "RELIANCE": {"tradingsymbol": "RELIANCE", "instrument_token": 738561, "tick_size": 0.10, "lot_size": 1, "segment": "NSE", "instrument_type": "EQ"},
        "TCS": {"tradingsymbol": "TCS", "instrument_token": 2953217, "tick_size": 0.10, "lot_size": 1, "segment": "NSE", "instrument_type": "EQ"},
    }
    adapter._cache_date = datetime.now(IST).strftime("%Y-%m-%d")

    mock_kite.quote.side_effect = Exception("Insufficient permission for that call.")
    mock_kite.order_margins.return_value = [{"total": 25.0}, {"total": 25.0}]

    info1 = adapter.get_instrument_info("RELIANCE")
    assert info1.mis_available is True
    assert info1.tick_size == 0.10
    assert adapter._quote_permission_denied is True
    assert mock_kite.quote.call_count == 1

    # Second symbol skips kite.quote() entirely!
    info2 = adapter.get_instrument_info("TCS")
    assert info2.mis_available is True
    assert mock_kite.quote.call_count == 1


def test_schedule_0930_to_0935_and_special_session_display():
    """Verify 09:30-09:35 returns OPENING_RANGE (not POST_CLOSE), SPECIAL_SESSION display works, and build_opening_range blocks prior-day leakage."""
    from apex3.strategy.schedule import SessionScheduler, SessionState
    from apex3.strategy.signal import build_opening_range
    cfg = load_config()
    object.__setattr__(cfg.schedule, "exchange", "NSE")
    sched = SessionScheduler(cfg.schedule)

    # At 09:32 IST on a regular trading Friday, state must be OPENING_RANGE (waiting for 09:35 entry_start), NOT POST_CLOSE
    dt_0932 = datetime(2026, 3, 20, 9, 32, tzinfo=IST)
    assert sched.get_session_state(dt_0932) == SessionState.OPENING_RANGE

    # Special session display must not raise KeyError
    dt_muhurat = datetime(2026, 11, 8, 18, 20, tzinfo=IST)
    disp = sched.get_state_display(dt_muhurat)
    assert disp["state"] == "SPECIAL_SESSION"

    # build_opening_range at 09:16 IST with only yesterday's completed bars must return is_complete=False
    yesterday_bars = [
        Candle(timestamp=datetime(2026, 3, 19, 9, 15, tzinfo=IST), open=100.0, high=102.0, low=99.0, close=101.0, volume=1000, is_complete=True),
        Candle(timestamp=datetime(2026, 3, 19, 9, 20, tzinfo=IST), open=101.0, high=103.0, low=100.0, close=102.0, volume=1000, is_complete=True),
        Candle(timestamp=datetime(2026, 3, 19, 9, 25, tzinfo=IST), open=102.0, high=104.0, low=101.0, close=103.0, volume=1000, is_complete=True),
        Candle(timestamp=datetime(2026, 3, 20, 9, 15, tzinfo=IST), open=105.0, high=106.0, low=104.0, close=105.5, volume=500, is_complete=False),
    ]
    or_today = build_opening_range("RELIANCE", yesterday_bars, session_date=datetime(2026, 3, 20, tzinfo=IST).date())
    assert or_today.is_complete is False
    assert or_today.candle_count == 0


def test_api_set_token_endpoint():
    """Verify POST /api/set_token updates access token and broker session hot-reload without mutating live .env."""
    import os
    from apex3.dashboard.app import create_app
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    orig_env_content = open(env_path, "r", encoding="utf-8").read() if os.path.exists(env_path) else None
    orig_env_var = os.environ.get("KITE_ACCESS_TOKEN")
    try:
        cfg = AppConfig()
        object.__setattr__(cfg.dashboard, "auth_required", True)
        mock_broker = MagicMock()
        mock_journal = MagicMock()
        engine = StrategyEngine(cfg, mock_broker, mock_journal)
        app = create_app(engine, cfg, mock_journal)
        client = app.test_client()

        # Unauthorized request without operator token fails
        r_unauth = client.post("/api/set_token", json={"access_token": "fresh_tok_123"}, headers={"Origin": "http://127.0.0.1:5000"})
        assert r_unauth.status_code == 401

        # Authorized request with Bearer token succeeds
        r_auth = client.post(
            "/api/set_token",
            json={"access_token": "fresh_tok_123"},
            headers={"Origin": "http://127.0.0.1:5000", "Authorization": "Bearer apex3_secure_operator_token"},
        )
        assert r_auth.status_code == 200
        assert r_auth.get_json()["status"] == "SUCCESS"
    finally:
        if orig_env_content is not None:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(orig_env_content)
        if orig_env_var is not None:
            os.environ["KITE_ACCESS_TOKEN"] = orig_env_var


def test_build_session_pnl_skips_unmapped_or_manual_executions():
    """Verify that external, manual, or unmapped trades do NOT crash strategy session PnL calculation."""
    from apex3.strategy.accounting import build_session_pnl
    from apex3.config import CostsConfig
    from apex3.models import Side, Product, OrderType, OrderState, OrderRequest
    from datetime import date

    costs = CostsConfig()
    today = date(2026, 11, 2)
    session_tag = "APEX3_TEST"

    # Suppose broker reports external or manual trades whose order_id is NOT in strategy orders
    unmapped_trade = {
        "trade_id": "TRD_EXT_999",
        "order_id": "ORD_EXT_UNKNOWN",
        "symbol": "INFY",
        "side": "BUY",
        "quantity": 10,
        "price": 1800.0,
    }
    pnl = build_session_pnl(
        trades=[unmapped_trade],
        orders=[],
        open_positions=[],
        session_tag=session_tag,
        session_date=today,
        costs=costs,
    )
    assert pnl.trade_count == 0
    assert pnl.realized_pnl == 0.0
    assert pnl.estimated_charges == 0.0


def test_pnl_accounting_transient_resilience():
    """Verify that a transient network/broker error does not immediately lock the engine for the day."""
    from unittest.mock import MagicMock
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from apex3.config import load_config
    from apex3.strategy.engine import StrategyEngine
    from apex3.models import Decision

    ist_zone = ZoneInfo("Asia/Kolkata")
    cfg = load_config()
    broker = MagicMock()
    journal = MagicMock()
    broker.is_session_valid.return_value = True
    broker.get_positions.return_value = []
    broker.get_orders.return_value = []
    # Simulate broker.get_trades raising transient network timeout
    broker.get_trades.side_effect = ConnectionError("Transient Zerodha API timeout")

    engine = StrategyEngine(cfg, broker, journal)
    engine._consecutive_pnl_failures = 0

    now = datetime(2026, 11, 2, 10, 0, 0, tzinfo=ist_zone)
    decisions = engine.evaluate_step(now)

    # Engine must NOT be permanently paused on first failure
    assert engine.is_paused is False
    assert engine._daily_loss_halt_date is None
    assert engine._consecutive_pnl_failures == 1
    assert any("TRANSIENT_ACCOUNTING_WAIT" in d.reason_codes for d in decisions)


def test_toggle_pause_clears_daily_loss_halt_date():
    """Verify that operator unpausing clears daily halt date and failure counters."""
    from unittest.mock import MagicMock
    from datetime import date
    from apex3.config import load_config
    from apex3.strategy.engine import StrategyEngine
    from apex3.dashboard.app import create_app

    cfg = load_config()
    broker = MagicMock()
    journal = MagicMock()
    engine = StrategyEngine(cfg, broker, journal)
    engine.is_paused = True
    engine.pause_reason = "DAILY_PNL_ACCOUNTING_FAILED"
    engine._daily_loss_halt_date = date(2026, 11, 2)
    engine._consecutive_pnl_failures = 5

    app = create_app(engine, cfg, journal)
    client = app.test_client()

    r = client.post(
        "/api/pause",
        json={"pause": False},
        headers={"Origin": "http://127.0.0.1:5000", "Authorization": "Bearer apex3_secure_operator_token"},
    )
    assert r.status_code == 200
    assert engine.is_paused is False
    assert engine._daily_loss_halt_date is None
    assert engine._consecutive_pnl_failures == 0


def test_manual_order_in_zerodha_does_not_halt_algo():
    """Verify that manual orders/positions on Zerodha do NOT halt the strategy or block other symbols."""
    from unittest.mock import MagicMock
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from apex3.config import load_config
    from apex3.strategy.engine import StrategyEngine
    from apex3.models import PositionState, Side, Product, Decision, MarginInfo

    ist_zone = ZoneInfo("Asia/Kolkata")
    cfg = load_config()
    broker = MagicMock()
    journal = MagicMock()

    broker.is_session_valid.return_value = True
    # User holds a manual MIS position in INFY
    broker.get_positions.return_value = [
        PositionState(symbol="INFY", side=Side.BUY, quantity=10, average_entry=1500.0, product=Product.MIS)
    ]
    broker.get_orders.return_value = []
    broker.get_trades.return_value = []
    broker.get_margins.return_value = MarginInfo(
        equity_available=12000.0,
        equity_used=3000.0,
        collateral=0.0,
        net_value=15000.0,
        timestamp=datetime(2026, 11, 2, 10, 0, 0, tzinfo=ist_zone),
    )

    engine = StrategyEngine(cfg, broker, journal)
    # Radar monitors INFY and RELIANCE
    engine.get_active_watchlist = MagicMock(return_value=["INFY", "RELIANCE"])

    now = datetime(2026, 11, 2, 10, 0, 0, tzinfo=ist_zone)
    decisions = engine.evaluate_step(now)

    # Strategy must NOT halt on UNKNOWN_MANUAL_POSITIONS
    assert not any("UNKNOWN_MANUAL_POSITIONS" in d.reason_codes for d in decisions)
    assert not any(d.decision == Decision.HALT for d in decisions)





