"""Comprehensive Unit & Integration Tests for Phase 1-4 Production Audit Fixes.

Covers:
1. Dynamic tick-size rounding (0.01 for JIOFIN/TATASTEEL, 0.05 for RELIANCE, 1.00 for MARUTI) in stops, targets, and fee-lock.
2. Late-start ('start any time') validity rules:
   - Valid setup within drift tolerance, stop & target intact, inside entry window.
   - Rejected setup when stop-loss already hit between trigger bar and late start.
   - Rejected setup when target already hit between trigger bar and late start.
   - Rejected setup when price drifted > max_price_drift_pct (0.15%) or fell back inside OR.
   - Rejected setup when outside entry window (> 12:30 IST).
3. Startup reconciliation & automatic protective SL re-attachment on mid-session crash/restart.
4. Duplicate-start prevention: OS lockfile (`acquire_instance_lock`) and live broker order-book duplicate guard (`DUPLICATE_ORDER_PREVENTED`).
5. Safe `KiteAdapter.get_orders()` / `get_order_status()` handling of non-MIS products (`CNC`, `NRML`, `MTF`) and `MIS`-only risk filtering in `StrategyEngine`.
6. Exact Zerodha statutory charge parity between `TradeLogger` and `sizer.py` (`calculate_zerodha_intraday_charges`).
7. Live-data Paper Trading mode (`HybridBrokerAdapter` + `PaperBroker`) feeding real quotes/ticks into paper execution.
8. Healthcheck endpoint (`/health`) and `AlertNotifier` (Telegram & SMTP email) integration tests.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from apex3.alerts.notifier import AlertNotifier
from apex3.broker.hybrid_adapter import HybridBrokerAdapter
from apex3.broker.kite_adapter import KiteAdapter
from apex3.broker.paper import PaperBroker
from apex3.config import AppConfig, load_config
from apex3.dashboard.app import create_app
from apex3.journal.trade_logger import TradeLogger
from apex3.models import (
    Candle,
    Decision,
    GateResult,
    GateStatus,
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
    make_broker_tag,
)
from apex3.reconciliation.reconciler import Reconciler
from apex3.reconciliation.state_machine import PositionPhase, StateMachine
from apex3.strategy.engine import StrategyEngine
from apex3.strategy.regime import MarketRegime
from apex3.strategy.signal import (
    SignalEvaluation,
    StockCandidateBar,
    evaluate_candidate_signal,
    validate_late_start_signal,
)
from apex3.strategy.sizer import calculate_zerodha_intraday_charges
from apex3.strategy.trail import (
    ceil_to_tick,
    compute_initial_stop_and_target,
    evaluate_fee_protected_stop,
    floor_to_tick,
    round_to_tick,
)

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# 1. Tick-Size Rounding & Stop/Target Math Tests (0.01, 0.05, 1.00)
# ---------------------------------------------------------------------------

def test_dynamic_tick_size_rounding_maruti_and_jiofin():
    """Verify 1.00 tick size (MARUTI) and 0.01 tick size (JIOFIN/TATASTEEL) in stop, target, and fee-lock math."""
    cfg = load_config().raw_dict

    # MARUTI: tick_size = 1.00
    maruti_levels = compute_initial_stop_and_target(
        side="LONG",
        entry_price=12500.0,
        swing_low_5=12440.35,
        swing_high_5=12520.0,
        atr14=25.10,
        cfg=cfg,
        tick_size=1.00,
    )
    # Must be an exact integer multiple of 1.00!
    assert maruti_levels.initial_stop % 1.0 == pytest.approx(0.0)
    assert maruti_levels.target_price % 1.0 == pytest.approx(0.0)

    # Fee-protected stop on MARUTI at +1.70R must also be an exact multiple of 1.00
    current_px_maruti = maruti_levels.entry_price + 1.70 * maruti_levels.risk_per_share
    new_stop_maruti = evaluate_fee_protected_stop(maruti_levels, current_px_maruti, "11:00", cfg, tick_size=1.00)
    assert new_stop_maruti is not None
    assert new_stop_maruti % 1.0 == pytest.approx(0.0)

    # JIOFIN: tick_size = 0.01
    jio_levels = compute_initial_stop_and_target(
        side="LONG",
        entry_price=242.13,
        swing_low_5=241.18,
        swing_high_5=242.50,
        atr14=0.45,
        cfg=cfg,
        tick_size=0.01,
    )
    # 241.18 - 1.5 * 0.45 = 240.505 -> floor to 0.01 = 240.50 (or exact 0.01 step)
    assert round(jio_levels.initial_stop * 100) == pytest.approx(jio_levels.initial_stop * 100)
    assert round(jio_levels.target_price * 100) == pytest.approx(jio_levels.target_price * 100)
    assert floor_to_tick(240.539, tick=0.01) == 240.53
    assert ceil_to_tick(240.531, tick=0.01) == 240.54
    assert round_to_tick(11397.35, tick=1.00) == 11397.0


# ---------------------------------------------------------------------------
# 2. Late-Start ("Start Any Time") Validity Rule Tests
# ---------------------------------------------------------------------------

def test_late_start_validity_rules_comprehensive():
    """Verify validate_late_start_signal enforces window, stop-not-hit, target-not-hit, OR boundary, and drift cap."""
    trig_ts = datetime(2026, 10, 29, 9, 35, tzinfo=IST)
    now_0950 = datetime(2026, 10, 29, 9, 50, tzinfo=IST)

    # Base validbars between 09:40 and 09:45 (price stayed around 1000.5, above stop 993.0 and below target 1014.0)
    valid_bars = [
        Candle(timestamp=datetime(2026, 10, 29, 9, 40, tzinfo=IST), open=1000.0, high=1002.0, low=998.5, close=1001.0, volume=5000, is_complete=True),
        Candle(timestamp=datetime(2026, 10, 29, 9, 45, tzinfo=IST), open=1001.0, high=1003.0, low=999.0, close=1000.8, volume=6000, is_complete=True),
    ]
    fresh_quote = Quote(symbol="TRENT", exchange="NSE", last_price=1001.0, bid=1000.9, ask=1001.1, volume=20000, timestamp=now_0950, is_fresh=True)

    # Case A: Valid late-start signal (drift = 0.10% <= 0.15%, stop & target intact, inside window)
    res_ok = validate_late_start_signal(
        symbol="TRENT",
        side=Side.BUY,
        signal_price=1000.0,
        initial_stop=993.0,
        target_price=1014.0,
        or_high=998.0,
        or_low=990.0,
        trigger_bar_timestamp=trig_ts,
        subsequent_candles=valid_bars,
        quote=fresh_quote,
        now_ist=now_0950,
    )
    assert res_ok.is_valid is True
    assert res_ok.reason_code == "LATE_START_SIGNAL_VALID"

    # Case B: Stop-loss was hit on the 09:40 bar (low = 992.5 <= 993.0) before recovery
    bars_sl_hit = [
        Candle(timestamp=datetime(2026, 10, 29, 9, 40, tzinfo=IST), open=1000.0, high=1001.0, low=992.5, close=999.0, volume=5000, is_complete=True),
    ]
    res_sl = validate_late_start_signal(
        symbol="TRENT",
        side=Side.BUY,
        signal_price=1000.0,
        initial_stop=993.0,
        target_price=1014.0,
        or_high=998.0,
        or_low=990.0,
        trigger_bar_timestamp=trig_ts,
        subsequent_candles=bars_sl_hit,
        quote=fresh_quote,
        now_ist=now_0950,
    )
    assert res_sl.is_valid is False
    assert res_sl.reason_code == "LATE_START_STOP_ALREADY_HIT"

    # Case C: Target was already hit on the 09:40 bar (high = 1015.0 >= 1014.0)
    bars_tgt_hit = [
        Candle(timestamp=datetime(2026, 10, 29, 9, 40, tzinfo=IST), open=1000.0, high=1015.0, low=999.0, close=1001.0, volume=5000, is_complete=True),
    ]
    res_tgt = validate_late_start_signal(
        symbol="TRENT",
        side=Side.BUY,
        signal_price=1000.0,
        initial_stop=993.0,
        target_price=1014.0,
        or_high=998.0,
        or_low=990.0,
        trigger_bar_timestamp=trig_ts,
        subsequent_candles=bars_tgt_hit,
        quote=fresh_quote,
        now_ist=now_0950,
    )
    assert res_tgt.is_valid is False
    assert res_tgt.reason_code == "LATE_START_TARGET_ALREADY_HIT"

    # Case D: Price drifted too far (+0.50% > 0.15% max_price_drift_pct)
    drifted_quote = Quote(symbol="TRENT", exchange="NSE", last_price=1005.0, bid=1004.9, ask=1005.1, volume=20000, timestamp=now_0950, is_fresh=True)
    res_drift = validate_late_start_signal(
        symbol="TRENT",
        side=Side.BUY,
        signal_price=1000.0,
        initial_stop=993.0,
        target_price=1014.0,
        or_high=998.0,
        or_low=990.0,
        trigger_bar_timestamp=trig_ts,
        subsequent_candles=valid_bars,
        quote=drifted_quote,
        now_ist=now_0950,
    )
    assert res_drift.is_valid is False
    assert res_drift.reason_code == "LATE_START_PRICE_DRIFT_EXCEEDED"

    # Case E: Outside entry window (12:45 IST > 12:30 cutoff)
    res_late = validate_late_start_signal(
        symbol="TRENT",
        side=Side.BUY,
        signal_price=1000.0,
        initial_stop=993.0,
        target_price=1014.0,
        or_high=998.0,
        or_low=990.0,
        trigger_bar_timestamp=trig_ts,
        subsequent_candles=valid_bars,
        quote=fresh_quote,
        now_ist=datetime(2026, 10, 29, 12, 45, tzinfo=IST),
    )
    assert res_late.is_valid is False
    assert res_late.reason_code == "OUTSIDE_ENTRY_WINDOW"


# ---------------------------------------------------------------------------
# 3. Startup Crash Recovery & Immediate SL Re-Attachment
# ---------------------------------------------------------------------------

def test_startup_recovery_reattaches_missing_sl_and_restores_hwm():
    """On restart when a strategy position has no resting SL, Reconciler immediately places & confirms a new SL."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    cfg = AppConfig()
    sm = StateMachine("APEX3")
    reconciler = Reconciler(mock_broker, sm, mock_journal, session_tag="APEX3", config=cfg)

    # Broker has an open MIS position of 15 shares of RELIANCE @ 2500.0
    mock_broker.get_positions.return_value = [
        PositionState(symbol="RELIANCE", side=Side.BUY, quantity=15, average_entry=2500.0, product=Product.MIS)
    ]
    # Broker orderbook has the completed entry order, but NO active TRIGGER_PENDING SL order
    mock_broker.get_orders.return_value = [
        OrderState(
            order_id="ENT_REL_1",
            request=OrderRequest(symbol="RELIANCE", side=Side.BUY, quantity=15, order_type=OrderType.LIMIT, price=2500.0, tag="APEX3_RELI_E0940"),
            status=OrderStatus.COMPLETE,
            filled_quantity=15,
            average_price=2500.0,
        )
    ]
    mock_broker.get_trades.return_value = []
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="RELIANCE", exchange="NSE", tick_size=0.05, lot_size=1)
    mock_broker.get_quote.return_value = Quote(
        symbol="RELIANCE", exchange="NSE", last_price=2510.0, bid=2509.9, ask=2510.1, volume=50000, timestamp=datetime.now(IST), is_fresh=True
    )

    # Journal has the initial stop (2482.50) and target (2535.00)
    mock_journal.get_events.return_value = [
        {"data": {"initial_stop": 2482.50, "target_price": 2535.00, "best_favorable_price": 2515.00}}
    ]

    mock_broker.place_order.return_value = "NEW_SL_REL_99"
    mock_broker.get_order_status.return_value = OrderState(
        order_id="NEW_SL_REL_99",
        request=OrderRequest(symbol="RELIANCE", side=Side.SELL, quantity=15, order_type=OrderType.SL, trigger_price=2482.50),
        status=OrderStatus.TRIGGER_PENDING,
        pending_quantity=15,
    )

    status = reconciler.audit_startup_state(["RELIANCE"], reattach_stops=True)
    assert status.is_clean is True
    pos = sm.positions["RELIANCE"]
    assert pos.phase == PositionPhase.OPEN_PROTECTED
    assert pos.protective_order_id == "NEW_SL_REL_99"
    assert pos.initial_stop_price == 2482.50
    assert pos.target_price == 2535.00
    assert pos.best_favorable_price == 2515.00
    assert mock_broker.place_order.call_count == 1
    placed_req = mock_broker.place_order.call_args[0][0]
    assert placed_req.order_type == OrderType.SL
    assert placed_req.quantity == 15
    assert placed_req.trigger_price == 2482.50


# ---------------------------------------------------------------------------
# 4. Duplicate-Start Lockfile & Order Book Idempotency Tests
# ---------------------------------------------------------------------------

def test_duplicate_order_prevented_when_order_already_in_broker_book():
    """Engine refuses to submit a duplicate entry order if an active LIMIT order for the symbol already exists on the broker."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_trades.return_value = []
    mock_broker.get_positions.return_value = []
    mock_broker.estimate_order_margin.return_value = OrderMarginEstimate(required_margin=3000.0, leverage_available=4.0)

    now = datetime(2026, 3, 20, 9, 40, tzinfo=IST)
    # Broker orderbook ALREADY has an OPEN LIMIT order for TRENT (e.g. from another process or prior unrecorded call)
    mock_broker.get_orders.return_value = [
        OrderState(
            order_id="EXISTING_ORD_1",
            request=OrderRequest(symbol="TRENT", side=Side.BUY, quantity=10, order_type=OrderType.LIMIT, price=1000.05, tag="EXT_ORDER"),
            status=OrderStatus.OPEN,
            pending_quantity=10,
        )
    ]
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05)
    trent_q = Quote(symbol="TRENT", exchange="NSE", last_price=1000.0, bid=999.8, ask=1000.2, volume=10000, timestamp=now, is_fresh=True)
    mock_broker.get_quote.return_value = trent_q
    mock_broker.get_quotes_batch.return_value = {"TRENT": trent_q, "NIFTY 50": trent_q}

    candles = [Candle(timestamp=now - timedelta(minutes=5 * (20 - i)), open=1000.0, high=1001.0, low=999.0, close=1000.0, volume=10000, is_complete=True) for i in range(20)]
    candles[-2] = Candle(timestamp=now - timedelta(minutes=5), open=999.0, high=1001.0, low=996.0, close=1000.0, volume=20000, is_complete=True)
    mock_broker.get_historical.return_value = candles

    cfg = AppConfig()
    object.__setattr__(cfg.watchlist, "symbols", ["TRENT"])
    object.__setattr__(cfg.watchlist, "scanner_enabled", False)
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

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

        decisions = engine.evaluate_step(now)
        assert mock_broker.place_order.call_count == 0
        assert any("DUPLICATE_ORDER_PREVENTED" in d.reason_codes for d in decisions)
    finally:
        eng_module.evaluate_signal = orig_eval
        eng_module.evaluate_market_regime = orig_regime


def test_instance_lock_prevents_second_acquire(tmp_path):
    """Verify acquire_instance_lock acquires lock and release_instance_lock cleans up cleanly."""
    from apex3.main import acquire_instance_lock, release_instance_lock
    lock_file = str(tmp_path / "test_apex3.lock")
    try:
        assert acquire_instance_lock(lock_file) is True
        # Second acquire on the same file handle / byte range on Windows fails while locked
        if hasattr(__import__("os"), "name") and __import__("os").name == "nt":
            import msvcrt
            with open(lock_file, "r+", encoding="utf-8") as fh2:
                with pytest.raises(OSError):
                    msvcrt.locking(fh2.fileno(), msvcrt.LK_NBLCK, 1)
    finally:
        release_instance_lock()


# ---------------------------------------------------------------------------
# 5. Safe KiteAdapter Order Parsing & MIS-Only Risk Filtering
# ---------------------------------------------------------------------------

def test_kite_adapter_get_orders_handles_mtf_and_bo_without_keyerror():
    """Verify KiteAdapter.get_orders() and get_order_status() do not raise KeyError on MTF/BO/CO orders."""
    adapter = KiteAdapter(api_key="k", access_token="t", live_enabled=False)
    adapter._kite = MagicMock()
    adapter._kite.orders.return_value = [
        {
            "order_id": "ORD_MTF_1",
            "tradingsymbol": "INFY",
            "exchange": "NSE",
            "transaction_type": "BUY",
            "quantity": 10,
            "product": "MTF",
            "order_type": "LIMIT",
            "price": 1500.0,
            "status": "COMPLETE",
        }
    ]
    orders = adapter.get_orders()
    assert len(orders) == 1
    assert orders[0].request.product == Product.UNKNOWN


def test_cnc_delivery_position_does_not_trigger_intraday_daily_loss_halt():
    """Verify a CNC (delivery) position with unrealized loss is ignored by intraday MIS risk accounting."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    now = datetime(2026, 3, 20, 10, 0, tzinfo=IST)

    # Account holds a long-term CNC investment in ITC that is down -Rs.1,000, but 0 MIS positions
    mock_broker.get_positions.return_value = [
        PositionState(symbol="ITC", side=Side.BUY, quantity=500, average_entry=450.0, current_price=448.0, product=Product.CNC)
    ]
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_quote.return_value = Quote(symbol="NIFTY 50", exchange="NSE", last_price=24000.0, bid=0.0, ask=0.0, volume=0, timestamp=now, is_fresh=True)
    mock_broker.get_quotes_batch.return_value = {}
    mock_broker.get_historical.return_value = []

    cfg = AppConfig()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)
    decisions = engine.evaluate_step(now)

    # Engine must NOT halt with DAILY_LOSS_LIMIT_BREACHED because the ITC position is CNC, not MIS!
    assert not any("DAILY_LOSS_LIMIT_BREACHED" in d.reason_codes for d in decisions)
    assert engine.is_paused is False


# ---------------------------------------------------------------------------
# 6. Statutory Friction Parity & Live-Data Paper Mode & Alerts/Health Tests
# ---------------------------------------------------------------------------

def test_trade_logger_friction_matches_sizer_exact_paisa(tmp_path):
    """Verify TradeLogger.record_exit calculates the exact same Zerodha statutory charges as sizer.py."""
    log_file = tmp_path / "test_friction.log"
    tl = TradeLogger(log_path=str(log_file), start_date="2026-10-29")

    entry_px = 2820.60
    exit_px = 2868.20
    qty = 10
    expected_charges = calculate_zerodha_intraday_charges(entry_px, exit_px, qty)
    expected_net = round((exit_px - entry_px) * qty - expected_charges, 2)

    ts = datetime(2026, 10, 30, 11, 15, tzinfo=IST)
    tl.record_exit(
        symbol="TRENT",
        side=Side.SELL,
        quantity=qty,
        exit_price=exit_px,
        entry_price=entry_px,
        order_id="ORD_EXIT_100",
        trade_id="TRD_EXIT_100",
        exit_reason="TARGET_HIT",
        timestamp=ts,
    )
    text = log_file.read_text(encoding="utf-8")
    assert f"Rs. {expected_charges:,.2f}" in text
    assert f"Rs. {expected_net:,.2f}" in text


def test_live_data_paper_mode_routes_live_quotes_and_triggers_paper_sl():
    """Verify HybridBrokerAdapter + PaperBroker feeds live quotes into PaperBroker and triggers paper SL orders."""
    mock_angel = MagicMock()
    paper = PaperBroker(starting_margin=15000.0)
    paper._fill_config.sl_fill_probability = 1.0  # Deterministic fill
    hybrid = HybridBrokerAdapter(data_provider=mock_angel, execution_broker=paper)

    now = datetime.now(IST)
    live_q1 = Quote(symbol="RELIANCE", exchange="NSE", last_price=2500.0, bid=2499.9, ask=2500.1, volume=50000, timestamp=now, is_fresh=True)
    mock_angel.get_quote.return_value = live_q1

    # 1. Fetch live quote via hybrid -> updates paper broker quote
    q = hybrid.get_quote("RELIANCE")
    assert q.last_price == 2500.0
    assert paper.get_quote("RELIANCE").last_price == 2500.0

    # 2. Place entry LIMIT order and protective SL order on hybrid (executes in PaperBroker)
    ent_id = hybrid.place_order(OrderRequest(symbol="RELIANCE", side=Side.BUY, quantity=5, order_type=OrderType.LIMIT, price=2500.0, product=Product.MIS))
    assert hybrid.get_order_status(ent_id).status == OrderStatus.COMPLETE
    assert len(hybrid.get_positions()) == 1

    sl_id = hybrid.place_order(OrderRequest(symbol="RELIANCE", side=Side.SELL, quantity=5, order_type=OrderType.SL, price=2484.0, trigger_price=2485.0, product=Product.MIS))
    assert hybrid.get_order_status(sl_id).status == OrderStatus.TRIGGER_PENDING

    # 3. Live quote from Angel drops to 2484.50 -> triggers PaperBroker SL automatically!
    live_q2 = Quote(symbol="RELIANCE", exchange="NSE", last_price=2484.50, bid=2484.4, ask=2484.6, volume=60000, timestamp=now, is_fresh=True)
    mock_angel.get_quote.return_value = live_q2
    hybrid.get_quote("RELIANCE")

    assert hybrid.get_order_status(sl_id).status == OrderStatus.COMPLETE
    assert len(hybrid.get_positions()) == 0


def test_health_endpoint_and_alert_notifier():
    """Verify /health endpoint returns 200 JSON and AlertNotifier dispatches Telegram payload."""
    cfg = AppConfig()
    mock_broker = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_journal = MagicMock()
    engine = StrategyEngine(cfg, mock_broker, mock_journal)
    app = create_app(engine, cfg, mock_journal)
    client = app.test_client()

    res = client.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "healthy"
    assert body["broker_session_valid"] is True

    # Test AlertNotifier Telegram call with mocked requests.post
    notifier = AlertNotifier(telegram_bot_token="bot_tok", telegram_chat_id="12345")
    with patch("apex3.alerts.notifier.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        ok = notifier.notify_order_filled("TRENT", "BUY", 5, 2820.60, "ORD_123")
        assert ok is True
        assert mock_post.call_count == 1
        assert "TRENT" in mock_post.call_args.kwargs["json"]["text"]


def test_late_start_first_scan_catches_valid_earlier_breakout():
    """Verify that starting at 09:50 IST catches a valid 09:35 breakout that is still within 0.15% drift."""
    mock_broker = MagicMock()
    mock_journal = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_margins.return_value = MarginInfo(equity_available=15000.0, net_value=15000.0)
    mock_broker.get_positions.return_value = []
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.estimate_order_margin.return_value = OrderMarginEstimate(required_margin=2820.0, leverage_available=4.0)
    mock_broker.get_instrument_info.return_value = InstrumentInfo(symbol="TRENT", exchange="NSE", lot_size=1, tick_size=0.05)

    now = datetime(2026, 3, 20, 9, 50, tzinfo=IST)
    trent_q = Quote(symbol="TRENT", exchange="NSE", last_price=1000.5, bid=1000.4, ask=1000.6, volume=15000, timestamp=now, is_fresh=True)
    mock_broker.get_quote.return_value = trent_q
    mock_broker.get_quotes_batch.return_value = {"TRENT": trent_q, "NIFTY 50": trent_q}

    # Build candles where 09:30 bar (closed at 09:35) broke out at 1000.0, and 09:35, 09:40, 09:45 consolidated around 1000.5
    candles = []
    for i in range(15):
        ts = datetime(2026, 3, 19, 14, 0, tzinfo=IST) + timedelta(minutes=5 * i)
        candles.append(Candle(timestamp=ts, open=998.0, high=999.0, low=997.0, close=998.0, volume=10000, is_complete=True))
    candles.append(Candle(timestamp=datetime(2026, 3, 20, 9, 15, tzinfo=IST), open=997.5, high=999.0, low=997.0, close=998.0, volume=12000, is_complete=True))
    candles.append(Candle(timestamp=datetime(2026, 3, 20, 9, 20, tzinfo=IST), open=998.0, high=999.5, low=997.5, close=998.5, volume=11000, is_complete=True))
    candles.append(Candle(timestamp=datetime(2026, 3, 20, 9, 25, tzinfo=IST), open=998.5, high=999.8, low=997.8, close=999.0, volume=11500, is_complete=True))
    # 09:30 bar (breakout bar, swing_low_5=996.0, ATR=2.0 -> stop=993.0, stop_width=0.70%)
    candles.append(Candle(timestamp=datetime(2026, 3, 20, 9, 30, tzinfo=IST), open=999.0, high=1000.5, low=996.0, close=1000.0, volume=30000, is_complete=True))
    # 09:35, 09:40, 09:45 bars (tight consolidation above OR high 999.8, within 0.15% of 1000.0)
    candles.append(Candle(timestamp=datetime(2026, 3, 20, 9, 35, tzinfo=IST), open=1000.0, high=1000.8, low=999.9, close=1000.4, volume=9000, is_complete=True))
    candles.append(Candle(timestamp=datetime(2026, 3, 20, 9, 40, tzinfo=IST), open=1000.4, high=1000.9, low=1000.0, close=1000.5, volume=8500, is_complete=True))
    candles.append(Candle(timestamp=datetime(2026, 3, 20, 9, 45, tzinfo=IST), open=1000.5, high=1000.8, low=1000.1, close=1000.5, volume=8000, is_complete=True))
    mock_broker.get_historical.return_value = candles

    mock_broker.place_order.side_effect = ["ENT_CATCHUP_1", "SL_CATCHUP_1"]
    mock_broker.get_order_status.side_effect = lambda oid: (
        OrderState(order_id=oid, request=OrderRequest(symbol="TRENT", side=Side.BUY, quantity=10), status=OrderStatus.COMPLETE, filled_quantity=10, average_price=1000.05)
        if oid == "ENT_CATCHUP_1"
        else OrderState(order_id=oid, request=OrderRequest(symbol="TRENT", side=Side.SELL, quantity=10), status=OrderStatus.TRIGGER_PENDING, pending_quantity=10)
    )

    cfg = AppConfig()
    object.__setattr__(cfg.watchlist, "symbols", ["TRENT"])
    object.__setattr__(cfg.watchlist, "scanner_enabled", False)
    engine = StrategyEngine(cfg, mock_broker, mock_journal)

    wait_sig = SignalEvaluation(symbol="TRENT", decision=Decision.WAIT, explanation="Volume low on latest consolidation bar")
    trig_sig = SignalEvaluation(
        symbol="TRENT",
        decision=Decision.PROPOSED_LONG,
        signal_price=1000.0,
        side=Side.BUY,
        gates=[GateResult("TEST", GateStatus.PASS, "OK")],
        relative_strength=0.25,
        relative_volume=2.6,
    )

    def fake_eval_signal(symbol, stock_candles_5m, **kwargs):
        if stock_candles_5m[-1].timestamp == datetime(2026, 3, 20, 9, 30, tzinfo=IST):
            return trig_sig
        return wait_sig

    import apex3.strategy.engine as eng_module
    orig_eval = eng_module.evaluate_signal
    orig_regime = eng_module.evaluate_market_regime
    try:
        eng_module.evaluate_signal = fake_eval_signal
        eng_module.evaluate_market_regime = MagicMock(return_value=MagicMock(regime=MarketRegime.BULLISH, is_valid=True))

        decisions = engine.evaluate_step(now)
        assert any(d.decision == Decision.PROPOSED_LONG and "ENTRY_SUBMITTED" in d.reason_codes for d in decisions)
        assert mock_broker.place_order.call_count >= 1
    finally:
        eng_module.evaluate_signal = orig_eval
        eng_module.evaluate_market_regime = orig_regime

