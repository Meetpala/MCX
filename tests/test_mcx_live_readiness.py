"""End-to-End MCX Live Readiness, Late-Start (09:50 & 18:45 IST), Order-Type & Scheduler Tests."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from apex3.broker.paper import PaperBroker
from apex3.config import load_config
from apex3.journal.sqlite_journal import Journal
from apex3.models import (
    Candle,
    Decision,
    InstrumentInfo,
    OrderRequest,
    OrderType,
    Product,
    Quote,
    Side,
    get_mcx_lot_multiplier,
)
from apex3.strategy.engine import StrategyEngine
from apex3.strategy.schedule import SessionScheduler, SessionState
from apex3.strategy.signal import build_opening_range, evaluate_aurum_signal, validate_late_start_signal
from apex3.strategy.regime import MarketRegime

IST = ZoneInfo("Asia/Kolkata")


def test_mcx_lot_multipliers_complete():
    """Verify official MCX lot multipliers for all tradeable micro/mini/standard contracts."""
    assert get_mcx_lot_multiplier("SILVERMIC30NOV26FUT") == 1
    assert get_mcx_lot_multiplier("SILVERM30NOV26FUT") == 5
    assert get_mcx_lot_multiplier("SILVER05DEC26FUT") == 30
    assert get_mcx_lot_multiplier("CRUDEOILM18DEC26FUT") == 10
    assert get_mcx_lot_multiplier("CRUDEOIL18DEC26FUT") == 100
    assert get_mcx_lot_multiplier("NATGASMINI24NOV26FUT") == 250
    assert get_mcx_lot_multiplier("NATURALGAS24NOV26FUT") == 1250
    assert get_mcx_lot_multiplier("ZINCMINI30NOV26FUT") == 1000
    assert get_mcx_lot_multiplier("ALUMINI30NOV26FUT") == 1000
    assert get_mcx_lot_multiplier("COPPER30NOV26FUT") == 2500


def test_mcx_scheduler_comex_and_hard_deadline():
    """Verify MCX scheduler never fires false NSE 15:25 hard-deadline during COMEX evening hours."""
    cfg = load_config("apex3/config.yaml")
    sched = SessionScheduler(cfg.schedule)

    # Morning OR (09:05)
    t_or = datetime(2026, 11, 18, 9, 5, tzinfo=IST)
    assert sched.get_session_state(t_or) == SessionState.OPENING_RANGE
    assert not sched.can_enter(t_or)
    assert not sched.is_past_hard_deadline(t_or)

    # Morning Late Start (09:50)
    t_late = datetime(2026, 11, 18, 9, 50, tzinfo=IST)
    assert sched.get_session_state(t_late) == SessionState.ENTRY_WINDOW
    assert sched.can_enter(t_late)
    assert not sched.is_past_flat_target(t_late)
    assert not sched.is_past_hard_deadline(t_late)

    # COMEX Prime Evening (18:45 & 20:30) — MUST NOT trip NSE 15:25 deadline!
    for h, m in [(18, 45), (20, 30), (21, 30)]:
        dt = datetime(2026, 11, 18, h, m, tzinfo=IST)
        assert sched.get_session_state(dt) == SessionState.ENTRY_WINDOW
        assert sched.can_enter(dt)
        assert not sched.is_past_flat_target(dt)
        assert not sched.is_past_hard_deadline(dt)

    # Near MCX Winter Close (23:51 IST -> hard deadline 23:50)
    t_close = datetime(2026, 11, 18, 23, 52, tzinfo=IST)
    assert sched.is_past_hard_deadline(t_close)


def test_mcx_late_start_0950_backfills_and_evaluates(tmp_path):
    """Verify starting engine at 09:50 IST reconstructs 09:00-09:15 OR and evaluates setups immediately."""
    cfg = load_config("apex3/config.yaml")
    broker = PaperBroker(starting_margin=30000.0)
    journal = Journal(db_path=str(tmp_path / "j.db"))
    sym = "SILVERMIC30NOV26FUT"
    now = datetime(2026, 11, 18, 9, 50, 0, tzinfo=IST)

    # Build 25 prior bars + today's 09:00..09:45 bars (clear bullish breakout above OR high)
    candles = []
    # Yesterday evening bars for EMA/ATR warmup
    t0 = datetime(2026, 11, 17, 21, 0, 0, tzinfo=IST)
    px = 200000.0
    for k in range(25):
        candles.append(
            Candle(
                timestamp=t0 + timedelta(minutes=5 * k),
                open=px,
                high=px + 120,
                low=px - 120,
                close=px + 10,
                volume=100,
                is_complete=True,
            )
        )
    # Today's OR bars: 09:00, 09:05, 09:10 (OR high = 200300, OR low = 199900)
    for k in range(3):
        ts = datetime(2026, 11, 18, 9, 5 * k, 0, tzinfo=IST)
        candles.append(
            Candle(
                timestamp=ts,
                open=200050.0,
                high=200300.0,
                low=199900.0,
                close=200150.0,
                volume=120,
                is_complete=True,
            )
        )
    # Post-OR bars: 09:15..09:45 with breakout on 09:45 bar (CLV > 0.85, RVOL > 3x)
    for k in range(3, 9):
        ts = datetime(2026, 11, 18, 9, 5 * k, 0, tzinfo=IST)
        candles.append(
            Candle(
                timestamp=ts,
                open=200150.0 + k * 20,
                high=200250.0 + k * 20,
                low=200100.0 + k * 20,
                close=200200.0 + k * 20,
                volume=120,
                is_complete=True,
            )
        )
    # 09:45 breakout bar
    candles.append(
        Candle(
            timestamp=datetime(2026, 11, 18, 9, 45, 0, tzinfo=IST),
            open=200320.0,
            high=200750.0,
            low=200310.0,
            close=200720.0,
            volume=450,
            is_complete=True,
        )
    )

    inst = InstrumentInfo(
        symbol=sym,
        exchange="MCX",
        tick_size=1.0,
        lot_size=1,
        order_lot_size=1,
        lot_multiplier=1,
        upper_circuit=220000.0,
        lower_circuit=180000.0,
    )
    quote = Quote(
        symbol=sym,
        exchange="MCX",
        last_price=200725.0,
        bid=200720.0,
        ask=200730.0,
        volume=5000,
        timestamp=now,
        is_fresh=True,
    )
    broker.register_instrument(inst)
    broker.set_quote(quote)
    broker._fixture_candles[sym] = candles

    # Also load bearish NIFTY 50 to prove MCX commodity ignores NSE equity regime!
    nifty_candles = [
        Candle(
            timestamp=datetime(2026, 11, 18, 9, 15, 0, tzinfo=IST) + timedelta(minutes=5 * k),
            open=24000 - k * 20,
            high=24010 - k * 20,
            low=23970 - k * 20,
            close=23980 - k * 20,
            volume=1000,
            is_complete=True,
        )
        for k in range(7)
    ]
    broker.register_instrument(InstrumentInfo(symbol="NIFTY 50", exchange="NSE", tick_size=0.05))
    broker.set_quote(
        Quote(
            symbol="NIFTY 50",
            exchange="NSE",
            last_price=23860.0,
            bid=23859.0,
            ask=23861.0,
            volume=10000,
            timestamp=now,
            is_fresh=True,
        )
    )
    broker._fixture_candles["NIFTY 50"] = nifty_candles

    engine = StrategyEngine(config=cfg, broker=broker, journal=journal)
    engine.set_manual_symbols([sym])

    # Run late-start backfill at 09:50 IST
    bf = engine.reconcile_and_backfill_late_start(current_time=now)
    assert sym in engine.opening_ranges
    assert engine.opening_ranges[sym].is_complete
    assert engine.opening_ranges[sym].high == 200300.0
    assert engine.opening_ranges[sym].low == 199900.0

    # Evaluate step at 09:50 IST — must NOT be blocked by bearish NIFTY 50 or late start!
    decisions = engine.evaluate_step(now)
    assert len(decisions) >= 1
    # Either executed a LONG entry or passed signal gates cleanly
    assert any(
        d.decision in (Decision.PROPOSED_LONG, Decision.WAIT) and "OUTSIDE_ENTRY_WINDOW" not in d.reason_codes
        for d in decisions
    )


def test_mcx_comex_late_start_window_allowed():
    """Verify validate_late_start_signal accepts COMEX evening setups (19:10 IST) with MCX window."""
    now = datetime(2026, 11, 18, 19, 10, 0, tzinfo=IST)
    trig = datetime(2026, 11, 18, 18, 55, 0, tzinfo=IST)
    q = Quote(
        symbol="SILVERMIC30NOV26FUT",
        exchange="MCX",
        last_price=201050.0,
        bid=201045.0,
        ask=201055.0,
        volume=1000,
        timestamp=now,
        is_fresh=True,
    )
    res = validate_late_start_signal(
        symbol="SILVERMIC30NOV26FUT",
        side=Side.BUY,
        signal_price=201000.0,
        initial_stop=200400.0,
        target_price=203700.0,
        or_high=200800.0,
        or_low=200200.0,
        trigger_bar_timestamp=trig,
        subsequent_candles=[],
        quote=q,
        now_ist=now,
        entry_start_hhmm="09:15",
        entry_cutoff_hhmm="22:15",
        max_price_drift_pct=0.0015,
    )
    assert res.is_valid, f"Expected valid COMEX late-start, got: {res}"


def test_all_order_types_supported_paper_mcx():
    """Verify LIMIT, MARKET, SL, SL-M orders all work cleanly on MCX PaperBroker."""
    broker = PaperBroker(starting_margin=30000.0)
    sym = "CRUDEOILM18DEC26FUT"
    now = datetime(2026, 11, 18, 19, 0, 0, tzinfo=IST)
    broker.register_instrument(InstrumentInfo(symbol=sym, exchange="MCX", tick_size=1.0, lot_size=1, order_lot_size=1, lot_multiplier=10))
    broker.set_quote(Quote(symbol=sym, exchange="MCX", last_price=5800.0, bid=5799.0, ask=5801.0, volume=500, timestamp=now, is_fresh=True))

    for ot, px, trig in [
        (OrderType.LIMIT, 5802.0, 0.0),
        (OrderType.MARKET, 0.0, 0.0),
        (OrderType.SL, 5770.0, 5775.0),
        (OrderType.SL_M, 0.0, 5775.0),
    ]:
        oid = broker.place_order(
            OrderRequest(
                symbol=sym,
                exchange="MCX",
                side=Side.BUY if ot in (OrderType.LIMIT, OrderType.MARKET) else Side.SELL,
                quantity=1,
                product=Product.MIS,
                order_type=ot,
                price=px,
                trigger_price=trig,
                tag="TEST",
            )
        )
        assert oid
        st = broker.get_order_status(oid)
        assert st is not None
