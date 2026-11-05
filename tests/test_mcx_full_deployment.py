"""Full Unit, Integration & Regression Verification Suite for AURUM-APEX (MCX).

Tests all 5 key aspects:
1. Unit Test: MCX Stop-Width & Dynamic Sizer Proof (SILVERMIC @ Rs 95,000)
2. Unit Test: Playbook B False-Trigger Prevention (VWAP/EMA20 reclaim)
3. Unit Test: 5-Stage Progressive Ratchet Execution (Stage 1 -> Stage 4)
4. Integration Test: Late Start at 09:50 IST & 19:15 IST (Winter COMEX)
"""
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from apex3.config import AppConfig, LimitsConfig, RiskConfig, SignalConfig, StopsConfig
from apex3.models import (
    Candle,
    Decision,
    InstrumentInfo,
    MarginInfo,
    OpeningRange,
    OrderMarginEstimate,
    OrderStatus,
    OrderType,
    Product,
    Quote,
    Side,
)
from apex3.strategy.engine import StrategyEngine
from apex3.strategy.regime import MarketRegime
from apex3.strategy.signal import (
    OpeningRange,
    build_opening_range,
    evaluate_aurum_signal,
    is_us_dst,
)
from apex3.strategy.sizer import calculate_mcx_intraday_charges, size_position
from apex3.strategy.trail import evaluate_progressive_ratchet

IST = ZoneInfo("Asia/Kolkata")


# ── TEST 1: MCX Stop-Width & Sizer Proof ──────────────────────────────────────


def test_mcx_stop_width_and_sizer_proof():
    """Verify SILVERMIC at Rs. 95,000 with stop = Rs. 94,750 sizes 1 lot and passes fee gate."""
    entry_price = 95000.0
    stop_price = 94750.0  # stop width = 250 / 95000 = 0.263% (inside [0.18%, 0.65%])
    stop_width_pct = (entry_price - stop_price) / entry_price
    assert 0.0018 <= stop_width_pct <= 0.0065

    # Capital: Rs. 30,000, max risk = 1.25% = Rs. 375
    margin_info = MarginInfo(
        equity_available=30000.0,
        equity_used=0.0,
        collateral=0.0,
        net_value=30000.0,
        timestamp=datetime.now(IST),
    )
    inst = InstrumentInfo(
        symbol="SILVERMIC26NOVFUT",
        exchange="MCX",
        tick_size=1.0,
        lot_size=1,
        order_lot_size=1,
        lot_multiplier=1,
        symbol_root="SILVERMIC",
    )
    limits_cfg = LimitsConfig(max_symbols=2, max_leverage=10.0, min_accepted_leverage=2.0, safety_reserve_pct=5.0)
    risk_cfg = RiskConfig(risk_per_trade_pct=1.25, max_fee_to_profit_ratio=0.18, bound_loss_with_charges=True)

    # Zerodha margin estimate for 1 lot: ~Rs. 8,500
    order_est = OrderMarginEstimate(required_margin=8500.0, leverage_available=11.1)

    res = size_position(
        symbol="SILVERMIC26NOVFUT",
        side=Side.BUY,
        entry_price=entry_price,
        stop_price=stop_price,
        num_qualified_symbols=1,
        margin_info=margin_info,
        instrument=inst,
        limits_config=limits_cfg,
        risk_config=risk_cfg,
        order_margin_estimate=order_est,
    )

    assert res.is_valid
    assert res.quantity == 1
    # Price risk = Rs. 250, statutory fees at target (~Rs 63.53)
    # Expected 2.0R gross profit = 2 * 250 = Rs. 500
    expected_profit_2r = 2.0 * abs(entry_price - stop_price)
    target_price = entry_price + expected_profit_2r
    charges = calculate_mcx_intraday_charges(entry_price, target_price, lots=1, lot_multiplier=1)
    fee_ratio = charges / expected_profit_2r
    assert fee_ratio <= 0.18, f"Fee ratio {fee_ratio:.2%} exceeded 18%"


# ── TEST 2: Playbook B False-Trigger Prevention ───────────────────────────────


def test_playbook_b_false_trigger_prevention():
    """Verify Playbook B does NOT trigger when below VWAP/EMA20, and DOES trigger when reclaimed."""
    session_d = date(2026, 11, 6)
    or_range = OpeningRange(
        symbol="SILVERMIC",
        high=95500.0,
        low=95000.0,
        is_complete=True,
        candle_count=3,
        session_type="MORNING",
    )
    inst = InstrumentInfo(symbol="SILVERMIC", exchange="MCX", tick_size=1.0, lot_size=1, lot_multiplier=1)
    sig_cfg = SignalConfig(max_spread_bps=25.0)

    # Base completed candles sweeping OR low (low = 94,880 -> sweep = 120 pts = 0.40x ATR)
    now_ist = datetime(2026, 11, 6, 10, 0, tzinfo=IST)
    c1 = Candle(timestamp=now_ist - timedelta(minutes=15), open=95100.0, high=95200.0, low=94950.0, close=95050.0, volume=5000, is_complete=True)
    c2 = Candle(timestamp=now_ist - timedelta(minutes=10), open=95050.0, high=95100.0, low=94880.0, close=94920.0, volume=8000, is_complete=True)

    # CASE A: Price sweeps low to 94,880 and closes at 94,980 (BELOW OR Low, and BELOW VWAP/EMA20 = 95,050)
    # MUST NOT TRIGGER
    c3_fail = Candle(timestamp=now_ist - timedelta(minutes=5), open=94920.0, high=94990.0, low=94900.0, close=94980.0, volume=9000, is_complete=True)
    q_fail = Quote(symbol="SILVERMIC", exchange="MCX", last_price=94980.0, bid=94979.0, ask=94981.0, volume=22000, timestamp=now_ist, is_fresh=True)

    eval_fail = evaluate_aurum_signal(
        symbol="SILVERMIC",
        candles_5m=[c1, c2, c3_fail],
        benchmark_candles_5m=[c1, c2, c3_fail],
        opening_range=or_range,
        market_regime=MarketRegime.BULLISH,
        instrument=inst,
        quote=q_fail,
        config=sig_cfg,
        is_entry_window=True,
    )
    assert eval_fail.decision == Decision.WAIT
    assert eval_fail.playbook != "SWEEP_RECLAIM" or eval_fail.decision != Decision.PROPOSED_LONG

    # CASE B: Price sweeps low to 94,880, then vigorously reclaims OR Low + 0.10 ATR (95,030+), closing at 95,120
    # Strong CLV >= 0.70, RVOL >= 2.0x, closing above VWAP and EMA20 -> MUST TRIGGER
    c3_pass = Candle(timestamp=now_ist - timedelta(minutes=5), open=94900.0, high=95130.0, low=94890.0, close=95120.0, volume=18000, is_complete=True)
    q_pass = Quote(symbol="SILVERMIC", exchange="MCX", last_price=95120.0, bid=95119.0, ask=95121.0, volume=31000, timestamp=now_ist, is_fresh=True)

    eval_pass = evaluate_aurum_signal(
        symbol="SILVERMIC",
        candles_5m=[c1, c2, c3_pass],
        benchmark_candles_5m=[c1, c2, c3_pass],
        opening_range=or_range,
        market_regime=MarketRegime.BULLISH,
        instrument=inst,
        quote=q_pass,
        config=sig_cfg,
        is_entry_window=True,
    )
    assert eval_pass.decision == Decision.PROPOSED_LONG
    assert eval_pass.playbook == "SWEEP_RECLAIM"
    assert eval_pass.relative_strength != 0.0


# ── TEST 3: 5-Stage Progressive Ratchet Execution ─────────────────────────────


def test_five_stage_progressive_ratchet_execution():
    """Simulate SILVERMIC Long (Entry=95,000, SL=94,750, 1R=250) progressing through Stages 1 to 4."""
    from apex3.models import PositionState
    stops_cfg = StopsConfig(trail_enabled=True, target_r=4.0)
    entry_px = 95000.0
    initial_sl = 94750.0
    r_val = 250.0
    tick = 1.0

    pos = PositionState(
        symbol="SILVERMIC",
        side=Side.BUY,
        quantity=1,
        average_entry=entry_px,
        initial_stop_price=initial_sl,
        protective_stop_price=initial_sl,
        protective_order_id="SL_123",
        best_favorable_price=entry_px,
        r_value=r_val,
    )

    # Stage 0: Open (0.50R, Current=95,125) -> No stop change
    ev0 = evaluate_progressive_ratchet(pos, current_price=95125.0, candles_5m=[], atr_14=100.0, config=stops_cfg, tick_size=tick)
    assert not ev0.should_modify
    assert ev0.ratchet_stage == 0

    # Stage 1: Breakeven+ Fee Lock (+1.0R, Current=95,250) -> Locks +0.12R (95,030)
    ev1 = evaluate_progressive_ratchet(pos, current_price=95250.0, candles_5m=[], atr_14=100.0, config=stops_cfg, tick_size=tick)
    assert ev1.should_modify
    assert ev1.ratchet_stage == 1
    assert ev1.proposed_stop >= entry_px + (0.10 * r_val)  # At least BE+

    # Stage 2: Structural Lock (+1.50R, Current=95,375) -> Locks +0.65R (95,162+)
    pos.protective_stop_price = ev1.proposed_stop
    pos.best_favorable_price = 95375.0
    ev2 = evaluate_progressive_ratchet(pos, current_price=95375.0, candles_5m=[], atr_14=100.0, config=stops_cfg, tick_size=tick)
    assert ev2.should_modify
    assert ev2.ratchet_stage == 2
    assert ev2.proposed_stop >= entry_px + (0.60 * r_val)

    # Stage 3: Dynamic Asymmetric Step (+2.20R, Current=95,550) -> Locks +1.30R (95,325+)
    pos.protective_stop_price = ev2.proposed_stop
    pos.best_favorable_price = 95550.0
    ev3 = evaluate_progressive_ratchet(pos, current_price=95550.0, candles_5m=[], atr_14=100.0, config=stops_cfg, tick_size=tick)
    assert ev3.should_modify
    assert ev3.ratchet_stage == 3
    assert ev3.proposed_stop >= entry_px + (1.20 * r_val)

    # Stage 4: Parabolic Harvest (+3.40R, Current=95,850) -> Locks +2.50R+
    pos.protective_stop_price = ev3.proposed_stop
    pos.best_favorable_price = 95850.0
    ev4 = evaluate_progressive_ratchet(pos, current_price=95850.0, candles_5m=[], atr_14=100.0, config=stops_cfg, tick_size=tick)
    assert ev4.should_modify
    assert ev4.ratchet_stage == 4
    assert ev4.proposed_stop >= entry_px + (2.40 * r_val)


# ── TEST 4: Integration Test — Late Start at 09:50 IST & 19:15 IST (Winter) ───


def test_late_start_integration_0950_and_1915_winter():
    """Verify engine handles late start at 09:50 IST (Morning) and 19:15 IST (Winter COMEX)."""
    # 1. Test is_us_dst for Winter (November 6, 2026 is after 1st Sunday of Nov -> Standard Time EST)
    d_winter = date(2026, 11, 6)
    assert not is_us_dst(d_winter), "Nov 6, 2026 must be Winter Standard Time (EST)"

    mock_broker = MagicMock()
    mock_broker.is_session_valid.return_value = True
    mock_broker.get_positions.return_value = []
    mock_broker.get_orders.return_value = []
    mock_broker.get_trades.return_value = []
    mock_broker.get_margins.return_value = MarginInfo(equity_available=30000.0, net_value=30000.0)

    # Instrument info
    inst = InstrumentInfo(symbol="SILVERMIC", exchange="MCX", tick_size=1.0, lot_size=1, lot_multiplier=1, upper_circuit=105000.0, lower_circuit=85000.0)
    mock_broker.get_instrument_info.return_value = inst
    mock_broker.estimate_order_margin.return_value = OrderMarginEstimate(required_margin=8500.0, leverage_available=11.1)

    from apex3.config import load_config
    app_cfg = load_config("apex3/config.yaml")
    object.__setattr__(app_cfg, "exchange", "MCX")

    mock_journal = MagicMock()
    engine = StrategyEngine(config=app_cfg, broker=mock_broker, journal=mock_journal)
    engine.set_manual_symbols(["SILVERMIC"])

    # PART A: Late start at 09:50 IST
    # Build completed candles covering 09:00 to 09:45
    t_0950 = datetime(2026, 11, 6, 9, 50, tzinfo=IST)
    morning_c = [
        Candle(timestamp=datetime(2026, 11, 6, 9, 0, tzinfo=IST), open=95000.0, high=95150.0, low=94950.0, close=95100.0, volume=5000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 9, 5, tzinfo=IST), open=95100.0, high=95200.0, low=95050.0, close=95180.0, volume=6000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 9, 10, tzinfo=IST), open=95180.0, high=95220.0, low=95120.0, close=95200.0, volume=7000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 9, 15, tzinfo=IST), open=95200.0, high=95230.0, low=95180.0, close=95210.0, volume=4000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 9, 40, tzinfo=IST), open=95210.0, high=95300.0, low=95200.0, close=95280.0, volume=15000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 9, 45, tzinfo=IST), open=95280.0, high=95310.0, low=95260.0, close=95290.0, volume=12000, is_complete=True),
    ]
    mock_broker.get_historical.return_value = morning_c
    mock_broker.get_quote.return_value = Quote(symbol="SILVERMIC", exchange="MCX", last_price=95290.0, bid=95289.0, ask=95291.0, volume=49000, timestamp=t_0950, is_fresh=True)
    mock_broker.get_quotes_batch.return_value = {"SILVERMIC": mock_broker.get_quote.return_value}

    backfill_res = engine.reconcile_and_backfill_late_start(current_time=t_0950)
    assert "SILVERMIC" in backfill_res["backfilled_symbols"]
    or_rebuilt = engine.opening_ranges.get("SILVERMIC")
    assert or_rebuilt is not None
    assert or_rebuilt.high == 95220.0  # OR High from 09:00 - 09:15
    assert or_rebuilt.session_type == "MORNING"

    # PART B: Winter COMEX late start at 19:15 IST (Winter: OR_COMEX is 18:00 - 18:30 IST)
    t_1915 = datetime(2026, 11, 6, 19, 15, tzinfo=IST)
    comex_c = [
        Candle(timestamp=datetime(2026, 11, 6, 18, 0, tzinfo=IST), open=95400.0, high=95550.0, low=95350.0, close=95500.0, volume=10000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 18, 5, tzinfo=IST), open=95500.0, high=95600.0, low=95480.0, close=95580.0, volume=12000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 18, 10, tzinfo=IST), open=95580.0, high=95620.0, low=95520.0, close=95600.0, volume=11000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 18, 15, tzinfo=IST), open=95600.0, high=95650.0, low=95580.0, close=95630.0, volume=14000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 18, 20, tzinfo=IST), open=95630.0, high=95680.0, low=95610.0, close=95670.0, volume=15000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 18, 25, tzinfo=IST), open=95670.0, high=95700.0, low=95640.0, close=95680.0, volume=16000, is_complete=True),
        Candle(timestamp=datetime(2026, 11, 6, 19, 10, tzinfo=IST), open=95680.0, high=95850.0, low=95670.0, close=95820.0, volume=25000, is_complete=True),
    ]
    mock_broker.get_historical.return_value = comex_c
    mock_broker.get_quote.return_value = Quote(symbol="SILVERMIC", exchange="MCX", last_price=95820.0, bid=95819.0, ask=95821.0, volume=103000, timestamp=t_1915, is_fresh=True)
    mock_broker.get_quotes_batch.return_value = {"SILVERMIC": mock_broker.get_quote.return_value}

    backfill_comex = engine.reconcile_and_backfill_late_start(current_time=t_1915)
    or_comex = engine.opening_ranges.get("SILVERMIC")
    assert or_comex is not None
    assert or_comex.session_type == "COMEX"
    assert or_comex.high == 95700.0  # Max high of 18:00 - 18:30 in winter
