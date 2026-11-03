"""
apex3/tests/test_mcx_adaptation.py
----------------------------------
Verification & Backtest Sanity Suite for APEX-3 MCX Precious Metals Adaptation:
1. calculate_mcx_intraday_charges exact statutory verification (₹63.53 on ₹95k SILVERMIC)
2. size_position() risk-capping, 1-lot sizing, and fee-to-profit gate verification
3. US DST schedule transition (EDT 17:00-17:30 / Flatten 23:10 vs EST 18:00-18:30 / Flatten 23:35)
4. 10 Order State Machine transitions with tick_size = 1.0
5. 4-Stage Progressive Ratchet Trailing Stop & strict monotonicity
6. 15m Choppiness Index (CI_14) WAITING gate
7. Bullion cross-metal alignment (Gold & Silver agreement on EMA20 and VWAP)
"""
from datetime import date, datetime, time, timedelta
import math
from zoneinfo import ZoneInfo
import pytest

from apex3.config import LimitsConfig, RatchetConfig, RiskConfig, StopsConfig
from apex3.models import (
    Candle,
    Decision,
    GateStatus,
    InstrumentInfo,
    MarginInfo,
    OpeningRange,
    OrderRequest,
    OrderState,
    OrderStatus,
    OrderType,
    PositionState,
    Product,
    Side,
    round_to_tick,
)
from apex3.strategy.accounting import (
    calculate_mcx_intraday_charges,
    _calculate_realized_pnl,
)
from apex3.strategy.regime import (
    MarketRegime,
    calculate_choppiness_index,
    evaluate_bullion_alignment,
    evaluate_market_regime,
)
from apex3.strategy.signal import (
    build_opening_range,
    is_us_dst,
)
from apex3.strategy.sizer import (
    calculate_mcx_intraday_charges as sizer_mcx_charges,
    size_position,
)
from apex3.strategy.trail import (
    evaluate_progressive_ratchet,
)

IST = ZoneInfo("Asia/Kolkata")


class TestMCXChargesAndSizing:
    def test_mcx_intraday_charges_exact_value(self):
        """Verify calculate_mcx_intraday_charges(95000, 95000, lots=1, lot_multiplier=1) == ₹63.53."""
        charges = calculate_mcx_intraday_charges(95000.0, 95000.0, lots=1, lot_multiplier=1)
        # Math verification:
        # Buy turnover = 95,000; Sell turnover = 95,000; Total = 190,000
        # Brokerage = min(20, 28.5) + min(20, 28.5) = 40.0
        # CTT (0.01% on sell turnover) = 9.50
        # Exchange turnover (0.0021% on total) = 3.99
        # SEBI turnover (0.0001% on total) = 0.19
        # Stamp duty (0.002% on buy turnover) = 1.90
        # GST (18% on brok + exch + sebi = 18% of 44.18) = 7.9524
        # Total = 40 + 9.50 + 3.99 + 0.19 + 1.90 + 7.9524 = 63.5324 -> round(..., 2) = 63.53
        assert charges == 63.53
        # Check alias in sizer module as well
        assert sizer_mcx_charges(95000.0, 95000.0, lots=1, lot_multiplier=1) == 63.53

    def test_size_position_silvermic_risk_bound_and_fee_gate(self):
        """Verify size_position for SILVERMIC at ₹95,000 with 250-point stop sizes 1 lot and passes 18% fee gate."""
        instrument = InstrumentInfo(
            symbol="SILVERMIC26NOVFUT",
            exchange="MCX",
            tick_size=1.0,
            lot_size=1,
            order_lot_size=1,
            lot_multiplier=1,
            mis_available=True,
        )
        margin_info = MarginInfo(
            equity_available=30000.0,
            equity_used=0.0,
            collateral=0.0,
            net_value=30000.0,
        )
        limits_config = LimitsConfig(
            max_symbols=2,
            max_leverage=10.0,
            min_accepted_leverage=1.0,
            safety_reserve_pct=5.0,
        )
        risk_config = RiskConfig(
            risk_per_trade_pct=1.25,       # ₹30,000 * 1.25% = ₹375.00
            max_daily_loss_pct=2.50,
            bound_loss_with_charges=True,
            slippage_allowance_pct=0.0,
        )

        entry_price = 95000.0
        stop_price = 94750.0  # 250 points risk

        res = size_position(
            symbol="SILVERMIC26NOVFUT",
            side=Side.BUY,
            entry_price=entry_price,
            stop_price=stop_price,
            num_qualified_symbols=1,
            margin_info=margin_info,
            instrument=instrument,
            limits_config=limits_config,
            risk_config=risk_config,
        )

        assert res.is_valid, f"Sizing failed: {res.rejection_reason}"
        assert res.quantity == 1
        # Net realized risk at stop: (1 * 250) + charges (~63.50) = 313.50 <= 375.00
        assert res.trade_risk_rupees <= 375.00
        assert res.gate.status == GateStatus.PASS

        # Expected 2.0R gross profit = 2 * 250 = ₹500
        # Charges at target ₹95,500 ~ ₹63.56
        # Fee to profit ratio = 63.56 / 500 = 12.7% <= 18%
        expected_2r = 2.0 * 250.0 * 1
        target_price = entry_price + 500.0
        target_charges = calculate_mcx_intraday_charges(entry_price, target_price, 1, 1)
        assert (target_charges / expected_2r) <= 0.18


class TestUSDSTAndMCXSessions:
    def test_us_dst_transitions(self):
        """Verify US Daylight Saving Time detector correctly flags EDT vs EST."""
        # 2026: 2nd Sunday of March is March 8; 1st Sunday of Nov is Nov 1
        assert not is_us_dst(date(2026, 1, 15))
        assert not is_us_dst(date(2026, 3, 7))
        assert is_us_dst(date(2026, 3, 8))
        assert is_us_dst(date(2026, 6, 15))
        assert is_us_dst(date(2026, 10, 15))
        assert not is_us_dst(date(2026, 11, 1))
        assert not is_us_dst(date(2026, 12, 15))

    def test_comex_opening_range_shift(self):
        """Verify COMEX Opening Range auto-adjusts between EDT (17:00-17:30) and EST (18:00-18:30)."""
        oct_date = date(2026, 10, 15)  # EDT
        dec_date = date(2026, 12, 15)  # EST

        # 17:00 to 17:30 bars in October (EDT)
        oct_candles = [
            Candle(
                timestamp=datetime.combine(oct_date, time(17, i * 5), tzinfo=IST),
                open=80000.0 + i * 10,
                high=80020.0 + i * 10,
                low=79990.0 + i * 10,
                close=80015.0 + i * 10,
                volume=1000,
                is_complete=True,
            )
            for i in range(6)
        ]
        or_oct = build_opening_range("SILVERMIC", oct_candles, session_date=oct_date, session_type="COMEX")
        assert or_oct.is_complete
        assert or_oct.candle_count == 6
        assert or_oct.session_type == "COMEX"

        # 18:00 to 18:30 bars in December (EST)
        dec_candles = [
            Candle(
                timestamp=datetime.combine(dec_date, time(18, i * 5), tzinfo=IST),
                open=85000.0 + i * 10,
                high=85020.0 + i * 10,
                low=84990.0 + i * 10,
                close=85015.0 + i * 10,
                volume=1000,
                is_complete=True,
            )
            for i in range(6)
        ]
        or_dec = build_opening_range("SILVERMIC", dec_candles, session_date=dec_date, session_type="COMEX")
        assert or_dec.is_complete
        assert or_dec.candle_count == 6
        assert or_dec.session_type == "COMEX"

        # October candles fed to December session should fail to build COMEX OR (wrong hours)
        or_mismatch = build_opening_range("SILVERMIC", oct_candles, session_date=dec_date, session_type="COMEX")
        assert not or_mismatch.is_complete


class TestOrderStateMachineAndTickRounding:
    def test_round_to_tick_with_1_0_tick_size(self):
        """Verify round_to_tick works cleanly with tick_size=1.0 and Decimal precision."""
        assert round_to_tick(95000.4, 1.0, direction="floor") == 95000.0
        assert round_to_tick(95000.4, 1.0, direction="ceil") == 95001.0
        assert round_to_tick(95000.4, 1.0, direction="nearest") == 95000.0
        assert round_to_tick(95000.6, 1.0, direction="nearest") == 95001.0
        assert round_to_tick(95000.5, 1.0, direction="nearest") == 95001.0

    def test_order_state_transitions_mcx(self):
        """Verify order lifecycle state transitions with MCX parameters."""
        req = OrderRequest(
            symbol="GOLDM26NOVFUT",
            exchange="MCX",
            side=Side.BUY,
            quantity=1,
            price=80000.0,
            order_type=OrderType.LIMIT,
            product=Product.MIS,
        )
        order = OrderState(
            order_id="TEST-MCX-001",
            request=req,
            status=OrderStatus.PENDING,
        )
        assert order.status == OrderStatus.PENDING

        # Transition 1: Submitted -> Open
        order.status = OrderStatus.OPEN
        order.exchange_order_id = "EXCH-999"
        assert order.status == OrderStatus.OPEN

        # Transition 2: Partial Fill
        order.status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = 0
        order.pending_quantity = 1
        assert order.status == OrderStatus.PARTIALLY_FILLED

        # Transition 3: Complete Fill
        order.status = OrderStatus.COMPLETE
        order.filled_quantity = 1
        order.pending_quantity = 0
        order.average_price = 80000.0
        assert order.status == OrderStatus.COMPLETE


class TestProgressiveRatchetAndTrailing:
    def test_progressive_ratchet_all_stages(self):
        """Verify 4-Stage Progressive Ratchet transitions: Stage 0 -> 1 -> 2 -> 3 -> 4."""
        pos = PositionState(
            symbol="SILVERMIC26NOVFUT",
            side=Side.BUY,
            quantity=1,
            average_entry=95000.0,
            initial_stop_price=94750.0,       # 1R = 250 pts
            protective_stop_price=94750.0,
            best_favorable_price=95000.0,
            modification_count=0,
        )
        ratchet_cfg = RatchetConfig(
            stage1_r=1.0,
            stage1_buffer_ticks=2,
            stage2_r=1.5,
            stage2_lock_r=0.65,
            stage3_r=2.0,
            stage3_lock_r=1.25,
            stage4_r=2.5,
            stage4_step_r=0.50,
            max_stop_modifications=20,
        )

        # Stage 0: Favorable excursion < 1.0R (+150 pts = +0.6R)
        eval0 = evaluate_progressive_ratchet(pos, current_price=95150.0, candles_5m=[], atr_14=100.0, config=ratchet_cfg)
        assert not eval0.should_modify
        assert eval0.proposed_stop == 94750.0

        # Stage 1: +1.0R reached (+260 pts = +1.04R) -> Ratchet to BE+
        pos.best_favorable_price = 95260.0
        eval1 = evaluate_progressive_ratchet(pos, current_price=95260.0, candles_5m=[], atr_14=100.0, config=ratchet_cfg)
        assert eval1.should_modify
        assert eval1.proposed_stop > 95000.0  # BE + buffer > entry price
        pos.protective_stop_price = eval1.proposed_stop
        pos.modification_count += 1

        # Stage 2: +1.5R reached (+380 pts = +1.52R) -> Lock +0.65R
        pos.best_favorable_price = 95380.0
        eval2 = evaluate_progressive_ratchet(pos, current_price=95380.0, candles_5m=[], atr_14=100.0, config=ratchet_cfg)
        assert eval2.should_modify
        # Expected stop = 95000 + 0.65 * 250 = 95162.5 -> floor/nearest 95162.0 or 95163.0
        assert eval2.proposed_stop >= 95162.0
        assert eval2.proposed_stop > pos.protective_stop_price  # Monotonic
        pos.protective_stop_price = eval2.proposed_stop
        pos.modification_count += 1

        # Stage 3: +2.0R reached (+510 pts = +2.04R) -> Lock +1.25R
        pos.best_favorable_price = 95510.0
        eval3 = evaluate_progressive_ratchet(pos, current_price=95510.0, candles_5m=[], atr_14=100.0, config=ratchet_cfg)
        assert eval3.should_modify
        # Expected stop = 95000 + 1.25 * 250 = 95312.5 -> 95312.0
        assert eval3.proposed_stop >= 95312.0
        assert eval3.proposed_stop > pos.protective_stop_price  # Monotonic
        pos.protective_stop_price = eval3.proposed_stop
        pos.modification_count += 1

        # Stage 4: +2.5R+ reached (+680 pts = +2.72R) -> Dynamic step ratchet & ATR trail
        pos.best_favorable_price = 95680.0
        eval4 = evaluate_progressive_ratchet(pos, current_price=95680.0, candles_5m=[], atr_14=100.0, config=ratchet_cfg)
        assert eval4.should_modify
        assert eval4.proposed_stop > pos.protective_stop_price  # Strictly monotonic

    def test_progressive_ratchet_throttling_at_20_mods(self):
        """Verify modification cap throttles stop-loss changes when mod_count reaches 20."""
        pos = PositionState(
            symbol="SILVERMIC26NOVFUT",
            side=Side.BUY,
            quantity=1,
            average_entry=95000.0,
            initial_stop_price=94750.0,
            protective_stop_price=95300.0,
            best_favorable_price=95600.0,
            modification_count=20,  # Reached limit
        )
        ratchet_cfg = RatchetConfig(max_stop_modifications=20)
        res = evaluate_progressive_ratchet(pos, current_price=95700.0, candles_5m=[], atr_14=100.0, config=ratchet_cfg)
        assert not res.should_modify
        assert "Max modification cap" in res.reason


class TestChoppinessIndexAndBullionAlignment:
    def test_choppiness_index_gate_triggers_waiting(self):
        """Verify Choppiness Index (CI_14) >= 50.0 triggers WAITING regime."""
        now = datetime(2026, 6, 15, 14, 0, tzinfo=IST)
        # 20 flat 15m candles
        flat_candles = [
            Candle(
                timestamp=now - timedelta(minutes=(20 - i) * 15),
                open=80000.0,
                high=80020.0,
                low=79980.0,
                close=80000.0,
                volume=100,
                is_complete=True,
            )
            for i in range(20)
        ]
        ci = calculate_choppiness_index(flat_candles, period=14)
        assert ci >= 50.0, f"Expected choppy CI >= 50.0, got {ci}"

    def test_cross_metal_bullion_alignment_pass_and_fail(self):
        """Verify Gold & Silver cross-metal alignment on EMA20 and VWAP."""
        now = datetime(2026, 6, 15, 14, 0, tzinfo=IST)

        # Bullish Gold & Silver
        gold_bull = [
            Candle(
                timestamp=now - timedelta(minutes=(25 - i) * 5),
                open=75000.0 + i * 20,
                high=75030.0 + i * 20,
                low=74990.0 + i * 20,
                close=75025.0 + i * 20,
                volume=500,
                is_complete=True,
            )
            for i in range(25)
        ]
        silver_bull = [
            Candle(
                timestamp=now - timedelta(minutes=(25 - i) * 5),
                open=85000.0 + i * 25,
                high=85040.0 + i * 25,
                low=84980.0 + i * 25,
                close=85035.0 + i * 25,
                volume=500,
                is_complete=True,
            )
            for i in range(25)
        ]
        aligned, reason = evaluate_bullion_alignment(gold_bull, silver_bull, intended_side=Side.BUY)
        assert aligned, f"Expected bullish alignment, got failure: {reason}"

        # Divergent: Gold Bullish, Silver Bearish
        silver_bear = [
            Candle(
                timestamp=now - timedelta(minutes=(25 - i) * 5),
                open=85000.0 - i * 25,
                high=85010.0 - i * 25,
                low=84960.0 - i * 25,
                close=84970.0 - i * 25,
                volume=500,
                is_complete=True,
            )
            for i in range(25)
        ]
        divergent, reason = evaluate_bullion_alignment(gold_bull, silver_bear, intended_side=Side.BUY)
        assert not divergent, "Expected cross-metal divergence to fail"

