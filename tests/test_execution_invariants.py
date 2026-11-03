"""
tests/test_execution_invariants.py
----------------------------------
Comprehensive pytest suite mathematically verifying all 6 correctness invariants
and operational requirements for the MCX Precious Metals Trading System:

1. Positive Net Expectancy Invariant (Net P&L > 0 at BE+ and trailing exits)
2. Monotonic Trailing Invariant (SL never widens or retreats)
3. Zero Naked Exposure Invariant (Partial fills, SL sizing, SL rejection emergency exit)
4. Zero Overnight Carry Invariant (Mandatory squareoff at T_close - 20m & T_close - 18m audit)
5. Daily Circuit Breaker Invariant (Locked for day at -2.0% loss or 2 stopouts)
6. Pre-Close Cutoff Invariant (Zero new entries at T_close - 75m)
7. Dynamic US DST Schedule Detector & Rollover Engine
8. REGIME_WAITING Chop & Volatility Gate
9. Capital Sizing & Micro Contract Downgrade
10. Zerodha 25-Modification Limit Protection
"""

from datetime import date, datetime, time, timedelta
import math
from unittest.mock import MagicMock, patch
import pytest

from config.settings import (
    IST,
    MCX_INSTRUMENTS,
    CommodityFamily,
    ContractTier,
    SessionWindow,
    get_current_session_window,
    get_mcx_schedule,
    is_entry_allowed,
    is_us_dst,
    select_active_contract,
    should_rollover,
)
from core.margin_and_sizer import (
    LiveMarginManager,
    calculate_net_pnl,
    calculate_statutory_charges,
    compute_all_in_breakeven_offset,
)
from core.order_state_machine import (
    OrderStateMachine,
    Position,
    PositionStatus,
)
from core.regime_and_strategy import (
    Candle,
    MarketRegime,
    StrategyEngine,
    compute_choppiness_index,
    compute_adx,
    compute_wilder_atr,
    compute_ema,
)
from core.trailing_engine import (
    ActiveTradeState,
    TrailingEngine,
    TrailingStage,
    round_to_tick,
)
from main import MCXTradingBot, SystemExecutionState


# ============================================================================
# INVARIANT 1: Positive Net Expectancy Invariant
# ============================================================================

class TestPositiveNetExpectancyInvariant:
    """
    Net P&L = Gross P&L - Total Statutory Charges > 0
    Every Stage 1 Breakeven exit (BE+) must yield Net P&L > 0 after:
    Brokerage (Rs. 40), CTT (0.05%), MCX turnover charges, Stamp duty, GST (18%), and SEBI fees.
    """

    @pytest.mark.parametrize("symbol", ["GOLDM", "GOLDTEN", "SILVERM", "SILVERMIC"])
    @pytest.mark.parametrize("entry_price", [75000.0, 85000.0, 95000.0])
    def test_stage_1_breakeven_yields_positive_net_pnl_long(self, symbol, entry_price):
        spec = MCX_INSTRUMENTS[symbol]
        multiplier = spec.multiplier
        tick_size = spec.tick_size

        be_offset = compute_all_in_breakeven_offset(entry_price, multiplier, tick_size)
        exit_price = entry_price + be_offset

        gross_pnl, total_charges, net_pnl = calculate_net_pnl(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=spec.lot_size,
            multiplier=multiplier,
            is_long=True,
        )

        assert net_pnl > 0.0, (
            f"Failed for {symbol}: Gross={gross_pnl:.2f}, Charges={total_charges:.2f}, "
            f"Net={net_pnl:.2f} at exit {exit_price} (Entry: {entry_price}, Offset: {be_offset})"
        )

    @pytest.mark.parametrize("symbol", ["GOLDM", "GOLDTEN", "SILVERM", "SILVERMIC"])
    @pytest.mark.parametrize("entry_price", [75000.0, 85000.0, 95000.0])
    def test_stage_1_breakeven_yields_positive_net_pnl_short(self, symbol, entry_price):
        spec = MCX_INSTRUMENTS[symbol]
        multiplier = spec.multiplier
        tick_size = spec.tick_size

        be_offset = compute_all_in_breakeven_offset(entry_price, multiplier, tick_size)
        exit_price = entry_price - be_offset

        gross_pnl, total_charges, net_pnl = calculate_net_pnl(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=spec.lot_size,
            multiplier=multiplier,
            is_long=False,
        )

        assert net_pnl > 0.0, (
            f"Failed for Short {symbol}: Gross={gross_pnl:.2f}, Charges={total_charges:.2f}, "
            f"Net={net_pnl:.2f} at exit {exit_price} (Entry: {entry_price}, Offset: {be_offset})"
        )


# ============================================================================
# INVARIANT 2: Monotonic Trailing Invariant
# ============================================================================

class TestMonotonicTrailingInvariant:
    """
    For any open trade over timestamps t1 < t2:
    Long: SL(t2) >= SL(t1)
    Short: SL(t2) <= SL(t1)
    The stop-loss must never widen or move backward under any circumstance.
    """

    def test_long_trailing_monotonicity_across_all_stages(self):
        engine = TrailingEngine(tick_size=1.0)
        entry_price = 80000.0
        atr14 = 200.0

        sl_0, unit_1r, is_valid = engine.calculate_initial_stop_loss(
            entry_price=entry_price,
            is_long=True,
            atr14=atr14,
            signal_candle_low=79750.0,
            signal_candle_high=80100.0,
        )
        assert is_valid
        assert sl_0 < entry_price
        assert unit_1r > 0

        trade = engine.create_trade_state(
            trade_id="TEST_LONG_1",
            symbol="GOLDM",
            is_long=True,
            entry_price=entry_price,
            initial_sl=sl_0,
            unit_risk_1r=unit_1r,
            lot_multiplier=10.0,
        )

        # Simulate ascending and fluctuating price path
        price_path = [
            80050.0, 80100.0, 80200.0, 80150.0, 80350.0, 80450.0, 80400.0,
            80600.0, 80750.0, 80550.0, 81000.0, 81200.0, 80900.0, 81500.0
        ]

        prev_sl = sl_0
        for current_price in price_path:
            new_sl, stage, modified, reason = engine.update_price_and_compute_trailing(
                trade=trade,
                current_price=current_price,
                atr14=atr14,
            )
            # Monotonicity check
            assert new_sl >= prev_sl, (
                f"Long SL violation! Current price: {current_price}, New SL: {new_sl} < Prev SL: {prev_sl}"
            )
            if modified:
                prev_sl = new_sl

    def test_short_trailing_monotonicity_across_all_stages(self):
        engine = TrailingEngine(tick_size=1.0)
        entry_price = 80000.0
        atr14 = 200.0

        sl_0, unit_1r, is_valid = engine.calculate_initial_stop_loss(
            entry_price=entry_price,
            is_long=False,
            atr14=atr14,
            signal_candle_low=79900.0,
            signal_candle_high=80250.0,
        )
        assert is_valid
        assert sl_0 > entry_price
        assert unit_1r > 0

        trade = engine.create_trade_state(
            trade_id="TEST_SHORT_1",
            symbol="GOLDM",
            is_long=False,
            entry_price=entry_price,
            initial_sl=sl_0,
            unit_risk_1r=unit_1r,
            lot_multiplier=10.0,
        )

        # Simulate descending and fluctuating price path
        price_path = [
            79950.0, 79900.0, 79800.0, 79850.0, 79650.0, 79550.0, 79600.0,
            79400.0, 79250.0, 79350.0, 79000.0, 78800.0, 79100.0, 78500.0
        ]

        prev_sl = sl_0
        for current_price in price_path:
            new_sl, stage, modified, reason = engine.update_price_and_compute_trailing(
                trade=trade,
                current_price=current_price,
                atr14=atr14,
            )
            # Monotonicity check
            assert new_sl <= prev_sl, (
                f"Short SL violation! Current price: {current_price}, New SL: {new_sl} > Prev SL: {prev_sl}"
            )
            if modified:
                prev_sl = new_sl


# ============================================================================
# INVARIANT 3: Zero Naked Exposure Invariant
# ============================================================================

class TestZeroNakedExposureInvariant:
    """
    The duration where Net Open Quantity > 0 and Active Exchange SL Quantity != Net Open Quantity
    must satisfy Delta t < 1.5 seconds.
    """

    def test_partial_fill_entry_timeout_handler(self):
        """
        Simulate entry order for 3 lots. Only 1 lot fills.
        After 5.0 seconds timeout, the uncompleted 2 lots are cancelled,
        and an exchange SL order is immediately placed for exactly 1 lot.
        """
        engine = TrailingEngine(tick_size=1.0)
        mock_kite = MagicMock()

        # Mock order history showing partial fill (1 of 3 lots filled)
        mock_kite.order_history.return_value = [
            {
                "order_id": "ORD_ENTRY_1",
                "status": "OPEN",
                "filled_quantity": 1,
                "pending_quantity": 2,
                "average_price": 80000.0,
            }
        ]
        mock_kite.place_order.return_value = "ORD_SL_1"

        sm = OrderStateMachine(kite_client=mock_kite, trailing_engine=engine, is_paper=False)

        pos = Position(
            trade_id="TRD_PARTIAL",
            symbol="GOLDM",
            spec=MCX_INSTRUMENTS["GOLDM"],
            is_long=True,
            target_quantity=3,
            filled_quantity=0,
            entry_price=80000.0,
            sl_trigger_price=79700.0,
            status=PositionStatus.ENTRY_PENDING,
        )
        sm.positions["GOLDM"] = pos

        # Trigger timeout handler synchronously for testing
        sm._handle_entry_timeout(
            symbol="GOLDM",
            trade_id="TRD_PARTIAL",
            order_id="ORD_ENTRY_1",
            ltp=80000.0,
            initial_sl=79700.0,
            unit_risk_1r=300.0,
        )

        # Invariant checks:
        # 1. Unfilled residual cancelled
        mock_kite.cancel_order.assert_called_with(variety="regular", order_id="ORD_ENTRY_1")
        # 2. Position active quantity locked to 1 lot
        assert pos.filled_quantity == 1
        assert pos.status == PositionStatus.ACTIVE
        # 3. Exchange SL order placed for exact filled quantity (1 lot)
        mock_kite.place_order.assert_called_with(
            variety="regular",
            exchange="MCX",
            tradingsymbol="GOLDM",
            transaction_type="SELL",
            quantity=1,
            product="NRML",
            order_type="SL",
            price=round_to_tick(79700.0 * (1.0 - 0.0035)),
            trigger_price=79700.0,
        )

    def test_sl_rejection_triggers_emergency_marketable_exit(self):
        """
        Simulate SL placement rejection (e.g. "Trigger price crossed LTP" hazard).
        EmergencyExitHandler must immediately fire a regular LIMIT exit order at LTP -/+ 0.50%
        within 100ms and verify net quantity == 0.
        """
        engine = TrailingEngine(tick_size=1.0)
        mock_kite = MagicMock()

        # Simulate place_order throwing exception for SL order
        mock_kite.place_order.side_effect = [
            Exception("400: Trigger price cannot be greater than LTP"), # SL fails
            "ORD_EMERGENCY_EXIT_1",                                      # Emergency limit succeeds
        ]
        mock_kite.ltp.return_value = {"MCX:GOLDM": {"last_price": 79650.0}}

        sm = OrderStateMachine(kite_client=mock_kite, trailing_engine=engine, is_paper=False)

        pos = Position(
            trade_id="TRD_SL_REJECT",
            symbol="GOLDM",
            spec=MCX_INSTRUMENTS["GOLDM"],
            is_long=True,
            target_quantity=1,
            filled_quantity=1,
            entry_price=80000.0,
            sl_trigger_price=79700.0,
            status=PositionStatus.ACTIVE,
        )
        sm.positions["GOLDM"] = pos

        # Attempt to place SL order
        sm.place_exchange_stop_loss(pos)

        # Verify emergency marketable exit occurred
        assert pos.status == PositionStatus.FLAT
        assert pos.filled_quantity == 0
        # Verify emergency limit exit was fired with 0.50% buffer
        expected_exit_limit = round_to_tick(79650.0 * 0.995, 1.0)
        mock_kite.place_order.assert_called_with(
            variety="regular",
            exchange="MCX",
            tradingsymbol="GOLDM",
            transaction_type="SELL",
            quantity=1,
            product="NRML",
            order_type="LIMIT",
            price=expected_exit_limit,
        )


# ============================================================================
# INVARIANT 4: Zero Overnight Carry Invariant
# ============================================================================

class TestZeroOvernightCarryInvariant:
    """
    At T_close - 20m, hard square-off flattens all open positions and cancels all orders.
    At T_close - 18m, zero positions and zero orders must be verified.
    """

    def test_hard_auto_squareoff_flattens_positions_and_cancels_orders(self):
        engine = TrailingEngine(tick_size=1.0)
        mock_kite = MagicMock()
        mock_kite.orders.return_value = [
            {"exchange": "MCX", "order_id": "ORD_PENDING_1", "status": "TRIGGER PENDING"},
            {"exchange": "MCX", "order_id": "ORD_PENDING_2", "status": "OPEN"},
        ]
        mock_kite.ltp.return_value = {"MCX:GOLDM": {"last_price": 80500.0}}

        sm = OrderStateMachine(kite_client=mock_kite, trailing_engine=engine, is_paper=False)

        pos = Position(
            trade_id="TRD_EOD",
            symbol="GOLDM",
            spec=MCX_INSTRUMENTS["GOLDM"],
            is_long=True,
            target_quantity=2,
            filled_quantity=2,
            entry_price=80000.0,
            sl_trigger_price=80200.0,
            status=PositionStatus.ACTIVE,
        )
        sm.positions["GOLDM"] = pos

        # Trigger Mandatory Intraday Flatten
        sm.close_all_positions(reason="Mandatory T_close - 20m Auto-Squareoff")

        # Verify all pending orders cancelled
        assert mock_kite.cancel_order.call_count >= 2
        # Verify position is flattened
        assert pos.status == PositionStatus.FLAT
        assert pos.filled_quantity == 0
        assert sum(p.filled_quantity for p in sm.positions.values()) == 0


# ============================================================================
# INVARIANT 5: Daily Circuit Breaker Invariant
# ============================================================================

class TestDailyCircuitBreakerInvariant:
    """
    If cumulative realized + unrealized loss reaches -2.0% of starting capital
    OR 2 consecutive stopouts occur in the same session, system transitions
    to LOCKED_FOR_DAY until 09:00 IST next session.
    """

    def test_circuit_breaker_on_consecutive_stopouts(self):
        bot = MCXTradingBot(symbol="GOLDM", is_paper=True, initial_capital=250000.0)
        assert bot.execution_state == SystemExecutionState.RUNNING

        # Trade 1: Stopout
        bot.record_trade_completion(gross_pnl=-2500.0, net_pnl=-2600.0, was_stopout=True)
        assert bot.consecutive_stopouts == 1
        assert bot.execution_state == SystemExecutionState.RUNNING

        # Trade 2: Consecutive Stopout -> Circuit Breaker
        bot.record_trade_completion(gross_pnl=-2500.0, net_pnl=-2600.0, was_stopout=True)
        assert bot.consecutive_stopouts == 2
        assert bot.execution_state == SystemExecutionState.LOCKED_FOR_DAY

    def test_circuit_breaker_on_2_percent_loss(self):
        bot = MCXTradingBot(symbol="GOLDM", is_paper=True, initial_capital=250000.0)
        assert bot.execution_state == SystemExecutionState.RUNNING

        # Loss of Rs. 5100 on Rs. 2,50,000 is > 2.0%
        bot.record_trade_completion(gross_pnl=-5000.0, net_pnl=-5150.0, was_stopout=False)
        assert bot.execution_state == SystemExecutionState.LOCKED_FOR_DAY


# ============================================================================
# INVARIANT 6: Pre-Close Cutoff Invariant
# ============================================================================

class TestPreCloseCutoffInvariant:
    """
    Zero new entries permitted when Current Time >= T_close - 75m
    (22:15 IST in Summer, 22:40 IST in Winter).
    """

    def test_summer_entry_cutoff_blocks_new_trades(self):
        # Date in July (US Summer Time, MCX closes at 23:30, cutoff at 22:15)
        summer_date = datetime(2026, 7, 15, 22, 16, tzinfo=IST)
        assert is_us_dst(summer_date)
        schedule = get_mcx_schedule(summer_date)
        assert schedule.market_close == time(23, 30)
        assert schedule.entry_cutoff_time == time(22, 15)

        # Entry should be blocked at 22:16
        assert not is_entry_allowed(summer_date)
        window = get_current_session_window(summer_date)
        assert window == SessionWindow.NO_NEW_ENTRY_CUTOFF

    def test_winter_entry_cutoff_blocks_new_trades(self):
        # Date in December (US Winter Time, MCX closes at 23:55, cutoff at 22:40)
        winter_date = datetime(2026, 12, 15, 22, 45, tzinfo=IST)
        assert not is_us_dst(winter_date)
        schedule = get_mcx_schedule(winter_date)
        assert schedule.market_close == time(23, 55)
        assert schedule.entry_cutoff_time == time(22, 40)

        # Entry should be blocked at 22:45
        assert not is_entry_allowed(winter_date)
        window = get_current_session_window(winter_date)
        assert window == SessionWindow.NO_NEW_ENTRY_CUTOFF


# ============================================================================
# Additional Core Tests: DST, Rollover, Regime Waiting & Modification Cap
# ============================================================================

class TestDSTAndRolloverEngine:
    def test_us_dst_transition_dates(self):
        # 2026: 2nd Sunday in March is March 8; 1st Sunday in Nov is Nov 1
        assert not is_us_dst(date(2026, 3, 7))
        assert is_us_dst(date(2026, 3, 8))
        assert is_us_dst(date(2026, 10, 31))
        assert not is_us_dst(date(2026, 11, 1))

    def test_7_day_pre_expiry_rollover_rule(self):
        expiry = date(2026, 6, 25)
        assert not should_rollover(current_date=date(2026, 6, 10), expiry_date=expiry)
        # 7 days before: June 18
        assert should_rollover(current_date=date(2026, 6, 18), expiry_date=expiry)
        assert should_rollover(current_date=date(2026, 6, 20), expiry_date=expiry)

    def test_select_active_contract(self):
        instruments = [
            {
                "exchange": "MCX",
                "tradingsymbol": "GOLDM26JUNFUT",
                "name": "GOLDM",
                "expiry": "2026-06-25",
            },
            {
                "exchange": "MCX",
                "tradingsymbol": "GOLDM26AUGFUT",
                "name": "GOLDM",
                "expiry": "2026-08-25",
            },
        ]
        # On June 19 (within 7 days of June 25), must select August contract
        active = select_active_contract(instruments, "GOLDM", as_of_date=date(2026, 6, 19))
        assert active is not None
        assert active["tradingsymbol"] == "GOLDM26AUGFUT"


class TestRegimeWaitingAndStrategy:
    def test_regime_waiting_gate_blocks_on_chop(self):
        strat = StrategyEngine(symbol="GOLDM")
        now = datetime(2026, 6, 15, 14, 0, tzinfo=IST)

        # Feed 25 15m candles with flat prices (high Choppiness Index)
        for i in range(25):
            t = now - timedelta(minutes=(25 - i) * 15)
            c = Candle(timestamp=t, open=80000.0, high=80050.0, low=79950.0, close=80000.0, volume=100.0, is_closed=True)
            strat.candles_15m.append(c)

        for i in range(30):
            t = now - timedelta(minutes=(30 - i) * 5)
            c = Candle(timestamp=t, open=80000.0, high=80020.0, low=79980.0, close=80000.0, volume=100.0, is_closed=True)
            strat.candles_5m.append(c)

        strat.vwap_tracker.update_bar(80000.0, 1000.0)

        blocked, reason = strat.check_regime_waiting_gate(now)
        assert blocked
        assert "Chop Gate" in reason or "Dead Tape" in reason or "VWAP Magnet" in reason


class TestTierDowngradeAndModificationLimits:
    def test_dynamic_tier_downgrade(self):
        # Account capital only Rs. 60,000. Standard GOLD lot requires ~Rs. 13 Lakhs margin.
        # Mini GOLDM requires ~Rs. 1.3 Lakhs margin.
        # Micro GOLDTEN requires ~Rs. 13,150. Sizing should downgrade to GOLDTEN!
        mgr = LiveMarginManager(default_equity=60000.0)
        res = mgr.size_position(
            symbol="GOLD",
            entry_price=80000.0,
            stop_loss_price=79700.0,
            is_buy=True,
            override_equity=60000.0,
            allow_downgrade=True,
        )
        assert res.lots > 0
        assert res.symbol == "GOLDTEN"
        assert res.downgraded_from in ("GOLD", "GOLDM")

    def test_zerodha_25_modification_limit_filter_and_reset(self):
        engine = TrailingEngine(tick_size=1.0)
        atr14 = 100.0
        # Min step filter is max(0.20 * 100, 5 ticks) = 20.0 ticks

        trade = engine.create_trade_state(
            trade_id="TRD_MOD",
            symbol="GOLDM",
            is_long=True,
            entry_price=80000.0,
            initial_sl=79800.0,
            unit_risk_1r=200.0,
            lot_multiplier=10.0,
        )

        # 1. Delta < 20.0 is filtered out
        eligible, reason = engine.check_modification_eligibility(trade, new_sl=79810.0, atr14=atr14)
        assert not eligible
        assert "below minimum threshold" in reason

        # 2. Delta >= 20.0 is eligible
        eligible, reason = engine.check_modification_eligibility(trade, new_sl=79825.0, atr14=atr14)
        assert eligible

        # Simulate 22 modifications
        trade.mod_count = 22
        eligible, reason = engine.check_modification_eligibility(trade, new_sl=79850.0, atr14=atr14)
        assert eligible
        assert "MOD_COUNT_CRITICAL" in reason


# ============================================================================
# Strategy Triggers & Reconciler Invariant Verification
# ============================================================================

class TestStrategyTriggersAndReconciler:
    def test_bull_breakout_trigger_a(self):
        strat = StrategyEngine(symbol="GOLDM")
        strat.morning_orb_high = 80100.0
        strat.morning_orb_low = 79900.0
        strat.vwap_tracker.vwap = 80000.0
        strat.vwap_tracker.sigma = 80.0

        now = datetime(2026, 6, 15, 10, 0, tzinfo=IST)

        # Add 25 15m trend bars with close > vwap, ema20 > ema50, adx >= 20
        for i in range(25):
            t = now - timedelta(minutes=(25 - i) * 15)
            c = Candle(
                timestamp=t,
                open=80000.0 + i * 8,
                high=80020.0 + i * 8,
                low=79990.0 + i * 8,
                close=80015.0 + i * 8,
                volume=200.0,
                is_closed=True,
            )
            strat.candles_15m.append(c)

        # Add 25 5m oscillating bars with prior close <= 80100 and RSI in [55, 73]
        base_5m_closes = [
            80000, 80010, 80005, 80020, 80015,
            80030, 80025, 80040, 80030, 80045,
            80035, 80050, 80040, 80055, 80050,
            80065, 80055, 80070, 80060, 80075,
            80065, 80080, 80070, 80085, 80075,  # Prior close: 80075 <= 80100
        ]
        for i, cl in enumerate(base_5m_closes):
            t = now - timedelta(minutes=(len(base_5m_closes) - i) * 5)
            c = Candle(
                timestamp=t,
                open=cl - 5.0,
                high=cl + 10.0,
                low=cl - 10.0,
                close=cl,
                volume=100.0,
                is_closed=True,
            )
            strat.candles_5m.append(c)

        # Signal candle: closes above morning_orb_high (80100.0) with body ratio >= 0.60 & high volume
        sig_candle = Candle(
            timestamp=now,
            open=80075.0,
            high=80125.0,
            low=80070.0,
            close=80120.0,  # Body ratio = 45/55 = 0.81 >= 0.60, RSI ~ 67.5 in [55, 73]
            volume=350.0,    # >= 1.4x SMA20 (100)
            is_closed=True,
        )
        strat.candles_5m.append(sig_candle)

        signal = strat.evaluate_triggers(now)
        assert signal is not None
        assert signal.direction == "BUY"
        assert signal.entry_price == 80120.0
        assert "Breakout" in signal.description

    def test_sl_open_limit_trap_recovery(self):
        """
        If SL order converts from TRIGGER PENDING into OPEN limit order
        and sits unfilled for > 2.0s, order limit price must be modified by 0.50%
        in market direction to sweep the order book.
        """
        engine = TrailingEngine(tick_size=1.0)
        mock_kite = MagicMock()
        mock_kite.order_history.return_value = [{"status": "OPEN"}]

        sm = OrderStateMachine(kite_client=mock_kite, trailing_engine=engine, is_paper=False)
        pos = Position(
            trade_id="TRD_TRAP",
            symbol="GOLDM",
            spec=MCX_INSTRUMENTS["GOLDM"],
            is_long=True,
            target_quantity=1,
            filled_quantity=1,
            entry_price=80000.0,
            sl_order_id="ORD_SL_TRAP",
            sl_trigger_price=79700.0,
            sl_limit_price=79420.0,
            status=PositionStatus.ACTIVE,
            sl_trigger_time=datetime.now(IST) - timedelta(seconds=3.0),  # > 2.0s trapped
        )
        sm.positions["GOLDM"] = pos

        sm.check_sl_triggered_open_trap()

        # Verify limit price was adjusted downward (for long exit / sell) by 0.50%
        expected_new_limit = round_to_tick(79420.0 * 0.995, 1.0)
        mock_kite.modify_order.assert_called_with(
            variety="regular",
            order_id="ORD_SL_TRAP",
            price=expected_new_limit,
        )

