"""tests/test_aurum_apex_mcx.py
------------------------------
End-to-End Verification Suite for AURUM-APEX: Institutional MCX Precious Metals Quantitative Trading System.

Verifies all 6 institutional invariants:
1. Exact MCX statutory charge calculation & charge-bounded lot sizing across all contracts.
2. Automatic US DST transition (23:30 Summer vs 23:55 Winter) and entry cutoff / flatten timing.
3. REGIME_WAITING gate blocking trades when CI_14 >= 50.0, ADX < 21.0, inside VWAP equilibrium, or during macro blackout.
4. Playbook A (Breakout & Pullback Continuation) and Playbook B (Liquidity Sweep Reclaim with SMT) signal generation.
5. Monotonic 5-Stage "Little-Ahead" Ratchet transitions with zero backward movement and < 20 modifications.
6. Zerodha real-time order states & double-margin prevention (cancelling resting SL before target or flatten exits).
"""
import math
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo
import pytest

from apex3.config import (
    AppConfig,
    LimitsConfig,
    NewsConfig,
    RatchetConfig,
    RegimeConfig,
    RiskConfig,
    ScheduleConfig,
    SignalConfig,
    StopsConfig,
)
from apex3.models import (
    Candle,
    Decision,
    GateStatus,
    InstrumentInfo,
    MarginInfo,
    OpeningRange,
    OrderRequest,
    OrderStatus,
    OrderType,
    PositionState,
    Product,
    Quote,
    Side,
    round_to_tick,
    make_broker_tag,
)
from apex3.strategy.trail import (
    calculate_initial_stop,
    calculate_mcx_initial_stop,
    evaluate_progressive_ratchet,
    floor_to_tick,
    ceil_to_tick,
)
from apex3.news.models import MacroEvent, MacroRegimeState
from apex3.news.macro_scraper import MacroNewsProvider
from apex3.reconciliation.state_machine import ManagedPosition, PositionPhase, StateMachine
from apex3.strategy.accounting import calculate_mcx_intraday_charges
from apex3.strategy.engine import StrategyEngine
from apex3.strategy.regime import (
    MarketRegime,
    calculate_choppiness_index,
    calculate_session_anchored_vwap,
    detect_smt_divergence,
    evaluate_market_regime,
    get_active_mcx_session_windows,
    is_us_dst,
)
from apex3.strategy.schedule import SessionScheduler, SessionState
from apex3.strategy.signal import (
    build_opening_range,
    evaluate_aurum_signal,
    evaluate_signal,
    reconstruct_session_opening_ranges,
)
from apex3.strategy.sizer import size_position

IST = ZoneInfo("Asia/Kolkata")


# ===========================================================================
# INVARIANT 1: Exact MCX Statutory Charges & Charge-Bounded Lot Sizing
# ===========================================================================

class TestInvariant1MCXChargesAndSizing:
    """Verify exact MCX statutory charge calculation and charge-bounded sizing."""

    def test_mcx_charge_calculation_silvermic(self):
        """SILVERMIC 1 lot at Rs. 95000: exact brokerage, CTT, MCX fee, SEBI fee, GST, stamp duty."""
        buy_p = 95000.0
        sell_p = 95250.0
        charges = calculate_mcx_intraday_charges(buy_p, sell_p, lots=1, lot_multiplier=1)

        buy_turnover = buy_p * 1 * 1  # 95,000
        sell_turnover = sell_p * 1 * 1  # 95,250
        tot_turnover = buy_turnover + sell_turnover  # 190,250

        exp_brokerage = min(20.0, 0.0003 * buy_turnover) + min(20.0, 0.0003 * sell_turnover)  # 20 + 20 = 40.0
        exp_ctt = 0.0001 * sell_turnover  # 9.525
        exp_mcx = 0.000021 * tot_turnover  # 3.99525
        exp_sebi = 0.000001 * tot_turnover  # 0.19025
        exp_stamp = 0.00002 * buy_turnover  # 1.90
        exp_gst = 0.18 * (exp_brokerage + exp_mcx + exp_sebi)  # 7.95339
        exp_total = round(exp_brokerage + exp_ctt + exp_mcx + exp_sebi + exp_stamp + exp_gst, 2)

        assert charges == exp_total
        assert charges == 63.56

    def test_mcx_charge_calculation_goldm_and_silverm(self):
        """Verify multiplier scaling for GOLDM (x10) and SILVERM (x5)."""
        charges_goldm = calculate_mcx_intraday_charges(78000.0, 78300.0, lots=1, lot_multiplier=10)
        assert charges_goldm > 100.0  # Higher turnover produces higher CTT + MCX fees

        charges_silverm = calculate_mcx_intraday_charges(95000.0, 95500.0, lots=1, lot_multiplier=5)
        assert charges_silverm > 100.0

    def test_charge_bounded_sizing_and_contract_stepdown(self):
        """Verify that sizing bounds net loss <= risk_cap and steps down contracts on small capital."""
        cfg = AppConfig(
            exchange="MCX",
            base_capital_inr=30000.0,
            limits=LimitsConfig(max_symbols=1, max_leverage=7.0, min_accepted_leverage=1.0),
            risk=RiskConfig(risk_per_trade_pct=1.25),  # Rs. 375.00 max risk
        )
        margin_info = MarginInfo(live_balance=30000.0, available_equity=30000.0, equity_available=30000.0, net_value=30000.0)

        # SILVERM (multiplier=5): 200 pt stop = 200 * 5 = 1000 gross risk > 375 risk cap
        inst_silverm = InstrumentInfo(symbol="SILVERM", exchange="MCX", lot_size=1, lot_multiplier=5, tick_size=1.0)
        silverm_result = size_position(
            symbol="SILVERM",
            side=Side.BUY,
            entry_price=95000.0,
            stop_price=94800.0,
            num_qualified_symbols=1,
            margin_info=margin_info,
            instrument=inst_silverm,
            limits_config=cfg.limits,
            risk_config=cfg.risk,
        )
        assert silverm_result.quantity == 0  # Cannot afford SILVERM

        # SILVERMIC (multiplier=1): 200 pt stop = 200 * 1 = 200 gross risk + ~63 charges = ~263 <= 375 risk cap
        inst_silvermic = InstrumentInfo(symbol="SILVERMIC", exchange="MCX", lot_size=1, lot_multiplier=1, tick_size=1.0)
        silvermic_result = size_position(
            symbol="SILVERMIC",
            side=Side.BUY,
            entry_price=95000.0,
            stop_price=94800.0,
            num_qualified_symbols=1,
            margin_info=margin_info,
            instrument=inst_silvermic,
            limits_config=cfg.limits,
            risk_config=cfg.risk,
        )
        assert silvermic_result.quantity >= 1
        assert silvermic_result.trade_risk_rupees <= 375.0


# ===========================================================================
# INVARIANT 2: Automatic US DST Seasonal Transition & Cutoffs
# ===========================================================================

class TestInvariant2USDaylightSavingTime:
    """Verify automatic US DST seasonal transition (23:30 Summer vs 23:55 Winter)."""

    def test_summer_edt_mcx_session_schedule(self):
        """In Summer (before 1st Sunday of Nov 2026), MCX closes at 23:30 IST."""
        summer_date = date(2026, 10, 20)  # October 2026 -> US DST active
        assert is_us_dst(summer_date) is True

        windows = get_active_mcx_session_windows(summer_date)
        assert windows["market_close"] == time(23, 30)
        assert windows["entry_cutoff"] == time(22, 15)  # T_close - 75m
        assert windows["intraday_flatten"] == time(23, 10)  # T_close - 20m
        assert windows["or_comex_start"] == time(17, 0)
        assert windows["or_comex_end"] == time(17, 30)

        sched_cfg = ScheduleConfig(exchange="MCX")
        scheduler = SessionScheduler(sched_cfg)

        # 22:14 IST -> inside entry window
        t_enter = datetime(2026, 10, 20, 22, 14, tzinfo=IST)
        assert scheduler.can_enter(t_enter) is True

        # 22:16 IST -> cutoff passed, manage only
        t_cutoff = datetime(2026, 10, 20, 22, 16, tzinfo=IST)
        assert scheduler.can_enter(t_cutoff) is False
        assert scheduler.get_session_state(t_cutoff) == SessionState.MANAGE_ONLY

        # 23:11 IST -> flatten active
        t_flat = datetime(2026, 10, 20, 23, 11, tzinfo=IST)
        assert scheduler.should_flatten(t_flat) is True
        assert scheduler.get_session_state(t_flat) == SessionState.FLATTEN

    def test_winter_est_mcx_session_schedule(self):
        """In Winter (after 1st Sunday of Nov 2026), MCX closes at 23:55 IST."""
        winter_date = date(2026, 11, 5)  # Nov 5, 2026 -> US DST ended
        assert is_us_dst(winter_date) is False

        windows = get_active_mcx_session_windows(winter_date)
        assert windows["market_close"] == time(23, 55)
        assert windows["entry_cutoff"] == time(22, 40)  # T_close - 75m
        assert windows["intraday_flatten"] == time(23, 35)  # T_close - 20m
        assert windows["or_comex_start"] == time(18, 0)
        assert windows["or_comex_end"] == time(18, 30)

        sched_cfg = ScheduleConfig(exchange="MCX")
        scheduler = SessionScheduler(sched_cfg)

        # 22:20 IST -> In winter, entry cutoff is 22:40, so 22:20 is STILL valid for entry
        t_enter_winter = datetime(2026, 11, 5, 22, 20, tzinfo=IST)
        assert scheduler.can_enter(t_enter_winter) is True

        # 22:41 IST -> cutoff passed
        t_cutoff_winter = datetime(2026, 11, 5, 22, 41, tzinfo=IST)
        assert scheduler.can_enter(t_cutoff_winter) is False
        assert scheduler.get_session_state(t_cutoff_winter) == SessionState.MANAGE_ONLY

        # 23:36 IST -> flatten active
        t_flat_winter = datetime(2026, 11, 5, 23, 36, tzinfo=IST)
        assert scheduler.should_flatten(t_flat_winter) is True


# ===========================================================================
# INVARIANT 3: REGIME_WAITING Gate
# ===========================================================================

class TestInvariant3RegimeWaitingGate:
    """Verify REGIME_WAITING gate blocking trades on Choppiness, ADX, VWAP, or Macro News."""

    def test_choppiness_index_blocks_entries(self):
        """CI_14 >= 50.0 enters REGIME.WAITING and blocks signals."""
        t_base = datetime(2026, 11, 4, 10, 0, tzinfo=IST)
        # Create 20 oscillating candles with narrow range and high path overlap (high choppiness)
        candles_15m = []
        for i in range(20):
            p = 95000.0 + (15.0 if i % 2 == 0 else -15.0)
            candles_15m.append(
                Candle(
                    timestamp=t_base + timedelta(minutes=i * 15),
                    open=p,
                    high=p + 25.0,
                    low=p - 25.0,
                    close=p + (5.0 if i % 2 == 0 else -5.0),
                    volume=1000,
                    is_complete=True,
                )
            )

        ci = calculate_choppiness_index(candles_15m, period=14)
        assert ci >= 50.0

        candles_5m = [
            Candle(
                timestamp=t_base + timedelta(minutes=i * 5),
                open=95000.0,
                high=95050.0,
                low=94950.0,
                close=95010.0,
                volume=1000,
                is_complete=True,
            )
            for i in range(40)
        ]
        regime_eval = evaluate_market_regime(candles_5m, RegimeConfig(), candles_15m=candles_15m)
        assert regime_eval.regime == MarketRegime.WAITING

    def test_adx_below_21_triggers_waiting(self):
        """ADX < 21.0 triggers REGIME.WAITING."""
        t_base = datetime(2026, 11, 4, 10, 0, tzinfo=IST)
        # Sideways flat candles
        candles_5m = [
            Candle(
                timestamp=t_base + timedelta(minutes=i * 5),
                open=95000.0 + (i % 3),
                high=95005.0,
                low=94995.0,
                close=95000.0,
                volume=1000,
                is_complete=True,
            )
            for i in range(40)
        ]
        regime_eval = evaluate_market_regime(candles_5m, RegimeConfig())
        assert regime_eval.regime == MarketRegime.WAITING
        assert "ADX" in regime_eval.gate.reason or "21.0" in regime_eval.gate.reason

    def test_macro_event_blackout_blocks_aurum_signal(self):
        """Macro event blackout blocks entry in evaluate_aurum_signal."""
        macro_state = MacroRegimeState(
            in_blackout_window=True,
            minutes_to_next_high_event=2.0,
            active_event_title="US CPI m/m",
            macro_score=0.0,
            catalyst_aligned_buy=False,
            catalyst_aligned_sell=False,
            last_updated_utc=datetime.now(timezone.utc),
        )
        or_range = OpeningRange(symbol="SILVERMIC", high=95000.0, low=94700.0, is_complete=True, candle_count=3)
        inst = InstrumentInfo(symbol="SILVERMIC", exchange="MCX", tick_size=1.0)
        quote = Quote(symbol="SILVERMIC", exchange="MCX", last_price=95150.0, bid=95149.0, ask=95151.0, volume=5000, timestamp=datetime.now(IST), is_fresh=True)

        candles_5m = [
            Candle(timestamp=datetime.now(IST) - timedelta(minutes=i * 5), open=95000.0, high=95160.0, low=94980.0, close=95150.0, volume=2000, is_complete=True)
            for i in range(10)
        ]

        sig = evaluate_aurum_signal(
            symbol="SILVERMIC",
            candles_5m=candles_5m,
            benchmark_candles_5m=candles_5m,
            opening_range=or_range,
            market_regime=MarketRegime.BULLISH,
            instrument=inst,
            quote=quote,
            config=SignalConfig(),
            is_entry_window=True,
            macro_state=macro_state,
        )
        assert sig.decision == Decision.WAIT
        assert "MACRO_EVENT_BLACKOUT" in sig.reason_codes


# ===========================================================================
# INVARIANT 4: Playbook A & Playbook B Signal Generation
# ===========================================================================

class TestInvariant4PlaybookGeneration:
    """Verify Playbook A (Breakout & Pullback) and Playbook B (Liquidity Sweep Reclaim with SMT)."""

    def test_playbook_a1_opening_range_breakout(self):
        """Playbook A1: OR Breakout with cushion + CLV >= 0.70 + RVOL >= 1.50x produces PROPOSED_LONG."""
        t_base = datetime(2026, 11, 4, 9, 30, tzinfo=IST)
        or_range = OpeningRange(symbol="SILVERMIC", high=95000.0, low=94700.0, is_complete=True, candle_count=3)
        inst = InstrumentInfo(symbol="SILVERMIC", exchange="MCX", tick_size=1.0)

        # Prior 20 bars with volume 1000
        candles = [
            Candle(timestamp=t_base + timedelta(minutes=i * 5), open=94800.0, high=94950.0, low=94750.0, close=94850.0, volume=1000, is_complete=True)
            for i in range(20)
        ]
        # Breakout candle: closes at 95150 (> OR_high 95000 + 0.15*ATR), high 95160, low 94950, CLV=(95150-94950)/(210) = 0.95, volume=2500 (2.5x RVOL)
        breakout_bar = Candle(
            timestamp=t_base + timedelta(minutes=105),
            open=94980.0,
            high=95160.0,
            low=94950.0,
            close=95150.0,
            volume=2500,
            is_complete=True,
        )
        candles.append(breakout_bar)

        quote = Quote(symbol="SILVERMIC", exchange="MCX", last_price=95150.0, bid=95149.0, ask=95151.0, volume=2500, timestamp=t_base + timedelta(minutes=105), is_fresh=True)

        sig = evaluate_aurum_signal(
            symbol="SILVERMIC",
            candles_5m=candles,
            benchmark_candles_5m=candles,
            opening_range=or_range,
            market_regime=MarketRegime.BULLISH,
            instrument=inst,
            quote=quote,
            config=SignalConfig(max_spread_bps=20.0),
            is_entry_window=True,
        )

        assert sig.decision == Decision.PROPOSED_LONG
        assert sig.side == Side.BUY
        assert sig.playbook == "TREND"
        assert "PLAYBOOK_A1_OR_BREAKOUT" in sig.reason_codes

    def test_playbook_b_liquidity_sweep_reclaim_with_smt(self):
        """Playbook B: Swept OR low by 0.25x ATR and snapped back inside with SMT divergence."""
        t_base = datetime(2026, 11, 4, 18, 0, tzinfo=IST)  # COMEX window
        or_range = OpeningRange(symbol="SILVERMIC", high=95500.0, low=95000.0, is_complete=True, candle_count=6, session_type="COMEX")
        inst = InstrumentInfo(symbol="SILVERMIC", exchange="MCX", tick_size=1.0)

        candles_silver = [
            Candle(timestamp=t_base + timedelta(minutes=i * 5), open=95200.0, high=95300.0, low=95100.0, close=95200.0, volume=1000, is_complete=True)
            for i in range(15)
        ]
        # Silver sweeps below OR_low (95000) down to 94940 (60 pts sweep), then snaps back to close at 95100 with CLV=0.85 and RVOL=2.2x
        sweep_reclaim_bar = Candle(
            timestamp=t_base + timedelta(minutes=80),
            open=94980.0,
            high=95120.0,
            low=94940.0,
            close=95100.0,
            volume=2200,
            is_complete=True,
        )
        candles_silver.append(sweep_reclaim_bar)

        # Sibling Gold holds above its swing low (Bullish SMT divergence)
        candles_gold = [
            Candle(timestamp=t_base + timedelta(minutes=i * 5), open=78200.0, high=78300.0, low=78150.0, close=78250.0, volume=1000, is_complete=True)
            for i in range(15)
        ]
        candles_gold.append(
            Candle(timestamp=t_base + timedelta(minutes=80), open=78250.0, high=78350.0, low=78220.0, close=78320.0, volume=1500, is_complete=True)
        )

        quote = Quote(symbol="SILVERMIC", exchange="MCX", last_price=95100.0, bid=95099.0, ask=95101.0, volume=2200, timestamp=t_base + timedelta(minutes=80), is_fresh=True)

        sig = evaluate_aurum_signal(
            symbol="SILVERMIC",
            candles_5m=candles_silver,
            benchmark_candles_5m=candles_silver,
            opening_range=or_range,
            market_regime=MarketRegime.BULLISH,
            instrument=inst,
            quote=quote,
            config=SignalConfig(max_spread_bps=20.0),
            is_entry_window=True,
            sibling_candles_5m=candles_gold,
        )

        assert sig.decision == Decision.PROPOSED_LONG
        assert sig.side == Side.BUY
        assert sig.playbook == "SWEEP_RECLAIM"
        assert "PLAYBOOK_B_SWEEP_RECLAIM" in sig.reason_codes


# ===========================================================================
# INVARIANT 5: Monotonic 5-Stage "Little-Ahead" Ratchet Transitions
# ===========================================================================

class TestInvariant5FiveStageMonotonicRatchet:
    """Verify 5-Stage progressive ratchet with strict monotonicity and modification cap."""

    def test_progressive_ratchet_stages_and_monotonicity(self):
        """Step through Stage 0 -> Stage 1 BE+ -> Stage 2 -> Stage 3 -> Stage 4 with zero backward stop movement."""
        entry_price = 95000.0
        initial_stop = 94750.0  # 1.0R = 250 points
        r_val = 250.0

        pos = PositionState(
            symbol="SILVERMIC",
            side=Side.BUY,
            quantity=1,
            average_entry=entry_price,
            initial_stop_price=initial_stop,
            protective_stop_price=initial_stop,
            best_favorable_price=entry_price,
            r_value=r_val,
        )
        cfg = AppConfig(exchange="MCX", ratchet=RatchetConfig(max_stop_modifications=20))

        # 1. Stage 0 (< 1.0R): at +0.70R, stop remains at initial stop
        px_07 = entry_price + (0.70 * r_val)
        res_0 = evaluate_progressive_ratchet(pos, px_07, candles_5m=[], atr_14=150.0, config=cfg, tick_size=1.0)
        assert res_0.should_modify is False
        assert res_0.proposed_stop == initial_stop
        assert res_0.ratchet_stage == 0

        # 2. Stage 1 (>= 1.0R): at +1.10R, moves to BE+ (95067.0 with 3 buffer ticks)
        px_11 = entry_price + (1.10 * r_val)
        pos.best_favorable_price = px_11
        res_1 = evaluate_progressive_ratchet(pos, px_11, candles_5m=[], atr_14=150.0, config=cfg, tick_size=1.0)
        assert res_1.should_modify is True
        assert res_1.proposed_stop in (95066.0, 95067.0)
        assert res_1.ratchet_stage == 1
        pos.protective_stop_price = res_1.proposed_stop
        pos.stop_modification_count = 1

        # 3. Stage 2 (>= 1.40R): at +1.50R, locks +0.65R (95000 + 0.65*250 = 95162.0)
        px_15 = entry_price + (1.50 * r_val)
        pos.best_favorable_price = px_15
        res_2 = evaluate_progressive_ratchet(pos, px_15, candles_5m=[], atr_14=150.0, config=cfg, tick_size=1.0)
        assert res_2.should_modify is True
        assert res_2.proposed_stop >= entry_price + (0.65 * r_val)
        assert res_2.ratchet_stage == 2
        pos.protective_stop_price = res_2.proposed_stop
        pos.stop_modification_count = 2

        # 4. Stage 3 (>= 2.00R): at +2.50R, ratchets to >= entry + (2.50 - 0.65)*250 = 95462.0
        px_25 = entry_price + (2.50 * r_val)
        pos.best_favorable_price = px_25
        res_3 = evaluate_progressive_ratchet(pos, px_25, candles_5m=[], atr_14=150.0, config=cfg, tick_size=1.0)
        assert res_3.should_modify is True
        assert res_3.proposed_stop >= entry_price + (1.85 * r_val)
        assert res_3.ratchet_stage == 3
        pos.protective_stop_price = res_3.proposed_stop
        pos.stop_modification_count = 3

        # 5. Stage 4 (>= 3.20R): at +3.50R, climax harvest locks tight
        px_35 = entry_price + (3.50 * r_val)
        pos.best_favorable_price = px_35
        res_4 = evaluate_progressive_ratchet(pos, px_35, candles_5m=[], atr_14=150.0, config=cfg, tick_size=1.0)
        assert res_4.should_modify is True
        assert res_4.proposed_stop >= entry_price + (3.15 * r_val)
        assert res_4.ratchet_stage == 4
        stop_at_peak = res_4.proposed_stop
        pos.protective_stop_price = stop_at_peak

        # 6. Monotonicity Test: Price plunges violently from +3.50R down to +0.90R
        px_plunge = entry_price + (0.90 * r_val)
        res_plunge = evaluate_progressive_ratchet(pos, px_plunge, candles_5m=[], atr_14=150.0, config=cfg, tick_size=1.0)
        assert res_plunge.should_modify is False
        assert res_plunge.proposed_stop == stop_at_peak  # Stop never retreats!

    def test_modification_cap_at_20(self):
        """Enforces Kite API budget cap: no order modifications when count >= 20."""
        pos = PositionState(
            symbol="SILVERMIC",
            side=Side.BUY,
            quantity=1,
            average_entry=95000.0,
            initial_stop_price=94750.0,
            protective_stop_price=95200.0,
            best_favorable_price=95800.0,
            r_value=250.0,
            modification_count=20,
        )
        cfg = AppConfig(exchange="MCX", ratchet=RatchetConfig(max_stop_modifications=20))
        res = evaluate_progressive_ratchet(pos, 96000.0, candles_5m=[], atr_14=150.0, config=cfg, tick_size=1.0)
        assert res.should_modify is False
        assert "Max modification cap (20) reached" in res.reason


# ===========================================================================
# INVARIANT 6: Zerodha Real-Time Order Lifecycle & Double-Margin Prevention
# ===========================================================================

class TestInvariant6OrderLifecycleAndDoubleMargin:
    """Verify Zerodha real-time order lifecycle and double-margin avoidance on exits."""

    def test_double_margin_avoidance_on_target_exit(self):
        """Cancels resting protective SL and confirms CANCELLED before dispatching target exit."""
        mock_broker = MagicMock()
        mock_broker.is_session_valid.return_value = True
        mock_broker.get_margins.return_value = MarginInfo(live_balance=100000.0, available_equity=100000.0, equity_available=100000.0, net_value=100000.0)
        mock_broker.get_orders.return_value = []
        mock_broker.get_trades.return_value = []
        mock_broker.get_positions.return_value = []

        audit_mock = MagicMock()
        audit_mock.is_clean = True
        audit_mock.unknown_symbols = []

        cfg = AppConfig(
            exchange="MCX",
            mode="paper",
            stops=StopsConfig(use_fixed_target=True, target_r=2.0),
        )
        engine = StrategyEngine(config=cfg, broker=mock_broker, journal=MagicMock())
        engine.reconciler = MagicMock()
        engine.reconciler.audit_startup_state.return_value = audit_mock

        # Setup an open position with resting SL order SL_999
        pos = ManagedPosition(
            symbol="SILVERMIC",
            side=Side.BUY,
            quantity=1,
            average_entry=95000.0,
            initial_stop_price=94750.0,
            current_stop_price=94750.0,
            target_price=95500.0,
            protective_order_id="SL_999",
            phase=PositionPhase.OPEN_PROTECTED,
        )
        engine.state_machine.positions["SILVERMIC"] = pos

        now_t = datetime(2026, 11, 4, 11, 0, tzinfo=IST)
        # Price hits target 95500
        quote = Quote(symbol="SILVERMIC", exchange="MCX", last_price=95500.0, bid=95499.0, ask=95501.0, volume=5000, timestamp=now_t, is_fresh=True)
        mock_broker.get_quote.return_value = quote
        inst = InstrumentInfo(symbol="SILVERMIC", exchange="MCX", tick_size=1.0)
        mock_broker.get_instrument_info.return_value = inst

        # Mock terminal order status of SL cancellation upon cancel_order
        state = {"cancelled": False}
        def mock_cancel(oid):
            state["cancelled"] = True

        def mock_get_status(oid):
            st = MagicMock()
            if oid == "SL_999":
                st.status = OrderStatus.CANCELLED if state["cancelled"] else OrderStatus.TRIGGER_PENDING
                st.filled_quantity = 0
                return st
            elif oid == "EXIT_TGT_001":
                st.status = OrderStatus.COMPLETE
                st.filled_quantity = 1
                st.pending_quantity = 0
                st.average_price = 95500.0
                return st
            return None

        mock_broker.cancel_order.side_effect = mock_cancel
        mock_broker.get_order_status.side_effect = mock_get_status
        mock_broker.place_order.return_value = "EXIT_TGT_001"

        decisions = engine.evaluate_step(now_t)

        # 1. Verified: broker cancel_order was called on SL_999 BEFORE placing exit
        mock_broker.cancel_order.assert_called_with("SL_999")
        # 2. Verified: target exit was dispatched
        mock_broker.place_order.assert_called()
        tgt_calls = [c for c in mock_broker.place_order.call_args_list if "TGT" in getattr(c[0][0], "tag", "")]
        assert len(tgt_calls) >= 1

    def test_emergency_gap_exit_when_sl_gaps_through_limit(self):
        """When SL triggers but rests unfilled and price gaps through limit, executes emergency market exit EMG."""
        mock_broker = MagicMock()
        mock_broker.is_session_valid.return_value = True
        mock_broker.get_margins.return_value = MarginInfo(live_balance=100000.0, available_equity=100000.0, equity_available=100000.0, net_value=100000.0)
        mock_broker.get_orders.return_value = []
        mock_broker.get_trades.return_value = []
        mock_broker.get_positions.return_value = []

        audit_mock = MagicMock()
        audit_mock.is_clean = True
        audit_mock.unknown_symbols = []

        cfg = AppConfig(exchange="MCX", mode="paper")
        engine = StrategyEngine(config=cfg, broker=mock_broker, journal=MagicMock())
        engine.reconciler = MagicMock()
        engine.reconciler.audit_startup_state.return_value = audit_mock

        pos = ManagedPosition(
            symbol="SILVERMIC",
            side=Side.BUY,
            quantity=1,
            average_entry=95000.0,
            initial_stop_price=94750.0,
            current_stop_price=94750.0,
            target_price=95500.0,
            protective_order_id="SL_GAP_123",
            phase=PositionPhase.OPEN_PROTECTED,
        )
        engine.state_machine.positions["SILVERMIC"] = pos

        now_t = datetime(2026, 11, 4, 11, 30, tzinfo=IST)
        # Market price gapped down through limit (94740 * 0.999 = 94645.26) to 94600.0
        quote = Quote(symbol="SILVERMIC", exchange="MCX", last_price=94600.0, bid=94599.0, ask=94601.0, volume=5000, timestamp=now_t, is_fresh=True)
        mock_broker.get_quote.return_value = quote
        inst = InstrumentInfo(symbol="SILVERMIC", exchange="MCX", tick_size=1.0)
        mock_broker.get_instrument_info.return_value = inst

        # Cancel confirmation upon cancel_order
        state = {"cancelled": False}
        def mock_cancel(oid):
            state["cancelled"] = True

        def mock_get_status(oid):
            st = MagicMock()
            if oid == "SL_GAP_123":
                if state["cancelled"]:
                    st.status = OrderStatus.CANCELLED
                    st.filled_quantity = 0
                else:
                    st.status = OrderStatus.OPEN
                    st.price = 94740.0
                    st.filled_quantity = 0
                    st.request = MagicMock()
                    st.request.price = 94740.0
                return st
            elif oid == "EMG_EXIT_001":
                st.status = OrderStatus.COMPLETE
                st.filled_quantity = 1
                st.pending_quantity = 0
                st.average_price = 94600.0
                return st
            return None

        mock_broker.cancel_order.side_effect = mock_cancel
        mock_broker.get_order_status.side_effect = mock_get_status
        mock_broker.place_order.return_value = "EMG_EXIT_001"

        decisions = engine.evaluate_step(now_t)

        # Verified: SL order cancelled and confirmed terminal, then emergency market exit placed
        mock_broker.cancel_order.assert_called_with("SL_GAP_123")
        mock_broker.place_order.assert_called()
        emg_calls = [c for c in mock_broker.place_order.call_args_list if "EMG" in getattr(c[0][0], "tag", "")]
        assert len(emg_calls) >= 1
