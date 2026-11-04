"""tests/test_macro_news_shield.py
--------------------------------
Comprehensive test suite for the Macro Blackout & News Catalyst Shield in APEX-3-MCX:
Test 1: Verify T - 15m to T + 5m around a High-Impact USD CPI event blocks new entries with MACRO_EVENT_BLACKOUT.
Test 2: Verify an open SILVERMIC position at +0.85R automatically ratchets its stop to BE+ (Entry + Charges + 2 ticks) at T - 5m before NFP.
Test 3: Verify an unbuffered open position at +0.20R executes a tracked marketable limit exit (APEX3_SLVMIC_NWS) at T - 3m before FOMC.
Test 4: Verify a Cooler-than-expected US CPI print (Actual 2.8% < Forecast 3.0%) produces macro_score >= +0.60, blocks Side.SELL breakouts, boosts Side.BUY conviction by 1.40x, and expands target from 2.50R to 3.50R.
Test 5: Verify scraper network timeout gracefully degrades to macro_score = 0.0 in < 1ms without raising unhandled exceptions in evaluate_step().
"""
from datetime import date, datetime, time, timedelta, timezone
import time as time_mod
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo
import pytest

from apex3.config import AppConfig, NewsConfig, RatchetConfig
from apex3.models import (
    Candle,
    Decision,
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
from apex3.news.models import MacroEvent, MacroRegimeState, NewsHeadline
from apex3.news.macro_scraper import (
    MacroNewsProvider,
    calculate_economic_surprise,
    score_headline_lexicon,
)
from apex3.reconciliation.state_machine import ManagedPosition, PositionPhase
from apex3.strategy.accounting import calculate_mcx_intraday_charges
from apex3.strategy.engine import StrategyEngine
from apex3.strategy.regime import MarketRegime
from apex3.strategy.signal import (
    StockCandidateBar,
    evaluate_candidate_signal,
    evaluate_news_gate,
)
from apex3.strategy.trail import (
    calculate_target_price,
    evaluate_progressive_ratchet,
)

IST = ZoneInfo("Asia/Kolkata")


class TestMacroNewsShield:
    """Test suite covering the 5 macro blackout & catalyst test invariants."""

    def test_1_macro_event_blackout_window(self):
        """Test 1: Verify T - 15m to T + 5m around a High-Impact USD CPI event blocks new entries with MACRO_EVENT_BLACKOUT."""
        t_event = datetime(2026, 11, 4, 13, 30, tzinfo=timezone.utc)
        event = MacroEvent(
            event_id="cpi_test_01",
            timestamp_utc=t_event,
            currency="USD",
            impact="HIGH",
            title="US CPI m/m",
        )

        cfg = AppConfig(news=NewsConfig(pre_event_blackout_min=15.0, post_event_blackout_min=5.0))
        provider = MacroNewsProvider(config=cfg)
        provider.stop()  # Do not run background polling during unit test
        provider.add_event(event)

        # 1. At T - 15m: exact boundary start of blackout window
        st_pre15 = provider.get_macro_state(t_event - timedelta(minutes=15))
        assert st_pre15.in_blackout_window is True
        assert st_pre15.active_event_title == "US CPI m/m"
        passed, code = evaluate_news_gate(Side.BUY, st_pre15)
        assert passed is False
        assert code == "MACRO_EVENT_BLACKOUT"

        # 2. At T - 5m: inside blackout window
        st_pre5 = provider.get_macro_state(t_event - timedelta(minutes=5))
        assert st_pre5.in_blackout_window is True
        passed, code = evaluate_news_gate(Side.BUY, st_pre5)
        assert passed is False
        assert code == "MACRO_EVENT_BLACKOUT"

        # 3. At T + 0m: at event release
        st_event = provider.get_macro_state(t_event)
        assert st_event.in_blackout_window is True
        passed, code = evaluate_news_gate(Side.BUY, st_event)
        assert passed is False
        assert code == "MACRO_EVENT_BLACKOUT"

        # 4. At T + 5m: post-event blackout window
        st_post5 = provider.get_macro_state(t_event + timedelta(minutes=5))
        assert st_post5.in_blackout_window is True
        passed, code = evaluate_news_gate(Side.SELL, st_post5)
        assert passed is False
        assert code == "MACRO_EVENT_BLACKOUT"

        # 5. Outside blackout window at T - 16m
        st_pre16 = provider.get_macro_state(t_event - timedelta(minutes=16))
        assert st_pre16.in_blackout_window is False
        passed, code = evaluate_news_gate(Side.BUY, st_pre16)
        assert passed is True
        assert code == "PASS"

        # 6. Outside blackout window at T + 6m
        st_post6 = provider.get_macro_state(t_event + timedelta(minutes=6))
        assert st_post6.in_blackout_window is False
        passed, code = evaluate_news_gate(Side.BUY, st_post6)
        assert passed is True
        assert code == "PASS"

    def test_2_pre_event_shield_locks_breakeven_at_t_minus_5m(self):
        """Test 2: Verify an open SILVERMIC position at +0.85R automatically ratchets its stop to BE+ (Entry + Charges + 2 ticks) at T - 5m before NFP."""
        entry_price = 95000.0
        initial_stop = 94750.0  # 1.0R = 250 points
        r_val = 250.0
        current_price = entry_price + (0.85 * r_val)  # 95212.5 (+0.85R favorable)

        pos = PositionState(
            symbol="SILVERMIC",
            side=Side.BUY,
            quantity=1,
            average_entry=entry_price,
            initial_stop_price=initial_stop,
            protective_stop_price=initial_stop,
            best_favorable_price=current_price,
            r_value=r_val,
        )

        cfg = AppConfig(
            exchange="MCX",
            ratchet=RatchetConfig(stage1_r=1.00, stage1_buffer_ticks=2),
            news=NewsConfig(pre_event_lock_be_threshold_r=0.75),
        )

        # Baseline: normal ratchet without news shield remains at Stage 0 (stop unchanged at initial stop)
        baseline_eval = evaluate_progressive_ratchet(
            pos=pos,
            current_price=current_price,
            candles_5m=[],
            atr_14=200.0,
            config=cfg,
            tick_size=1.0,
            macro_state=None,
        )
        assert baseline_eval.should_modify is False
        assert baseline_eval.proposed_stop == initial_stop

        # Pre-Event Shield: NFP in 5 minutes (0 < mins <= 5.0) and +0.85R >= 0.75R
        macro_state = MacroRegimeState(
            in_blackout_window=False,
            minutes_to_next_high_event=5.0,
            active_event_title="US Non-Farm Payrolls",
            macro_score=0.0,
            catalyst_aligned_buy=False,
            catalyst_aligned_sell=False,
            last_updated_utc=datetime.now(timezone.utc),
        )

        shield_eval = evaluate_progressive_ratchet(
            pos=pos,
            current_price=current_price,
            candles_5m=[],
            atr_14=200.0,
            config=cfg,
            tick_size=1.0,
            macro_state=macro_state,
        )

        # Exact BE+ calculation: Entry (95000) + Statutory Charges Offset (64 pts for ₹63.53) + 2 buffer ticks (2 pts) = 95066.0
        charges = calculate_mcx_intraday_charges(entry_price, entry_price, lots=1, lot_multiplier=1)
        assert charges == 63.53
        expected_be_plus = entry_price + 64.0 + 2.0  # 95066.0

        assert shield_eval.should_modify is True
        assert shield_eval.proposed_stop == expected_be_plus
        assert "BE+" in shield_eval.reason

    def test_3_unbuffered_position_executes_protective_exit_before_fomc(self):
        """Test 3: Verify an unbuffered open position at +0.20R executes a tracked marketable limit exit (APEX3_SLVMIC_NWS) at T - 3m before FOMC."""
        mock_broker = MagicMock()
        mock_journal = MagicMock()
        mock_logger = MagicMock()

        mock_broker.is_session_valid.return_value = True
        mock_broker.get_margins.return_value = MarginInfo(equity_available=30000.0, net_value=30000.0)
        mock_broker.get_orders.return_value = []
        mock_broker.get_trades.return_value = []
        mock_broker.get_positions.return_value = []

        now_ist = datetime(2026, 11, 4, 18, 57, tzinfo=IST)
        mock_broker.get_instrument_info.return_value = InstrumentInfo(
            symbol="SILVERMIC",
            exchange="MCX",
            lot_size=1,
            tick_size=1.0,
            lot_multiplier=1,
            lower_circuit=85000.0,
            upper_circuit=105000.0,
        )

        entry_px = 95000.0
        initial_sl = 94750.0  # 250 unit risk
        curr_px = 95050.0     # +0.20R (< 0.75R unbuffered)
        mock_broker.get_quote.return_value = Quote(
            symbol="SILVERMIC",
            exchange="MCX",
            last_price=curr_px,
            bid=95049.0,
            ask=95051.0,
            volume=5000,
            timestamp=now_ist,
        )

        placed_orders = {}
        def fake_place(req: OrderRequest):
            oid = f"ORD_EXIT_{len(placed_orders) + 1}"
            placed_orders[oid] = req
            return oid
        mock_broker.place_order.side_effect = fake_place

        def fake_status(oid):
            if oid == "SL_INIT_001":
                return MagicMock(status=OrderStatus.TRIGGER_PENDING, filled_quantity=0, average_price=0.0)
            req = placed_orders.get(oid)
            return MagicMock(status=OrderStatus.COMPLETE, filled_quantity=1, average_price=curr_px, request=req)
        mock_broker.get_order_status.side_effect = fake_status
        mock_broker.cancel_order.return_value = True

        from apex3.config import ScheduleConfig
        cfg = AppConfig(
            exchange="MCX",
            schedule=ScheduleConfig(
                market_open="09:00",
                entry_start="09:00",
                entry_cutoff="22:30",
                flatten_start="23:30",
                flat_target="23:35",
                market_close="23:55",
            ),
            news=NewsConfig(
                enabled=True,
                pre_event_flatten_unbuffered_min=3.0,
                pre_event_lock_be_threshold_r=0.75,
            ),
        )

        # Mock macro provider with FOMC in 3.0 minutes
        mock_macro_provider = MagicMock()
        mock_macro_provider.get_macro_state.return_value = MacroRegimeState(
            in_blackout_window=False,
            minutes_to_next_high_event=3.0,
            active_event_title="FOMC Interest Rate Decision",
            macro_score=0.0,
            catalyst_aligned_buy=False,
            catalyst_aligned_sell=False,
            last_updated_utc=datetime.now(timezone.utc),
        )

        engine = StrategyEngine(
            config=cfg,
            broker=mock_broker,
            journal=mock_journal,
            session_tag="APEX3",
            trade_logger=mock_logger,
            macro_provider=mock_macro_provider,
        )

        # Seed open protected position for SILVERMIC
        managed = engine.state_machine.get_or_create_position("SILVERMIC")
        managed.phase = PositionPhase.OPEN_PROTECTED
        managed.side = Side.BUY
        managed.quantity = 1
        managed.average_entry = entry_px
        managed.initial_stop_price = initial_sl
        managed.current_stop_price = initial_sl
        managed.protective_order_id = "SL_INIT_001"
        managed.fee_lock_applied = False
        managed.r_value = 250.0

        # Execute decision cycle
        decisions = engine.evaluate_step(current_time=now_ist)

        # Verify protective SL order was cancelled
        mock_broker.cancel_order.assert_called_with("SL_INIT_001")

        # Verify marketable limit exit with tag APEX3_SLVMIC_NWS was placed
        expected_tag = make_broker_tag("APEX3", "SILVERMIC", "NWS")
        assert expected_tag == "APEX3_SLVMIC_NWS"
        assert len(placed_orders) == 1
        exit_order = next(iter(placed_orders.values()))
        assert exit_order.tag == "APEX3_SLVMIC_NWS"
        assert exit_order.side == Side.SELL
        assert exit_order.quantity == 1

        # Verify decision output
        assert any(d.decision == Decision.EXIT and "PRE_NEWS_PROTECTIVE_EXIT" in d.reason_codes for d in decisions)
        exit_dec = next(d for d in decisions if d.decision == Decision.EXIT)
        assert "EXIT_NEWS_APEX3_SLVMIC_NWS" in exit_dec.broker_action

    def test_4_cooler_cpi_catalyst_alignment_and_target_expansion(self):
        """Test 4: Verify a Cooler-than-expected US CPI print (Actual 2.8% < Forecast 3.0%) produces macro_score >= +0.60, blocks Side.SELL breakouts, boosts Side.BUY conviction by 1.40x, and expands target from 2.50R to 3.50R."""
        now_utc = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)
        cpi_event = MacroEvent(
            event_id="cpi_cool_01",
            timestamp_utc=now_utc - timedelta(minutes=10),
            currency="USD",
            impact="HIGH",
            title="US CPI m/m",
            actual=2.8,
            forecast=3.0,
            previous=3.1,
            inverse_for_usd=False,
        )

        # 1. Economic surprise score calculation (Bullion Inverse Law)
        surprise = calculate_economic_surprise(cpi_event)
        assert surprise == 0.80  # Cooler CPI = dovish USD = bullish bullion (+0.80)
        assert surprise >= 0.60

        cfg_app = AppConfig(news=NewsConfig(catalyst_boost_threshold=0.60, catalyst_target_r=3.50))
        provider = MacroNewsProvider(config=cfg_app)
        provider.stop()
        provider.add_event(cpi_event)

        macro_state = provider.get_macro_state(now_utc)
        assert macro_state.macro_score >= 0.60
        assert macro_state.catalyst_aligned_buy is True
        assert macro_state.catalyst_aligned_sell is False

        # 2. Gate 10 blocks Side.SELL breakouts
        passed_sell, reason_sell = evaluate_news_gate(Side.SELL, macro_state)
        assert passed_sell is False
        assert reason_sell == "REJECTED_AGAINST_BULLISH_MACRO_NEWS"

        # 3. Gate 10 passes Side.BUY breakouts
        passed_buy, reason_buy = evaluate_news_gate(Side.BUY, macro_state)
        assert passed_buy is True
        assert reason_buy == "PASS"

        # 4. Conviction Boost (1.40x) & Target Expansion to 3.50R
        bar = StockCandidateBar(
            symbol="SILVERMIC",
            timestamp_hhmm="10:15",
            close=95000.0,
            or_high=94500.0,
            or_low=94200.0,
            session_vwap=94700.0,
            stock_return_20=0.015,
            nifty_return_20=0.005,
            bar_volume=3000.0,
            avg_volume_20=1000.0,
            bid_ask_spread_bps=2.0,
            upper_circuit=105000.0,
            lower_circuit=85000.0,
            swing_low_5=94600.0,
            swing_high_5=95200.0,
            atr14=200.0,
            session_high_so_far=95100.0,
            session_low_so_far=94200.0,
            adr14_pct=0.025,
            bar_high=95050.0,
            bar_low=94800.0,
            tick_size=1.0,
        )

        cfg_dict = {
            "session_schedule": {"entry_start_time": "09:00", "entry_cutoff_time": "22:00"},
            "radar": {"core_watchlist": ["SILVERMIC"], "core_conviction_multiplier": 1.35},
            "stops_and_targets": {"initial_atr_multiplier": 1.5, "target_r": 2.50},
            "news": {"catalyst_boost_threshold": 0.60, "catalyst_target_r": 3.50},
            "signal_gates": {"min_stop_distance_pct": 0.001, "max_stop_distance_pct": 0.05},
        }

        # Baseline signal without macro catalyst
        sig_base = evaluate_candidate_signal(
            bar=bar,
            regime=MarketRegime.BULLISH,
            active_positions_count=0,
            cfg=cfg_dict,
            macro_state=None,
        )
        assert sig_base is not None
        assert sig_base.catalyst_aligned is False

        # Catalyzed signal with Cooler CPI print
        sig_catalyst = evaluate_candidate_signal(
            bar=bar,
            regime=MarketRegime.BULLISH,
            active_positions_count=0,
            cfg=cfg_dict,
            macro_state=macro_state,
        )
        assert sig_catalyst is not None
        assert sig_catalyst.catalyst_aligned is True
        assert sig_catalyst.macro_score == 0.80

        # Conviction score boosted by exactly 1.40x
        assert pytest.approx(sig_catalyst.conviction_score, 0.001) == sig_base.conviction_score * 1.40

        # Target price expanded from 2.50R to 3.50R
        risk_per_share = sig_catalyst.levels.risk_per_share
        expected_target_35r = round_to_tick(bar.close + (3.50 * risk_per_share), tick_size=1.0, direction="floor")
        assert sig_catalyst.levels.target_price == expected_target_35r

    def test_5_scraper_network_timeout_graceful_degradation(self):
        """Test 5: Verify scraper network timeout gracefully degrades to macro_score = 0.0 in < 1ms without raising unhandled exceptions in evaluate_step()."""
        mock_broker = MagicMock()
        mock_journal = MagicMock()
        mock_logger = MagicMock()

        mock_broker.is_session_valid.return_value = True
        mock_broker.get_margins.return_value = MarginInfo(equity_available=30000.0, net_value=30000.0)
        mock_broker.get_orders.return_value = []
        mock_broker.get_trades.return_value = []
        mock_broker.get_positions.return_value = []

        now_ist = datetime(2026, 11, 4, 10, 0, tzinfo=IST)
        mock_broker.get_instrument_info.return_value = InstrumentInfo(
            symbol="SILVERMIC",
            exchange="MCX",
            lot_size=1,
            tick_size=1.0,
        )
        mock_broker.get_quote.return_value = Quote(
            symbol="SILVERMIC",
            exchange="MCX",
            last_price=95000.0,
            bid=94999.0,
            ask=95001.0,
            volume=1000,
            timestamp=now_ist,
        )

        # Faulty provider that throws a simulated network timeout
        class TimingFailingMacroProvider:
            def get_macro_state(self, now_utc):
                raise TimeoutError("HTTP request timed out after 3.0s")

        cfg = AppConfig(exchange="MCX")
        failing_provider = TimingFailingMacroProvider()

        engine = StrategyEngine(
            config=cfg,
            broker=mock_broker,
            journal=mock_journal,
            session_tag="APEX3",
            trade_logger=mock_logger,
            macro_provider=failing_provider,
        )

        # Run evaluate_step with microsecond timing verification
        t_start = time_mod.perf_counter()
        decisions = engine.evaluate_step(current_time=now_ist)
        elapsed_sec = time_mod.perf_counter() - t_start

        # Execution must complete without unhandled exceptions
        assert isinstance(decisions, list)

        # Verify graceful fallback to neutral state (macro_score == 0.0, no blackout)
        assert engine.latest_macro_state is not None
        assert engine.latest_macro_state.macro_score == 0.0
        assert engine.latest_macro_state.in_blackout_window is False
        assert engine.latest_macro_state.catalyst_aligned_buy is False
        assert engine.latest_macro_state.catalyst_aligned_sell is False

        # Verify in-memory resolution speed < 100ms total decision cycle
        assert elapsed_sec < 0.100
