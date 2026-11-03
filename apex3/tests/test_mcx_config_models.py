"""
apex3/tests/test_mcx_config_models.py
-------------------------------------
Unit tests verifying Step 1 and Step 2 of the APEX-3 to APEX-3-MCX upgrade:
- InstrumentInfo MCX fields (exchange, order_lot_size, lot_multiplier, expiry_date, days_to_expiry)
- round_to_tick Decimal helper (floor, ceil, nearest)
- OpeningRange session_type support (MORNING, COMEX)
- config.yaml and config.py MCX Precious Metals production parameters
"""

from datetime import date
import pytest
from apex3.models import InstrumentInfo, OpeningRange, round_to_tick
from apex3.config import load_config


class TestStep1Models:
    def test_instrument_info_mcx_fields(self):
        inst = InstrumentInfo(
            symbol="SILVERMIC",
            exchange="MCX",
            tick_size=1.0,
            lot_size=1,
            order_lot_size=1,
            lot_multiplier=1,
            expiry_date=date(2026, 6, 30),
            days_to_expiry=15,
        )
        assert inst.exchange == "MCX"
        assert inst.order_lot_size == 1
        assert inst.lot_multiplier == 1
        assert inst.expiry_date == date(2026, 6, 30)
        assert inst.days_to_expiry == 15

    def test_round_to_tick_decimal(self):
        # Tick size 1.0 (MCX)
        assert round_to_tick(80000.4, 1.0, "nearest") == 80000.0
        assert round_to_tick(80000.5, 1.0, "nearest") == 80001.0
        assert round_to_tick(80000.6, 1.0, "nearest") == 80001.0
        assert round_to_tick(80000.9, 1.0, "floor") == 80000.0
        assert round_to_tick(80000.1, 1.0, "ceil") == 80001.0

        # Tick size 0.05 (NSE equity compatibility)
        assert round_to_tick(123.456, 0.05, "nearest") == 123.45
        assert round_to_tick(123.41, 0.05, "floor") == 123.40
        assert round_to_tick(123.41, 0.05, "ceil") == 123.45

    def test_opening_range_session_type(self):
        or_morning = OpeningRange(symbol="SILVERMIC", high=85000.0, low=84000.0, session_type="MORNING")
        assert or_morning.session_type == "MORNING"

        or_comex = OpeningRange(symbol="GOLDM", high=80500.0, low=80000.0, session_type="COMEX")
        assert or_comex.session_type == "COMEX"


class TestStep2Config:
    def test_mcx_config_yaml_parameters(self):
        cfg = load_config()

        # Top-level parameters
        assert cfg.exchange == "MCX"
        assert cfg.margin_segment == "commodity"
        assert cfg.fetch_live_zerodha_margin is True
        assert cfg.base_capital_inr == 30000.0
        assert cfg.safety_reserve_buffer_pct == 0.05
        assert cfg.max_concurrent_positions == 2
        assert cfg.max_new_entries_per_bar == 1
        assert cfg.per_trade_risk_pct == 0.0125
        assert cfg.daily_max_loss_pct == 0.0250
        assert cfg.bound_loss_with_charges is True
        assert cfg.max_fee_to_profit_ratio == 0.18

        # Universe
        core = {i.symbol_root: i for i in cfg.universe.core_watchlist}
        assert "SILVERMIC" in core
        assert core["SILVERMIC"].lot_multiplier == 1
        assert core["SILVERMIC"].tick_size == 1.0
        assert core["SILVERMIC"].max_spread_bps == 3.5
        assert core["SILVERMIC"].min_atr_pct == 0.0014

        assert "GOLDGUINEA" in core
        assert core["GOLDGUINEA"].lot_multiplier == 1
        assert core["GOLDGUINEA"].tick_size == 1.0
        assert core["GOLDGUINEA"].max_spread_bps == 5.0
        assert core["GOLDGUINEA"].min_atr_pct == 0.0010

        scale = {i.symbol_root: i for i in cfg.universe.scale_up_watchlist}
        assert "SILVERM" in scale
        assert scale["SILVERM"].lot_multiplier == 5
        assert "GOLDM" in scale
        assert scale["GOLDM"].lot_multiplier == 10

        # Sessions
        assert cfg.sessions.morning_or_start == "09:00"
        assert cfg.sessions.morning_or_end == "09:15"
        assert cfg.sessions.morning_entry_end == "13:30"
        assert cfg.sessions.morning_profit_lock_time == "14:15"
        assert cfg.sessions.comex_or_start_edt == "17:00"
        assert cfg.sessions.comex_or_end_edt == "17:30"
        assert cfg.sessions.comex_entry_end_edt == "21:45"
        assert cfg.sessions.eod_flatten_start_edt == "23:10"
        assert cfg.sessions.comex_or_start_est == "18:00"
        assert cfg.sessions.comex_or_end_est == "18:30"
        assert cfg.sessions.comex_entry_end_est == "22:15"
        assert cfg.sessions.eod_flatten_start_est == "23:35"

        # Gates
        assert cfg.gates.min_days_to_expiry == 5
        assert cfg.gates.max_choppiness_index == 50.0
        assert cfg.gates.min_adx == 22.0
        assert cfg.gates.max_or_width_pct == 0.0065
        assert cfg.gates.breakout_atr_cushion == 0.15
        assert cfg.gates.min_clv == 0.70
        assert cfg.gates.min_rvol == 1.80
        assert cfg.gates.stop_atr_multiplier == 1.35
        assert cfg.gates.min_stop_width_pct == 0.0020
        assert cfg.gates.max_stop_width_pct == 0.0065

        # Ratchet
        assert cfg.ratchet.target_r == 2.50
        assert cfg.ratchet.stage1_trigger_r == 1.00
        assert cfg.ratchet.stage1_lock_r == 0.15
        assert cfg.ratchet.stage2_trigger_r == 1.50
        assert cfg.ratchet.stage2_lock_r == 0.65
        assert cfg.ratchet.stage3_trigger_r == 2.00
        assert cfg.ratchet.stage3_lock_r == 1.25
        assert cfg.ratchet.stage4_step_r == 0.50
        assert cfg.ratchet.stage4_trail_offset_r == 0.65
        assert cfg.ratchet.max_stop_modifications == 20

