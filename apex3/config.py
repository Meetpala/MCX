"""APEX-3 Configuration Loader.

Loads strategy parameters from config.yaml and secrets from
environment variables. Secrets are never logged or stored in
configuration files.

Usage:
    config = load_config()
    config = load_config("path/to/custom_config.yaml")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniverseItemConfig:
    symbol_root: str
    lot_multiplier: int = 1
    tick_size: float = 1.0
    max_spread_bps: float = 3.5
    min_atr_pct: float = 0.0014


@dataclass(frozen=True)
class UniverseConfig:
    core_watchlist: list[UniverseItemConfig] = field(default_factory=list)
    scale_up_watchlist: list[UniverseItemConfig] = field(default_factory=list)


@dataclass(frozen=True)
class SessionsConfig:
    morning_or_start: str = "09:00"
    morning_or_end: str = "09:15"
    morning_entry_end: str = "13:30"
    morning_profit_lock_time: str = "14:15"
    comex_or_start_edt: str = "17:00"
    comex_or_end_edt: str = "17:30"
    comex_entry_end_edt: str = "21:45"
    eod_flatten_start_edt: str = "23:10"
    comex_or_start_est: str = "18:00"
    comex_or_end_est: str = "18:30"
    comex_entry_end_est: str = "22:15"
    eod_flatten_start_est: str = "23:35"


@dataclass(frozen=True)
class GatesConfig:
    min_days_to_expiry: int = 5
    max_choppiness_index: float = 50.0
    min_adx: float = 22.0
    max_or_width_pct: float = 0.0065
    breakout_atr_cushion: float = 0.15
    min_clv: float = 0.70
    min_rvol: float = 1.80
    stop_atr_multiplier: float = 1.35
    min_stop_width_pct: float = 0.0020
    max_stop_width_pct: float = 0.0065


@dataclass(frozen=True)
class RatchetConfig:
    target_r: float = 2.50
    stage1_trigger_r: float = 1.00
    stage1_lock_r: float = 0.15
    stage2_trigger_r: float = 1.50
    stage2_lock_r: float = 0.65
    stage3_trigger_r: float = 2.00
    stage3_lock_r: float = 1.25
    stage4_step_r: float = 0.50
    stage4_trail_offset_r: float = 0.65
    max_stop_modifications: int = 20
    # MCX aliases
    stage1_r: float = 1.00
    stage1_buffer_ticks: int = 2
    stage2_r: float = 1.50
    stage3_r: float = 2.00
    stage4_r: float = 2.50


@dataclass(frozen=True)
class WatchlistConfig:
    symbols: list[str] = field(default_factory=list)
    scanner_enabled: bool = False
    scanner_universe: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LimitsConfig:
    max_symbols: int = 2
    max_leverage: float = 4.0
    min_accepted_leverage: float = 3.5
    safety_reserve_pct: float = 5.0


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.80
    max_combined_open_risk_pct: float = 1.60
    daily_loss_limit_pct: float = 1.60
    daily_max_loss_pct: float = 2.50
    max_daily_loss_pct: float = 2.50
    max_fee_to_profit_ratio: float = 0.18
    slippage_allowance_pct: float = 0.0
    bound_loss_with_charges: bool = True
    max_concurrent_positions: int = 2
    max_new_entries_per_bar: int = 1
    second_slot_requires_slot1_profit_r: float = 0.0


@dataclass(frozen=True)
class RegimeConfig:
    benchmark_symbol: str = "NIFTY 50"
    benchmark_exchange: str = "NSE"
    ema_period: int = 20
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    ema_slope_lookback: int = 3
    min_adx: float = 22.0
    adx_period: int = 14
    require_vwap_alignment: bool = True
    require_rising_adx: bool = True


@dataclass(frozen=True)
class SignalConfig:
    min_relative_volume: float = 1.5
    volume_avg_period: int = 20
    max_spread_bps: float = 8.0
    min_circuit_distance_pct: float = 2.0
    candle_timeframe_entry: str = "5min"
    candle_timeframe_trail: str = "1min"
    relative_strength_lookback: int = 20
    min_stop_distance_pct: float = 0.0055
    max_stop_distance_pct: float = 0.0090
    min_breakout_cushion_atr: float = 0.15
    min_clv_ratio: float = 0.70
    min_atr_pct: float = 0.0022
    max_quote_age_sec: float = 15.0


@dataclass(frozen=True)
class EntryConfig:
    order_type: str = "LIMIT"
    max_slippage_pct: float = 0.15
    unfilled_timeout_sec: int = 30
    max_retries: int = 1
    exit_order_type: str = "LIMIT"
    marketable_limit_buffer_pct: float = 0.5


@dataclass(frozen=True)
class TrailConfig:
    activation_r: float = 1.0
    profit_lock_rupees: float = 0.50
    atr_multiplier_normal: float = 2.5
    atr_multiplier_tight: float = 1.5
    min_step_ticks: int = 2
    swing_pad_ticks: int = 2
    exhaustion_adx_threshold: float = 18.0
    late_session_tighten_time: str = "13:30"


@dataclass(frozen=True)
class StopsConfig:
    initial_atr_multiplier: float = 1.5
    initial_atr_period: int = 14
    swing_lookback_bars: int = 5
    sl_limit_buffer_pct: float = 0.25
    target_r: float = 2.0
    use_fixed_target: bool = True
    trail_enabled: bool = False
    fee_lock_enabled: bool = True
    fee_lock_trigger_r: float = 1.60
    fee_lock_profit_r: float = 0.60
    late_day_lock_trigger_r: float = 1.15
    late_day_lock_profit_r: float = 0.35
    late_day_protection_time: str = "14:15"
    fee_lock_offset_pct: float = 0.0015
    trail: TrailConfig = field(default_factory=TrailConfig)
    max_modifications_per_order: int = 25
    modification_headroom: int = 3


@dataclass(frozen=True)
class ScheduleConfig:
    market_open: str = "09:15"
    opening_range_end: str = "09:30"
    entry_start: str = "09:35"
    entry_cutoff: str = "12:30"
    flatten_start: str = "14:55"
    flat_target: str = "14:58"
    market_close: str = "15:30"
    hard_deadline: str = "15:00"
    stale_threshold_sec: int = 15
    holidays: list[str] = field(default_factory=list)
    special_sessions: dict[str, str] = field(default_factory=dict)

    def get_time(self, field_name: str) -> time:
        """Parse an HH:MM time string from the config."""
        val = getattr(self, field_name)
        parts = val.split(":")
        return time(int(parts[0]), int(parts[1]))

    def is_holiday(self, check_date: date) -> bool:
        """Check if a date is in the configured holiday list."""
        date_str = check_date.isoformat()
        return date_str in self.holidays

    def is_special_session(self, check_date: date) -> tuple[bool, str]:
        """Check if a date has a special trading session (e.g. Muhurat Trading)."""
        date_str = check_date.isoformat()
        if date_str in self.special_sessions:
            return True, self.special_sessions[date_str]
        return False, ""


@dataclass(frozen=True)
class RateLimitsConfig:
    orders_per_second: int = 10
    orders_per_minute: int = 200
    quotes_per_second: int = 5
    historical_per_second: int = 3
    general_per_second: int = 10


@dataclass(frozen=True)
class CostsConfig:
    brokerage_per_order: float = 20.0
    stt_pct: float = 0.025
    exchange_txn_pct: float = 0.00318
    gst_pct: float = 18.0
    sebi_turnover_pct: float = 0.0001
    stamp_duty_pct: float = 0.003


@dataclass(frozen=True)
class JournalConfig:
    db_path: str = "apex3_journal.db"
    log_decisions: bool = True
    log_orders: bool = True
    log_fills: bool = True
    log_stop_changes: bool = True
    log_warnings: bool = True
    log_daily_summary: bool = True


@dataclass(frozen=True)
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    refresh_interval_sec: int = 2
    auth_required: bool = False
    api_token: str = ""


@dataclass(frozen=True)
class SecretsConfig:
    """Secrets loaded from environment variables only.

    WARNING: Never log, print, serialize, or store these values
    in files, commits, screenshots, or frontend code.
    """
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""
    angel_api_key: str = ""
    angel_client_code: str = ""
    angel_pin: str = ""
    angel_totp_secret: str = ""
    live_trading_enabled: bool = False
    static_ip_configured: bool = False
    sebi_compliance_acknowledged: bool = False


@dataclass
class AppConfig:
    """Complete application configuration."""
    mode: str = "paper"
    paper_starting_capital: float = 30_000.0
    exchange: str = "MCX"
    margin_segment: str = "commodity"
    fetch_live_zerodha_margin: bool = True
    base_capital_inr: float = 30_000.0
    safety_reserve_buffer_pct: float = 0.05
    max_concurrent_positions: int = 2
    max_new_entries_per_bar: int = 1
    per_trade_risk_pct: float = 0.0125
    daily_max_loss_pct: float = 0.0250
    bound_loss_with_charges: bool = True
    max_fee_to_profit_ratio: float = 0.18
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    ratchet: RatchetConfig = field(default_factory=RatchetConfig)
    watchlist: WatchlistConfig = field(default_factory=WatchlistConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    stops: StopsConfig = field(default_factory=StopsConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    rate_limits: RateLimitsConfig = field(default_factory=RateLimitsConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    journal: JournalConfig = field(default_factory=JournalConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    secrets: SecretsConfig = field(default_factory=SecretsConfig)
    raw_dict: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_secrets() -> SecretsConfig:
    """Load secrets from environment variables.

    Never reads from files, config, or defaults to real values.
    """
    return SecretsConfig(
        kite_api_key=os.environ.get("KITE_API_KEY", ""),
        kite_api_secret=os.environ.get("KITE_API_SECRET", ""),
        kite_access_token=os.environ.get("KITE_ACCESS_TOKEN", ""),
        angel_api_key=os.environ.get("ANGEL_API_KEY", ""),
        angel_client_code=os.environ.get("ANGEL_CLIENT_CODE", ""),
        angel_pin=os.environ.get("ANGEL_PIN", ""),
        angel_totp_secret=os.environ.get("ANGEL_TOTP_SECRET", ""),
        live_trading_enabled=os.environ.get(
            "LIVE_TRADING_ENABLED", "false"
        ).lower() == "true",
        static_ip_configured=os.environ.get(
            "STATIC_IP_CONFIGURED", "false"
        ).lower() == "true",
        sebi_compliance_acknowledged=os.environ.get(
            "SEBI_COMPLIANCE_ACKNOWLEDGED", "false"
        ).lower() == "true",
    )


def _parse_schedule(raw: dict, fallback_raw: Optional[dict] = None) -> ScheduleConfig:
    """Parse schedule section, collecting holidays from all schedule and holiday keys."""
    merged = dict(fallback_raw or {})
    merged.update(raw)

    holidays: list[str] = []
    special_sessions: dict[str, str] = {}
    sources = [raw, fallback_raw or {}, merged]
    for s in sources:
        if isinstance(s, dict):
            for key, value in s.items():
                if (key.startswith("holidays") or key == "holiday_list") and isinstance(value, list):
                    for v in value:
                        v_str = str(v)
                        if v_str not in holidays:
                            holidays.append(v_str)
                elif "special_sessions" in key and isinstance(value, dict):
                    special_sessions.update({str(k): str(v) for k, v in value.items()})

    return ScheduleConfig(
        market_open=str(merged.get("or_start_time", merged.get("market_open", "09:15"))),
        opening_range_end=str(merged.get("or_end_time", merged.get("opening_range_end", "09:30"))),
        entry_start=str(merged.get("entry_start_time", merged.get("entry_start", "09:35"))),
        entry_cutoff=str(merged.get("entry_cutoff_time", merged.get("entry_cutoff", "12:30"))),
        flatten_start=str(merged.get("square_off_time", merged.get("flatten_start", "14:55"))),
        flat_target=str(merged.get("flat_verification_time", merged.get("flat_target", "14:58"))),
        market_close=str(merged.get("market_close", "15:30")),
        hard_deadline=str(merged.get("hard_deadline", "15:00")),
        stale_threshold_sec=int(merged.get("stale_threshold_sec", 15)),
        holidays=sorted(holidays),
        special_sessions=special_sessions,
    )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load application configuration from YAML file and environment.

    Args:
        config_path: Path to config.yaml. If None, uses the default
                     config.yaml in the apex3 package directory.

    Returns:
        Fully populated AppConfig instance.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Support both new 3-tier schema and baseline schema
    acc_raw = raw.get("account", {})
    risk_raw = raw.get("risk_management", raw.get("risk", {}))
    regime_raw = raw.get("regime_filter", raw.get("regime", {}))
    signal_raw = raw.get("signal_gates", raw.get("signal", {}))
    stops_raw = raw.get("stops_and_targets", raw.get("stops", {}))
    session_sched = raw.get("session_schedule", {})
    sched_section = raw.get("schedule", {})
    schedule_raw = dict(sched_section)
    schedule_raw.update(session_sched)
    radar_raw = raw.get("radar", {})
    watchlist_raw = raw.get("watchlist", {})
    limits_raw = raw.get("limits", {})
    entry_raw = raw.get("entry", {})
    rate_limits_raw = raw.get("rate_limits", {})
    costs_raw = raw.get("costs", {})
    journal_raw = raw.get("journal", {})
    dashboard_raw = raw.get("dashboard", {})

    # MCX Production Fields
    exchange = str(raw.get("exchange", "MCX"))
    margin_segment = str(raw.get("margin_segment", "commodity"))
    fetch_live_margin = bool(raw.get("fetch_live_zerodha_margin", True))
    base_cap = float(raw.get("base_capital_inr", acc_raw.get("starting_capital", raw.get("paper_starting_capital", 30000.0))))
    safety_buffer = float(raw.get("safety_reserve_buffer_pct", acc_raw.get("safety_reserve_buffer_pct", limits_raw.get("safety_reserve_pct", 0.05))))
    max_positions = int(raw.get("max_concurrent_positions", risk_raw.get("max_concurrent_positions", 2)))
    max_entries = int(raw.get("max_new_entries_per_bar", risk_raw.get("max_new_entries_per_bar", 1)))
    per_trade_risk = float(raw.get("per_trade_risk_pct", risk_raw.get("risk_per_trade_pct", 0.0125)))
    daily_max_loss = float(raw.get("daily_max_loss_pct", risk_raw.get("daily_loss_limit_pct", 0.0250)))
    bound_loss = bool(raw.get("bound_loss_with_charges", risk_raw.get("bound_loss_with_charges", True)))
    max_fee_ratio = float(raw.get("max_fee_to_profit_ratio", 0.18))

    univ_raw = raw.get("universe", {})
    core_items = [
        UniverseItemConfig(
            symbol_root=item.get("symbol_root", ""),
            lot_multiplier=int(item.get("lot_multiplier", 1)),
            tick_size=float(item.get("tick_size", 1.0)),
            max_spread_bps=float(item.get("max_spread_bps", 3.5)),
            min_atr_pct=float(item.get("min_atr_pct", 0.0014)),
        )
        for item in univ_raw.get("core_watchlist", [])
        if isinstance(item, dict)
    ]
    scale_items = [
        UniverseItemConfig(
            symbol_root=item.get("symbol_root", ""),
            lot_multiplier=int(item.get("lot_multiplier", 1)),
            tick_size=float(item.get("tick_size", 1.0)),
            max_spread_bps=float(item.get("max_spread_bps", 3.0)),
            min_atr_pct=float(item.get("min_atr_pct", 0.0010)),
        )
        for item in univ_raw.get("scale_up_watchlist", [])
        if isinstance(item, dict)
    ]
    universe_config = UniverseConfig(core_watchlist=core_items, scale_up_watchlist=scale_items)

    sessions_raw = raw.get("sessions", {})
    sessions_config = SessionsConfig(
        morning_or_start=str(sessions_raw.get("morning_or_start", "09:00")),
        morning_or_end=str(sessions_raw.get("morning_or_end", "09:15")),
        morning_entry_end=str(sessions_raw.get("morning_entry_end", "13:30")),
        morning_profit_lock_time=str(sessions_raw.get("morning_profit_lock_time", "14:15")),
        comex_or_start_edt=str(sessions_raw.get("comex_or_start_edt", "17:00")),
        comex_or_end_edt=str(sessions_raw.get("comex_or_end_edt", "17:30")),
        comex_entry_end_edt=str(sessions_raw.get("comex_entry_end_edt", "21:45")),
        eod_flatten_start_edt=str(sessions_raw.get("eod_flatten_start_edt", "23:10")),
        comex_or_start_est=str(sessions_raw.get("comex_or_start_est", "18:00")),
        comex_or_end_est=str(sessions_raw.get("comex_or_end_est", "18:30")),
        comex_entry_end_est=str(sessions_raw.get("comex_entry_end_est", "22:15")),
        eod_flatten_start_est=str(sessions_raw.get("eod_flatten_start_est", "23:35")),
    )

    gates_raw = raw.get("gates", {})
    gates_config = GatesConfig(
        min_days_to_expiry=int(gates_raw.get("min_days_to_expiry", 5)),
        max_choppiness_index=float(gates_raw.get("max_choppiness_index", 50.0)),
        min_adx=float(gates_raw.get("min_adx", 22.0)),
        max_or_width_pct=float(gates_raw.get("max_or_width_pct", 0.0065)),
        breakout_atr_cushion=float(gates_raw.get("breakout_atr_cushion", 0.15)),
        min_clv=float(gates_raw.get("min_clv", 0.70)),
        min_rvol=float(gates_raw.get("min_rvol", 1.80)),
        stop_atr_multiplier=float(gates_raw.get("stop_atr_multiplier", 1.35)),
        min_stop_width_pct=float(gates_raw.get("min_stop_width_pct", 0.0020)),
        max_stop_width_pct=float(gates_raw.get("max_stop_width_pct", 0.0065)),
    )

    ratchet_raw = raw.get("ratchet", {})
    ratchet_config = RatchetConfig(
        target_r=float(ratchet_raw.get("target_r", 2.50)),
        stage1_trigger_r=float(ratchet_raw.get("stage1_trigger_r", 1.00)),
        stage1_lock_r=float(ratchet_raw.get("stage1_lock_r", 0.15)),
        stage2_trigger_r=float(ratchet_raw.get("stage2_trigger_r", 1.50)),
        stage2_lock_r=float(ratchet_raw.get("stage2_lock_r", 0.65)),
        stage3_trigger_r=float(ratchet_raw.get("stage3_trigger_r", 2.00)),
        stage3_lock_r=float(ratchet_raw.get("stage3_lock_r", 1.25)),
        stage4_step_r=float(ratchet_raw.get("stage4_step_r", 0.50)),
        stage4_trail_offset_r=float(ratchet_raw.get("stage4_trail_offset_r", 0.65)),
        max_stop_modifications=int(ratchet_raw.get("max_stop_modifications", 20)),
    )

    # Build merged watchlist from radar tiers or universe core watchlist
    merged_symbols = list(radar_raw.get("core_watchlist", []))
    if radar_raw.get("enable_extended_radar", True):
        merged_symbols.extend(radar_raw.get("extended_radar", []))
    merged_symbols.extend(radar_raw.get("custom_user_radar", []))
    if not merged_symbols:
        merged_symbols = [item.symbol_root for item in core_items]
    if not merged_symbols:
        merged_symbols = watchlist_raw.get("symbols", [])

    # Build trail config
    trail_raw = stops_raw.get("trail", {})
    trail_config = TrailConfig(
        activation_r=float(trail_raw.get("activation_r", 1.0)),
        profit_lock_rupees=float(trail_raw.get("profit_lock_rupees", 0.50)),
        atr_multiplier_normal=float(trail_raw.get("atr_multiplier_normal", 2.5)),
        atr_multiplier_tight=float(trail_raw.get("atr_multiplier_tight", 1.5)),
        min_step_ticks=int(trail_raw.get("min_step_ticks", 2)),
        swing_pad_ticks=int(trail_raw.get("swing_pad_ticks", 2)),
        exhaustion_adx_threshold=float(trail_raw.get("exhaustion_adx_threshold", 18.0)),
        late_session_tighten_time=str(trail_raw.get("late_session_tighten_time", "13:30")),
    )

    safety_pct = safety_buffer
    if safety_pct <= 1.0:
        safety_pct *= 100.0

    risk_per_trade = per_trade_risk
    if risk_per_trade <= 0.1:
        risk_per_trade *= 100.0

    daily_loss = daily_max_loss
    if daily_loss <= 0.1:
        daily_loss *= 100.0

    circuit_dist = float(signal_raw.get("min_circuit_distance_pct", 2.0))
    if circuit_dist <= 0.1:
        circuit_dist *= 100.0

    return AppConfig(
        mode=str(raw.get("mode", "paper")),
        paper_starting_capital=float(acc_raw.get("starting_capital", raw.get("paper_starting_capital", base_cap))),
        exchange=exchange,
        margin_segment=margin_segment,
        fetch_live_zerodha_margin=fetch_live_margin,
        base_capital_inr=base_cap,
        safety_reserve_buffer_pct=safety_buffer,
        max_concurrent_positions=max_positions,
        max_new_entries_per_bar=max_entries,
        per_trade_risk_pct=per_trade_risk,
        daily_max_loss_pct=daily_max_loss,
        bound_loss_with_charges=bound_loss,
        max_fee_to_profit_ratio=max_fee_ratio,
        universe=universe_config,
        sessions=sessions_config,
        gates=gates_config,
        ratchet=ratchet_config,
        watchlist=WatchlistConfig(
            symbols=merged_symbols,
            scanner_enabled=bool(watchlist_raw.get("scanner_enabled", bool(radar_raw or core_items))),
            scanner_universe=merged_symbols,
        ),
        limits=LimitsConfig(
            max_symbols=max_positions,
            max_leverage=float(acc_raw.get("max_intraday_leverage", limits_raw.get("max_leverage", 4.0))),
            min_accepted_leverage=float(limits_raw.get("min_accepted_leverage", 3.5)),
            safety_reserve_pct=safety_pct,
        ),
        risk=RiskConfig(
            risk_per_trade_pct=risk_per_trade,
            max_combined_open_risk_pct=float(risk_raw.get("max_combined_open_risk_pct", 1.60)),
            daily_loss_limit_pct=daily_loss,
            slippage_allowance_pct=float(risk_raw.get("slippage_allowance_pct", 0.0)),
            bound_loss_with_charges=bound_loss,
            max_concurrent_positions=max_positions,
            max_new_entries_per_bar=max_entries,
            second_slot_requires_slot1_profit_r=float(risk_raw.get("second_slot_requires_slot1_profit_r", 0.0)),
        ),
        regime=RegimeConfig(
            benchmark_symbol=str(regime_raw.get("benchmark_symbol", "NIFTY 50")),
            benchmark_exchange=str(regime_raw.get("benchmark_exchange", "NSE")),
            ema_period=int(regime_raw.get("ema_period", 20)),
            ema_fast_period=int(regime_raw.get("ema_fast_period", 20)),
            ema_slow_period=int(regime_raw.get("ema_slow_period", 50)),
            ema_slope_lookback=int(regime_raw.get("ema_slope_lookback", 3)),
            min_adx=float(gates_config.min_adx),
            adx_period=int(regime_raw.get("adx_period", 14)),
            require_vwap_alignment=bool(regime_raw.get("require_vwap_alignment", True)),
            require_rising_adx=bool(regime_raw.get("require_rising_adx", True)),
        ),
        signal=SignalConfig(
            min_relative_volume=float(gates_config.min_rvol),
            volume_avg_period=int(signal_raw.get("volume_avg_bars", signal_raw.get("volume_avg_period", 20))),
            max_spread_bps=float(signal_raw.get("max_spread_bps", 3.5)),
            min_circuit_distance_pct=circuit_dist,
            candle_timeframe_entry=str(signal_raw.get("candle_timeframe_entry", "5min")),
            candle_timeframe_trail=str(signal_raw.get("candle_timeframe_trail", "1min")),
            relative_strength_lookback=int(signal_raw.get("rs_lookback_bars", signal_raw.get("relative_strength_lookback", 20))),
            min_stop_distance_pct=float(gates_config.min_stop_width_pct),
            max_stop_distance_pct=float(gates_config.max_stop_width_pct),
            min_breakout_cushion_atr=float(gates_config.breakout_atr_cushion),
            min_clv_ratio=float(gates_config.min_clv),
            min_atr_pct=float(signal_raw.get("min_atr_pct", 0.0014)),
            max_quote_age_sec=float(signal_raw.get("max_quote_age_sec", 15.0)),
        ),
        entry=EntryConfig(
            order_type=str(entry_raw.get("order_type", "LIMIT")),
            max_slippage_pct=float(entry_raw.get("max_slippage_pct", 0.15)),
            unfilled_timeout_sec=int(entry_raw.get("unfilled_timeout_sec", 30)),
            max_retries=int(entry_raw.get("max_retries", 1)),
            exit_order_type=str(entry_raw.get("exit_order_type", "LIMIT")),
            marketable_limit_buffer_pct=float(entry_raw.get("marketable_limit_buffer_pct", 0.15)),
        ),
        stops=StopsConfig(
            initial_atr_multiplier=float(gates_config.stop_atr_multiplier),
            initial_atr_period=int(stops_raw.get("atr_period", stops_raw.get("initial_atr_period", 14))),
            swing_lookback_bars=int(stops_raw.get("swing_lookback_bars", 5)),
            sl_limit_buffer_pct=float(stops_raw.get("sl_limit_buffer_pct", 0.25)),
            target_r=float(stops_raw.get("target_r_multiple", stops_raw.get("target_r", 2.0))),
            use_fixed_target=bool(stops_raw.get("use_fixed_target", True)),
            trail_enabled=bool(stops_raw.get("trail_enabled", False)),
            fee_lock_enabled=bool(stops_raw.get("fee_lock_enabled", True)),
            fee_lock_trigger_r=float(stops_raw.get("fee_lock_trigger_r", ratchet_config.stage1_trigger_r)),
            fee_lock_profit_r=float(stops_raw.get("fee_lock_profit_r", ratchet_config.stage1_lock_r)),
            late_day_lock_trigger_r=float(stops_raw.get("late_day_lock_trigger_r", 1.15)),
            late_day_lock_profit_r=float(stops_raw.get("late_day_lock_profit_r", 0.35)),
            late_day_protection_time=str(sessions_config.morning_profit_lock_time),
            fee_lock_offset_pct=float(stops_raw.get("fee_lock_offset_pct", 0.0015)),
            trail=trail_config,
            max_modifications_per_order=int(ratchet_config.max_stop_modifications),
            modification_headroom=int(stops_raw.get("modification_headroom", 3)),
        ),
        schedule=_parse_schedule(session_sched, sched_section),
        rate_limits=RateLimitsConfig(
            orders_per_second=int(rate_limits_raw.get("orders_per_second", 10)),
            orders_per_minute=int(rate_limits_raw.get("orders_per_minute", 200)),
            quotes_per_second=int(rate_limits_raw.get("quotes_per_second", 5)),
            historical_per_second=int(rate_limits_raw.get("historical_per_second", 3)),
            general_per_second=int(rate_limits_raw.get("general_per_second", 10)),
        ),
        costs=CostsConfig(
            brokerage_per_order=float(costs_raw.get("brokerage_per_order", 20.0)),
            stt_pct=float(costs_raw.get("stt_pct", 0.025)),
            exchange_txn_pct=float(costs_raw.get("exchange_txn_pct", 0.00318)),
            gst_pct=float(costs_raw.get("gst_pct", 18.0)),
            sebi_turnover_pct=float(costs_raw.get("sebi_turnover_pct", 0.0001)),
            stamp_duty_pct=float(costs_raw.get("stamp_duty_pct", 0.003)),
        ),
        journal=JournalConfig(
            db_path=str(journal_raw.get("db_path", "apex3_journal.db")),
            log_decisions=bool(journal_raw.get("log_decisions", True)),
            log_orders=bool(journal_raw.get("log_orders", True)),
            log_fills=bool(journal_raw.get("log_fills", True)),
            log_stop_changes=bool(journal_raw.get("log_stop_changes", True)),
            log_warnings=bool(journal_raw.get("log_warnings", True)),
            log_daily_summary=bool(journal_raw.get("log_daily_summary", True)),
        ),
        dashboard=DashboardConfig(
            host=str(dashboard_raw.get("host", "127.0.0.1")),
            port=int(dashboard_raw.get("port", 5000)),
            refresh_interval_sec=int(dashboard_raw.get("refresh_interval_sec", 2)),
            auth_required=bool(dashboard_raw.get("auth_required", False)),
            api_token=str(dashboard_raw.get("api_token", "")),
        ),
        secrets=_load_secrets(),
        raw_dict=raw,
    )
