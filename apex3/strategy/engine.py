"""APEX-3 Deterministic Strategy Engine.

Executes the 12-step deterministic decision pipeline:
 1. Session and compliance checks
 2. Connectivity, clock sync, and quote freshness
 3. Account reconciliation (pauses on foreign/manual positions)
 4. Broad-market regime filter (NIFTY 50 EMA-20 slope + ADX >= 22)
 5. Universe / Watchlist validation (max 3 symbols, MIS available)
 6. Opening range construction (09:15 - 09:30 IST)
 7. Stock signal generation (mirror rules for Long/Short, 8 gates)
 8. Position sizing and equal allocation across qualifying candidates
 9. Bounded entry execution (LIMIT order, verified broker fills)
10. Immediate protective stop placement for actual filled quantity
11. Position management: Monotonic trailing stop (+1R, ATR/structure)
12. End-of-day square-off: cutoff at 14:15, flatten at 14:55, verify flat by 14:58

CRITICAL: Fail-closed architecture. All decisions return one of:
WAIT, PROPOSED_LONG, PROPOSED_SHORT, MANAGE, EXIT, HALT.
"""
from __future__ import annotations

import logging
import threading
import time as time_mod
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from apex3.broker.interface import BrokerInterface
from apex3.config import AppConfig
from apex3.indicators.atr import compute_atr
from apex3.journal.sqlite_journal import Journal
from apex3.journal.trade_logger import TradeLogger
from apex3.models import (
    Candle,
    Decision,
    GateResult,
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
    StrategyDecision,
    make_broker_tag,
)
from apex3.reconciliation.reconciler import Reconciler
from apex3.reconciliation.state_machine import PositionPhase, StateMachine
from apex3.strategy.accounting import build_session_pnl
from apex3.strategy.regime import MarketRegime, evaluate_market_regime
from apex3.strategy.risk import evaluate_portfolio_risk
from apex3.strategy.schedule import SessionScheduler, SessionState
from apex3.strategy.signal import build_opening_range, evaluate_signal, validate_late_start_signal
from apex3.strategy.sizer import size_position
from apex3.strategy.trail import (
    calculate_initial_stop,
    calculate_stop_limit_prices,
    calculate_target_price,
    ceil_to_tick,
    evaluate_trailing_stop,
    floor_to_tick,
    round_to_tick,
)

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("apex3.engine")


def build_active_radar(cfg: dict) -> list[str]:
    """
    Merges Core Watchlist + Extended Radar + Custom User Radar into a deduplicated list.
    """
    radar_cfg = cfg.get("radar", {})
    symbols = list(radar_cfg.get("core_watchlist", []))
    if radar_cfg.get("enable_extended_radar", True):
        symbols.extend(radar_cfg.get("extended_radar", []))
    symbols.extend(radar_cfg.get("custom_user_radar", []))

    # Fallback to watchlist if radar is empty
    if not symbols:
        wl = cfg.get("watchlist", {})
        symbols = list(wl.get("symbols", [])) or list(wl.get("scanner_universe", []))

    # Deduplicate while preserving order
    seen = set()
    active_radar = []
    for sym in symbols:
        clean_sym = sym.strip().upper()
        if clean_sym and clean_sym not in seen:
            seen.add(clean_sym)
            active_radar.append(clean_sym)
    return active_radar


class StrategyEngine:
    """The central deterministic trading system for APEX-3."""

    def __init__(
        self,
        config: AppConfig,
        broker: BrokerInterface,
        journal: Journal,
        session_tag: str = "APEX3",
        trade_logger: Optional[TradeLogger] = None,
        macro_provider: Optional[Any] = None,
    ):
        self.config = config
        self.broker = broker
        self.journal = journal
        self.session_tag = session_tag
        self.trade_logger = trade_logger or TradeLogger(log_path="info.log", start_date="2026-10-28")

        from apex3.news.macro_scraper import MacroNewsProvider
        self.macro_provider = macro_provider or MacroNewsProvider(self.config)
        self.latest_macro_state: Optional[Any] = None

        self.scheduler = SessionScheduler(config.schedule)
        self.state_machine = StateMachine(session_tag_prefix=session_tag)
        self.reconciler = Reconciler(broker, self.state_machine, journal, session_tag, config=self.config)

        self.is_halted = False
        self.halt_reason = ""
        self.is_paused = False
        self.pause_reason = ""
        self._daily_loss_halt_date: Optional[date] = None
        self._consecutive_pnl_failures: int = 0
        self._current_session_date: Optional[date] = None
        self._flatten_completed: bool = False
        self._last_entry_bar_timestamp: Optional[datetime] = None
        self._execution_lock = threading.Lock()
        self.last_decisions: list[StrategyDecision] = []

        # Opening ranges cache: symbol -> OpeningRange
        self.opening_ranges: dict[str, OpeningRange] = {}
        self.latest_regime_eval: Optional[Any] = None
        self.latest_session_pnl: Optional[Any] = None
        self._late_start_catchup_done: set[str] = set()

    def _ensure_ist(self, dt: datetime) -> datetime:
        """Normalize any naive or aware datetime to Asia/Kolkata (IST)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=IST)
        return dt.astimezone(IST)

    def _filter_completed_candles(self, candles: list[Candle], now: datetime) -> list[Candle]:
        """Return only completed candles whose bar close timestamp is <= now."""
        now_ist = self._ensure_ist(now)
        res = []
        for c in candles:
            if not getattr(c, "is_complete", True):
                continue
            ts = self._ensure_ist(c.timestamp)
            if ts + timedelta(minutes=5) <= now_ist + timedelta(seconds=2):
                res.append(c)
        return res or [c for c in candles if getattr(c, "is_complete", True)]


    def _confirm_order_terminal(self, order_id: str, timeout_sec: float = 3.0) -> Optional[OrderState]:
        """Poll broker until order reaches a terminal state (CANCELLED, REJECTED, COMPLETE)."""
        import time as time_mod
        start_t = time_mod.time()
        last_state = None
        while time_mod.time() - start_t < timeout_sec:
            st = self.broker.get_order_status(order_id)
            if st:
                last_state = st
                if st.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.COMPLETE):
                    return st
            time_mod.sleep(0.05)
        return last_state

    def _confirm_protective_sl(self, sl_order_id: str, expected_qty: int, timeout_sec: float = 2.0) -> bool:
        """Poll broker for newly placed SL order while in transitional PENDING states until TRIGGER_PENDING."""
        import time as time_mod
        start_t = time_mod.time()
        while time_mod.time() - start_t < timeout_sec:
            sl_state = self.broker.get_order_status(sl_order_id)
            if sl_state:
                sl_qty = (
                    getattr(sl_state, "quantity", None)
                    or getattr(getattr(sl_state, "request", None), "quantity", 0)
                    or getattr(sl_state, "pending_quantity", 0)
                )
                if sl_state.status == OrderStatus.TRIGGER_PENDING and sl_qty == expected_qty:
                    return True
                if sl_state.status not in (OrderStatus.PENDING, None):
                    break
            time_mod.sleep(0.05)
        return False

    def _create_exit_order_request(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        tag: str,
        current_price: float = 0.0,
    ) -> OrderRequest:
        """Create an exchange-compliant, bounded marketable limit exit order request.

        Under NSE Retail Algo guidelines, algorithmic market orders are prohibited.
        All exits must be bounded limit orders priced using fresh quotes (best bid for SELL,
        best ask for BUY) and constrained within circuit limits and configured slippage caps.
        """
        inst = self.broker.get_instrument_info(symbol)
        tick = float(inst.tick_size) if inst and isinstance(getattr(inst, "tick_size", None), (int, float)) and inst.tick_size > 0 else 0.05

        # Enforce max_slippage_pct cap on marketable limit buffer
        max_slip = getattr(getattr(self.config, "entry", None), "max_slippage_pct", 0.15) if self.config else 0.15
        cfg_buf = getattr(getattr(self.config, "entry", None), "marketable_limit_buffer_pct", 0.15) if self.config else 0.15
        buffer_pct = min(cfg_buf, max_slip) / 100.0

        # Query fresh executable quote (best bid for SELL, best ask for BUY)
        px = 0.0
        now_ts = datetime.now(IST)
        for _ in range(3):
            try:
                q = self.broker.get_quote(symbol)
                is_fresh = getattr(q, "is_fresh", True)
                q_ts = getattr(q, "timestamp", None)
                age_sec = (now_ts - q_ts).total_seconds() if q_ts else 0.0
                if q and isinstance(getattr(q, "last_price", None), (int, float)) and q.last_price > 0 and is_fresh and age_sec <= 15.0:
                    if side == Side.SELL and isinstance(getattr(q, "bid", None), (int, float)) and q.bid > 0:
                        px = float(q.bid)
                    elif side == Side.BUY and isinstance(getattr(q, "ask", None), (int, float)) and q.ask > 0:
                        px = float(q.ask)
                    else:
                        px = float(q.last_price)
                    if px > 0.0:
                        break
            except Exception:
                pass
            time_mod.sleep(0.05)

        if px <= 0.0 and isinstance(current_price, (int, float)) and current_price > 0:
            px = float(current_price)

        if px <= 0.0:
            err_msg = f"Cannot construct exit order for {symbol}: valid price quote is required (got {px})"
            logger.critical(err_msg)
            raise ValueError(err_msg)

        lower_c = float(inst.lower_circuit) if inst and isinstance(getattr(inst, "lower_circuit", None), (int, float)) and inst.lower_circuit > 0 else 0.0
        upper_c = float(inst.upper_circuit) if inst and isinstance(getattr(inst, "upper_circuit", None), (int, float)) and inst.upper_circuit > 0 else 0.0

        if side == Side.SELL:
            raw_limit = px * (1.0 - buffer_pct)
            if lower_c > 0:
                raw_limit = max(raw_limit, lower_c)
            limit_px = floor_to_tick(raw_limit, tick)
        else:
            raw_limit = px * (1.0 + buffer_pct)
            if upper_c > 0:
                raw_limit = min(raw_limit, upper_c)
            limit_px = ceil_to_tick(raw_limit, tick)

        return OrderRequest(
            symbol=symbol,
            exchange=getattr(self.config, "exchange", "MCX"),
            side=side,
            quantity=quantity,
            product=Product.MIS,
            order_type=OrderType.LIMIT,
            price=limit_px,
            tag=tag,
            strategy_id=self.session_tag,
        )

    def has_in_flight_orders(self) -> bool:
        """Return True if any managed position has orders in-flight or is unprotected."""
        for pos in self.state_machine.positions.values():
            if pos.phase in (PositionPhase.ENTERING, PositionPhase.OPEN_UNPROTECTED, PositionPhase.EXITING):
                return True
            if pos.phase == PositionPhase.OPEN_PROTECTED and not pos.protective_order_id:
                return True
        return False

    def get_current_regime(self) -> Optional[Any]:
        """Return the most recently evaluated market regime without extra API calls."""
        return getattr(self, "latest_regime_eval", None)

    def get_active_watchlist(self) -> list[str]:
        """Return manual symbols or the full scanner candidate universe dynamically."""
        if getattr(self, "_manual_watchlist_override", None):
            return self._manual_watchlist_override
        if hasattr(self.config, "raw_dict") and isinstance(self.config.raw_dict, dict) and "radar" in self.config.raw_dict:
            radar_symbols = build_active_radar(self.config.raw_dict)
            if radar_symbols:
                return radar_symbols
        if not self.config.watchlist.scanner_enabled:
            # Manual mode: user selected symbols
            return self.config.watchlist.symbols
        else:
            # Scanner mode evaluates the full universe, then ranks valid signals.
            return list(self.config.watchlist.scanner_universe)

    def set_manual_symbols(self, symbols: list[str]) -> list[str]:
        """Update the active watchlist dynamically without artificial caps."""
        valid_symbols = [s.strip().upper() for s in symbols if s.strip()]
        if not valid_symbols or "ALL" in valid_symbols or "DEFAULT" in valid_symbols or "RESET" in valid_symbols:
            self._manual_watchlist_override = None
            object.__setattr__(self.config.watchlist, "scanner_enabled", True)
            return self.get_active_watchlist()

        self._manual_watchlist_override = valid_symbols
        object.__setattr__(self.config.watchlist, "symbols", valid_symbols)
        object.__setattr__(self.config.watchlist, "scanner_enabled", False)
        return valid_symbols

    def reconcile_and_backfill_late_start(self, current_time: Optional[datetime] = None) -> dict:
        """Execute full Phase 2 Late-Start ('start any time') reconciliation and candle backfill.

        1. Fetches positions, open orders, and today's trades from broker(s) and rebuilds state.
        2. Immediately re-attaches protective SL orders to any unprotected strategy positions.
        3. Backfills 5m candles from 09:15 IST to `now` and recomputes Opening Range, VWAP, and EMAs.
        4. Updates `best_favorable_price` (HWM/LWM) on recovered open positions from backfilled candles.
        """
        with self._execution_lock:
            now = self._ensure_ist(current_time or self.scheduler.now_ist())
            session_date = now.date()
            self._current_session_date = session_date
            self._late_start_catchup_done.clear()
            symbols = self.get_active_watchlist()

            # 1. Reconcile positions, orders, and trades; re-attach missing SL orders
            audit = self.reconciler.audit_startup_state(symbols, reattach_stops=True)

            # 2. Backfill 5m candles for benchmark and watchlist symbols
            hist_from_5m = now - timedelta(days=25)
            bench_sym = self.config.regime.benchmark_symbol
            try:
                bench_candles = self._filter_completed_candles(
                    self.broker.get_historical(bench_sym, "5minute", hist_from_5m, now), now
                )
            except Exception as e:
                logger.warning("Late-start benchmark backfill failed: %s", e)
                bench_candles = []

            backfilled_symbols: list[str] = []
            rebuilt_ors: list[str] = []
            for sym in symbols:
                try:
                    raw_c = self.broker.get_historical(sym, "5minute", hist_from_5m, now)
                    comp_c = self._filter_completed_candles(raw_c, now)
                    if comp_c:
                        backfilled_symbols.append(sym)
                    or_obj = build_opening_range(sym, comp_c, session_date=session_date)
                    self.opening_ranges[sym] = or_obj
                    if or_obj.is_complete:
                        rebuilt_ors.append(sym)

                    # 3. If position is open on this symbol, update HWM/LWM from today's candles since entry
                    pos = self.state_machine.positions.get(sym)
                    if pos and pos.quantity > 0:
                        raw_entry_ts = self._ensure_ist(pos.entry_timestamp) if pos.entry_timestamp else None
                        if raw_entry_ts and raw_entry_ts.date() == session_date and raw_entry_ts <= now:
                            entry_ts = raw_entry_ts
                        else:
                            entry_ts = now.replace(hour=9, minute=30, second=0, microsecond=0)
                        today_post_entry = [
                            c for c in comp_c
                            if self._ensure_ist(c.timestamp).date() == session_date and self._ensure_ist(c.timestamp) >= entry_ts
                        ]
                        if today_post_entry:
                            if pos.side == Side.BUY:
                                pos.best_favorable_price = max([pos.best_favorable_price, pos.average_entry] + [c.high for c in today_post_entry])
                            else:
                                cand_lows = [c.low for c in today_post_entry] + ([pos.best_favorable_price] if pos.best_favorable_price > 0 else []) + [pos.average_entry]
                                pos.best_favorable_price = min(x for x in cand_lows if x > 0)
                        try:
                            q_live = self.broker.get_quote(sym)
                            if q_live and getattr(q_live, "is_fresh", False) and q_live.last_price > 0:
                                if pos.side == Side.BUY:
                                    pos.best_favorable_price = max(pos.best_favorable_price, q_live.last_price)
                                else:
                                    pos.best_favorable_price = min(pos.best_favorable_price, q_live.last_price) if pos.best_favorable_price > 0 else q_live.last_price
                        except Exception:
                            pass
                        if pos.initial_stop_price > 0 and pos.average_entry > 0:
                            pos.r_value = abs(pos.average_entry - pos.initial_stop_price)
                            if pos.target_price <= 0:
                                inst_obj = self.broker.get_instrument_info(sym)
                                tick_sz = inst_obj.tick_size if inst_obj and inst_obj.tick_size > 0 else 0.05
                                pos.target_price = calculate_target_price(
                                    side=pos.side,
                                    entry_price=pos.average_entry,
                                    initial_stop=pos.initial_stop_price,
                                    target_r=self.config.stops.target_r,
                                    tick_size=tick_sz,
                                )
                except Exception as se:
                    logger.warning("Late-start backfill failed for %s: %s", sym, se)

            return {
                "timestamp_ist": now.isoformat(),
                "is_clean": audit.is_clean,
                "is_flat": audit.is_flat,
                "backfilled_symbols": backfilled_symbols,
                "rebuilt_opening_ranges": rebuilt_ors,
                "recovered_positions": [
                    p.symbol for p in self.state_machine.positions.values() if p.quantity > 0
                ],
                "benchmark_candles_count": len(bench_candles),
            }

    def evaluate_step(self, current_time: Optional[datetime] = None) -> list[StrategyDecision]:
        """Run one complete cycle of the 12-step deterministic pipeline (thread-safe)."""
        with self._execution_lock:
            decisions = self._evaluate_step_locked(current_time)
            self.last_decisions = decisions
            try:
                self.trade_logger.sync_from_broker(self.broker)
            except Exception as te:
                logger.debug("TradeLogger background sync: %s", te)
            return decisions

    def _evaluate_step_locked(self, current_time: Optional[datetime] = None) -> list[StrategyDecision]:
        """Internal execution pipeline under execution lock."""
        now = current_time or self.scheduler.now_ist()
        hist_from_5m = now - timedelta(days=25)
        hist_from_1m = now - timedelta(days=5)
        session_date = (now if now.tzinfo else now.replace(tzinfo=IST)).astimezone(IST).date()
        if self._current_session_date != session_date:
            self._current_session_date = session_date
            self.opening_ranges.clear()
            self._late_start_catchup_done.clear()
            self._last_entry_bar_timestamp = None
            self._flatten_completed = False
        if self._daily_loss_halt_date is not None and self._daily_loss_halt_date != session_date:
            self._daily_loss_halt_date = None
            if self.pause_reason in ("DAILY_LOSS_LIMIT_BREACHED", "DAILY_PNL_ACCOUNTING_FAILED"):
                self.is_paused = False
                self.pause_reason = ""
        gates: list[GateResult] = []

        # -------------------------------------------------------------
        # STEP 1: Session and Compliance Gate
        # -------------------------------------------------------------
        session_state = self.scheduler.get_session_state(now)

        if session_state in (SessionState.HOLIDAY, SessionState.WEEKEND):
            return [
                StrategyDecision(
                    decision=Decision.WAIT,
                    timestamp_ist=now,
                    reason_codes=["NON_TRADING_DAY"],
                    explanation=f"Market closed: {session_state.name}",
                )
            ]

        # -------------------------------------------------------------
        # STEP 12 (Priority): End of Day / Flatten Window (Tracked One-Shot)
        # -------------------------------------------------------------
        if self.scheduler.should_flatten(now):
            if not self._flatten_completed:
                success, msg = self.reconciler.execute_flatten_sequence()
                decision_type = Decision.EXIT
                if success:
                    self._flatten_completed = True
                else:
                    self.journal.log_warning("EMERGENCY", "FLATTEN_FAILED", msg)

                return [
                    StrategyDecision(
                        decision=decision_type,
                        timestamp_ist=now,
                        reason_codes=["FLATTEN_WINDOW_ACTIVE"],
                        explanation=f"Flattening active: {msg}",
                        broker_action="EXIT_ALL_MIS",
                    )
                ]
            else:
                return [
                    StrategyDecision(
                        decision=Decision.WAIT,
                        timestamp_ist=now,
                        reason_codes=["FLATTEN_COMPLETED"],
                        explanation="Account confirmed flat for the session; no further actions.",
                    )
                ]

        # -------------------------------------------------------------
        # STEP 2: Connectivity, Quote Freshness, and Macro News Gate
        # -------------------------------------------------------------
        now_utc = (now if now.tzinfo else now.replace(tzinfo=IST)).astimezone(timezone.utc)
        try:
            macro_state = self.macro_provider.get_macro_state(now_utc)
        except Exception as me:
            logger.warning("Error reading macro state, gracefully degrading to neutral: %s", me)
            from apex3.news.models import MacroRegimeState
            macro_state = MacroRegimeState(
                in_blackout_window=False,
                minutes_to_next_high_event=None,
                active_event_title="",
                macro_score=0.0,
                catalyst_aligned_buy=False,
                catalyst_aligned_sell=False,
                last_updated_utc=now_utc,
            )
        self.latest_macro_state = macro_state

        if not self.broker.is_session_valid():
            return [
                StrategyDecision(
                    decision=Decision.HALT,
                    timestamp_ist=now,
                    reason_codes=["BROKER_SESSION_INVALID"],
                    explanation="Broker session or token is invalid or expired",
                )
            ]

        # -------------------------------------------------------------
        # STEP 3: Account & Order Reconciliation Gate
        # -------------------------------------------------------------
        raw_symbols = self.get_active_watchlist()
        audit = self.reconciler.audit_startup_state(raw_symbols)
        if not audit.is_clean and not audit.unknown_symbols:
            # Broker fetch failure during reconciliation: wait for next tick rather than locking engine
            return [
                StrategyDecision(
                    decision=Decision.WAIT,
                    timestamp_ist=now,
                    reason_codes=["RECONCILIATION_FETCH_WAIT"],
                    explanation=audit.message,
                )
            ]

        # Manual / foreign positions on Zerodha do NOT halt the algo; instead, the manual symbols
        # are safely excluded from algo scanning so manual user trading never interferes with algo operation.
        manual_symbols = set(audit.unknown_symbols) if audit.unknown_symbols else set()
        symbols = [s for s in raw_symbols if s not in manual_symbols]
        if manual_symbols:
            logger.info("Manual/foreign positions detected on Zerodha in %s; excluding from strategy radar", manual_symbols)
        if not symbols and raw_symbols:
            return [
                StrategyDecision(
                    decision=Decision.WAIT,
                    timestamp_ist=now,
                    reason_codes=["MANUAL_POSITIONS_ACTIVE"],
                    explanation=f"All monitored symbols currently traded manually by operator: {sorted(manual_symbols)}",
                )
            ]

        # Operator Pause check: if paused and no open positions, wait immediately
        if self.is_paused:
            open_count = sum(1 for p in self.state_machine.positions.values() if p.quantity > 0) + len(
                [p for p in self.state_machine.positions.values() if p.phase == PositionPhase.ENTERING]
            )
            if open_count == 0:
                return [
                    StrategyDecision(
                        decision=Decision.WAIT,
                        timestamp_ist=now,
                        reason_codes=["ENGINE_PAUSED"],
                        explanation=f"Strategy engine is paused: {self.pause_reason or 'Operator pause'}",
                    )
                ]

        # -------------------------------------------------------------
        # STEP 11 (Priority): Manage Open Positions
        # -------------------------------------------------------------
        decisions: list[StrategyDecision] = []

        # Step 11.A: Reconcile pending entry orders (PositionPhase.ENTERING)
        entering_managed = [
            p for p in self.state_machine.positions.values()
            if p.phase == PositionPhase.ENTERING and p.entry_order_id
        ]
        for ent_pos in entering_managed:
            try:
                entry_state = self.broker.get_order_status(ent_pos.entry_order_id)
                if not entry_state:
                    continue

                if entry_state.status == OrderStatus.COMPLETE or entry_state.filled_quantity > 0:
                    current_fill_qty = int(getattr(entry_state, "filled_quantity", 0) or 0)
                    if current_fill_qty <= 0:
                        if entry_state.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                            ent_pos.phase = PositionPhase.FLAT
                            ent_pos.quantity = 0
                        continue

                    ent_pos.quantity = current_fill_qty
                    if entry_state.average_price > 0:
                        ent_pos.average_entry = entry_state.average_price
                    ent_pos.best_favorable_price = ent_pos.average_entry
                    ent_pos.r_value = abs(ent_pos.average_entry - ent_pos.initial_stop_price)
                    inst = self.broker.get_instrument_info(ent_pos.symbol)
                    raw_tick = getattr(inst, "tick_size", 0.05) if inst else 0.05
                    tick = float(raw_tick) if isinstance(raw_tick, (int, float)) and not isinstance(raw_tick, bool) and raw_tick > 0 else 0.05
                    ent_pos.target_price = calculate_target_price(
                        side=ent_pos.side,
                        entry_price=ent_pos.average_entry,
                        initial_stop=ent_pos.initial_stop_price,
                        target_r=self.config.stops.target_r,
                        tick_size=tick,
                    )

                    exit_side = Side.SELL if ent_pos.side == Side.BUY else Side.BUY
                    trigger, limit = calculate_stop_limit_prices(
                        ent_pos.side,
                        ent_pos.initial_stop_price,
                        self.config.stops,
                        tick,
                        lower_circuit=inst.lower_circuit if inst else 0.0,
                        upper_circuit=inst.upper_circuit if inst else 0.0,
                    )

                    # Place protective stop immediately for filled quantity
                    if not ent_pos.protective_order_id:
                        sl_req = OrderRequest(
                            symbol=ent_pos.symbol,
                            exchange=getattr(self.config, "exchange", "MCX"),
                            side=exit_side,
                            quantity=ent_pos.quantity,
                            product=Product.MIS,
                            order_type=OrderType.SL,
                            price=limit,
                            trigger_price=trigger,
                            tag=make_broker_tag(self.session_tag, ent_pos.symbol, "SL"),
                            strategy_id=self.session_tag,
                        )
                        try:
                            sl_order_id = self.broker.place_order(sl_req)
                            ent_pos.protective_order_id = sl_order_id
                        except Exception as sl_err:
                            logger.error("Failed to place protective stop for pending fill %s: %s", ent_pos.symbol, sl_err)
                    else:
                        # Stop order exists: check if quantity needs updating due to fill increase
                        try:
                            sl_check = self.broker.get_order_status(ent_pos.protective_order_id)
                            sl_qty = getattr(sl_check, "quantity", None) or getattr(getattr(sl_check, "request", None), "quantity", 0) or getattr(sl_check, "pending_quantity", 0)
                            if sl_check and sl_check.status == OrderStatus.TRIGGER_PENDING and sl_qty < ent_pos.quantity:
                                logger.info("Modifying protective stop %s for %s from %d to %d shares", ent_pos.protective_order_id, ent_pos.symbol, sl_qty, ent_pos.quantity)
                                self.broker.modify_order(ent_pos.protective_order_id, quantity=ent_pos.quantity)
                        except Exception as me:
                            logger.error("Error updating stop %s for fill increase: %s", ent_pos.protective_order_id, me)

                    # On partial fill, cancel the unfilled remainder and wait for terminal confirmation
                    if entry_state.status != OrderStatus.COMPLETE:
                        try:
                            self.broker.cancel_order(ent_pos.entry_order_id)
                        except Exception as cancel_err:
                            logger.warning("Failed to cancel remaining entry for %s: %s", ent_pos.symbol, cancel_err)
                        term_state = self._confirm_order_terminal(ent_pos.entry_order_id, timeout_sec=0.5)
                        if term_state:
                            term_filled_qty = int(getattr(term_state, "filled_quantity", 0) or 0)
                            if term_filled_qty > ent_pos.quantity:
                                # Late fill occurred while cancelling!
                                logger.info("Late fill detected during cancel for %s: qty increased from %d to %d", ent_pos.symbol, ent_pos.quantity, term_filled_qty)
                                ent_pos.quantity = term_filled_qty
                                if term_state.average_price > 0:
                                    ent_pos.average_entry = term_state.average_price
                                if ent_pos.protective_order_id:
                                    try:
                                        self.broker.modify_order(ent_pos.protective_order_id, quantity=ent_pos.quantity)
                                    except Exception as e:
                                        logger.error("Failed to modify stop after late fill for %s: %s", ent_pos.symbol, e)

                            if term_state.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.COMPLETE):
                                entry_state = term_state
                            else:
                                logger.warning(
                                    "Entry order %s remainder cancel unconfirmed (%s); proceeding to verify protection",
                                    ent_pos.entry_order_id,
                                    getattr(term_state, "status", "UNKNOWN"),
                                )
                                entry_state = term_state
                        else:
                            logger.warning("Entry order %s remainder cancel timed out; proceeding to verify protection", ent_pos.entry_order_id)

                    confirmed_fill_qty = int(getattr(entry_state, "filled_quantity", 0) or 0)
                    if confirmed_fill_qty <= 0:
                        ent_pos.phase = PositionPhase.FLAT
                        ent_pos.quantity = 0
                        continue

                    # Reconcile confirmed fill quantity against live broker positions (strictly verify, no max() absorption)
                    actual_qty = confirmed_fill_qty
                    try:
                        fresh_pos = [p for p in self.broker.get_positions() if p.symbol == ent_pos.symbol and getattr(p, "product", Product.MIS) == Product.MIS]
                        if fresh_pos:
                            broker_net_qty = abs(fresh_pos[0].quantity)
                            if broker_net_qty > 0 and broker_net_qty != confirmed_fill_qty:
                                logger.critical(
                                    "EMERGENCY: Broker MIS position (%d) != confirmed fill quantity (%d) for %s! Halting for manual reconciliation.",
                                    broker_net_qty, confirmed_fill_qty, ent_pos.symbol,
                                )
                                self.journal.log_warning(
                                    severity="EMERGENCY",
                                    code="POSITION_QTY_DISAGREEMENT",
                                    message=f"Broker MIS position {broker_net_qty} does not match fill {confirmed_fill_qty} for {ent_pos.symbol}",
                                    data={"broker_qty": broker_net_qty, "fill_qty": confirmed_fill_qty},
                                )
                                self.is_paused = True
                                self.pause_reason = f"QTY_MISMATCH_{ent_pos.symbol}"
                                ent_pos.phase = PositionPhase.ERROR
                                continue
                    except Exception as pe:
                        logger.warning("Error fetching broker positions for %s fill reconciliation: %s", ent_pos.symbol, pe)

                    ent_pos.quantity = actual_qty
                    if entry_state.average_price > 0:
                        ent_pos.average_entry = entry_state.average_price
                    ent_pos.best_favorable_price = ent_pos.average_entry
                    ent_pos.r_value = abs(ent_pos.average_entry - ent_pos.initial_stop_price)
                    ent_pos.target_price = calculate_target_price(
                        side=ent_pos.side,
                        entry_price=ent_pos.average_entry,
                        initial_stop=ent_pos.initial_stop_price,
                        target_r=self.config.stops.target_r,
                        tick_size=tick,
                    )

                    # Verify protective stop is active and matching confirmed fill quantity
                    sl_confirmed = False
                    if ent_pos.protective_order_id:
                        try:
                            sl_confirmed = self._confirm_protective_sl(
                                ent_pos.protective_order_id,
                                ent_pos.quantity,
                                timeout_sec=2.0,
                            )
                            if sl_confirmed:
                                ent_pos.phase = PositionPhase.OPEN_PROTECTED
                                ent_pos.entry_timestamp = ent_pos.entry_timestamp or now
                        except Exception as ce:
                            logger.error("Error verifying final stop status for %s: %s", ent_pos.symbol, ce)

                    if sl_confirmed:
                        self.journal.log_event(
                            event_type="ENTRY_FILLED_PROTECTED",
                            symbol=ent_pos.symbol,
                            data={
                                "quantity": ent_pos.quantity,
                                "average_entry": ent_pos.average_entry,
                                "sl_order_id": ent_pos.protective_order_id,
                            },
                            explanation=f"Entry filled for {ent_pos.symbol} (qty={ent_pos.quantity}); protective SL placed & confirmed",
                        )
                        try:
                            self.trade_logger.record_entry(
                                symbol=ent_pos.symbol,
                                side=ent_pos.side,
                                quantity=ent_pos.quantity,
                                price=ent_pos.average_entry,
                                order_id=ent_pos.entry_order_id,
                                trade_id=ent_pos.entry_order_id,
                                timestamp=now,
                                initial_stop=ent_pos.initial_stop_price,
                                target_price=ent_pos.target_price,
                                risk_rupees=getattr(ent_pos, "r_value", 0.0) * ent_pos.quantity,
                                tag=getattr(ent_pos, "strategy_tag", self.session_tag),
                            )
                        except Exception as log_err:
                            logger.error("Error logging entry to trade_logger: %s", log_err)
                    else:
                        self._last_entry_bar_timestamp = None
                        logger.critical("EMERGENCY: Protective stop unconfirmed for filled entry in %s. Liquidating immediately.", ent_pos.symbol)
                        self.journal.log_warning(
                            severity="EMERGENCY",
                            code="PROTECTIVE_STOP_UNCONFIRMED",
                            message=f"Protective stop for filled entry in {ent_pos.symbol} failed. Liquidating immediately.",
                        )
                        try:
                            emer_exit = self._create_exit_order_request(
                                symbol=ent_pos.symbol,
                                side=exit_side,
                                quantity=ent_pos.quantity,
                                tag=make_broker_tag(self.session_tag, ent_pos.symbol, "FLT"),
                                current_price=ent_pos.average_entry,
                            )
                            emer_id = self.broker.place_order(emer_exit)
                            ent_pos.exit_order_id = emer_id
                            ent_pos.phase = PositionPhase.EXITING
                        except Exception as em_err:
                            logger.critical("FATAL: Failed to execute emergency liquidation for %s: %s", ent_pos.symbol, em_err)
                            ent_pos.phase = PositionPhase.ERROR
                        self.is_paused = True
                        self.pause_reason = f"PROTECTIVE_STOP_FAILED_{ent_pos.symbol}"
                elif entry_state.status in (OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.ERROR):
                    ent_pos.phase = PositionPhase.FLAT
                    ent_pos.quantity = 0
                    self._last_entry_bar_timestamp = None
                else:
                    timeout_sec = getattr(getattr(self.config, "entry", None), "unfilled_timeout_sec", 30)
                    if ent_pos.entry_timestamp and (now - ent_pos.entry_timestamp).total_seconds() > timeout_sec:
                        try:
                            self.broker.cancel_order(ent_pos.entry_order_id)
                        except Exception as ce:
                            logger.warning("Error cancelling timed-out entry %s: %s", ent_pos.symbol, ce)
                        term_refreshed = self._confirm_order_terminal(ent_pos.entry_order_id)
                        if term_refreshed and term_refreshed.filled_quantity > 0:
                            ent_pos.quantity = term_refreshed.filled_quantity
                            if term_refreshed.average_price > 0:
                                ent_pos.average_entry = term_refreshed.average_price
                            ent_pos.best_favorable_price = ent_pos.average_entry
                            ent_pos.phase = PositionPhase.OPEN_UNPROTECTED
                            ent_pos.r_value = abs(ent_pos.average_entry - ent_pos.initial_stop_price)

                            inst = self.broker.get_instrument_info(ent_pos.symbol)
                            tick = inst.tick_size if inst and inst.tick_size > 0 else 0.05
                            ent_pos.target_price = calculate_target_price(
                                side=ent_pos.side,
                                entry_price=ent_pos.average_entry,
                                initial_stop=ent_pos.initial_stop_price,
                                target_r=self.config.stops.target_r,
                                tick_size=tick,
                            )

                            exit_side = Side.SELL if ent_pos.side == Side.BUY else Side.BUY
                            trigger, limit = calculate_stop_limit_prices(
                                ent_pos.side,
                                ent_pos.initial_stop_price,
                                self.config.stops,
                                tick,
                                lower_circuit=inst.lower_circuit if inst else 0.0,
                                upper_circuit=inst.upper_circuit if inst else 0.0,
                            )
                            sl_req = OrderRequest(
                                symbol=ent_pos.symbol,
                                exchange=getattr(self.config, "exchange", "MCX"),
                                side=exit_side,
                                quantity=ent_pos.quantity,
                                product=Product.MIS,
                                order_type=OrderType.SL,
                                price=limit,
                                trigger_price=trigger,
                                tag=make_broker_tag(self.session_tag, ent_pos.symbol, "SL"),
                                strategy_id=self.session_tag,
                            )
                            sl_confirmed = False
                            try:
                                sl_order_id = self.broker.place_order(sl_req)
                                ent_pos.protective_order_id = sl_order_id
                                sl_confirmed = self._confirm_protective_sl(sl_order_id, ent_pos.quantity, timeout_sec=2.0)
                                if sl_confirmed:
                                    ent_pos.phase = PositionPhase.OPEN_PROTECTED
                            except Exception as sl_err:
                                logger.error("Failed to place stop on timeout fill %s: %s", ent_pos.symbol, sl_err)

                            if not sl_confirmed:
                                self._last_entry_bar_timestamp = None
                                logger.critical("EMERGENCY: Protective stop unconfirmed for timeout fill in %s. Liquidating immediately.", ent_pos.symbol)
                                try:
                                    emer_exit = self._create_exit_order_request(
                                        symbol=ent_pos.symbol,
                                        side=exit_side,
                                        quantity=ent_pos.quantity,
                                        tag=make_broker_tag(self.session_tag, ent_pos.symbol, "FLT"),
                                        current_price=ent_pos.average_entry,
                                    )
                                    emer_id = self.broker.place_order(emer_exit)
                                    ent_pos.exit_order_id = emer_id
                                    ent_pos.phase = PositionPhase.EXITING
                                except Exception as em_err:
                                    logger.critical("FATAL: Failed emergency liquidation for %s: %s", ent_pos.symbol, em_err)
                                    ent_pos.phase = PositionPhase.ERROR
                                self.is_paused = True
                                self.pause_reason = f"PROTECTIVE_STOP_FAILED_{ent_pos.symbol}"
                        elif term_refreshed and term_refreshed.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                            ent_pos.phase = PositionPhase.FLAT
                            ent_pos.quantity = 0
                            self._last_entry_bar_timestamp = None
                        else:
                            # Cancellation unconfirmed / non-terminal: do not mark FLAT! Keep ENTERING.
                            logger.warning(
                                "Entry cancellation unconfirmed for %s (order %s, status=%s); retaining ENTERING phase to prevent unmanaged exposure.",
                                ent_pos.symbol,
                                ent_pos.entry_order_id,
                                getattr(term_refreshed, "status", "UNKNOWN"),
                            )
            except Exception as exc:
                logger.error("Error reconciling entering position %s: %s", ent_pos.symbol, exc)

        open_managed = [p for p in self.state_machine.positions.values() if p.quantity > 0]

        for pos in open_managed:
            # If position is already in EXITING phase, poll existing exit order and reconcile
            if pos.phase == PositionPhase.EXITING:
                if pos.exit_order_id:
                    try:
                        exit_state = self.broker.get_order_status(pos.exit_order_id)
                        if exit_state:
                            if exit_state.status == OrderStatus.COMPLETE:
                                closed_qty = pos.quantity or (exit_state.filled_quantity if exit_state else 0) or getattr(pos, "requested_quantity", 0)
                                pos.phase = PositionPhase.CLOSED
                                pos.quantity = 0
                                fill_px = exit_state.average_price or pos.average_entry
                                self.journal.log_event(
                                    event_type="EXIT_ORDER_COMPLETED",
                                    symbol=pos.symbol,
                                    data={
                                        "exit_order_id": pos.exit_order_id,
                                        "fill_price": fill_px,
                                    },
                                    explanation=f"Exit order {pos.exit_order_id} filled for {pos.symbol}",
                                 )
                                try:
                                    self.trade_logger.record_exit(
                                        symbol=pos.symbol,
                                        side=Side.SELL if pos.side == Side.BUY else Side.BUY,
                                        quantity=closed_qty,
                                        exit_price=fill_px,
                                        entry_price=pos.average_entry,
                                        order_id=pos.exit_order_id,
                                        trade_id=pos.exit_order_id,
                                        exit_reason="EXIT_ORDER_FILLED",
                                        timestamp=now,
                                        entry_time=getattr(pos, "entry_time", None) or getattr(pos, "entry_timestamp", None),
                                        initial_stop=pos.initial_stop_price,
                                    )
                                except Exception as log_err:
                                    logger.error("Error logging exit to trade_logger: %s", log_err)
                                continue
                            elif exit_state.status == OrderStatus.UNKNOWN:
                                logger.critical("Exit order %s for %s in UNKNOWN state. Flagging reconciliation alert.", pos.exit_order_id, pos.symbol)
                                self.journal.log_warning(
                                    severity="EMERGENCY",
                                    code="EXIT_ORDER_STATUS_UNKNOWN",
                                    message=f"Exit order {pos.exit_order_id} for {pos.symbol} in UNKNOWN state.",
                                    data={"symbol": pos.symbol, "exit_order_id": pos.exit_order_id},
                                )
                                # Do NOT dispatch a conflicting second exit order while broker state is unknown!
                                continue
                            elif exit_state.status in (OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.ERROR):
                                logger.warning(
                                    "Exit order %s for %s was %s; clearing exit_order_id for retry",
                                    pos.exit_order_id,
                                    pos.symbol,
                                    exit_state.status.value,
                                )
                                if getattr(exit_state, "filled_quantity", 0) > 0:
                                    pos.quantity = max(0, pos.quantity - exit_state.filled_quantity)
                                pos.exit_order_id = ""
                                if pos.quantity <= 0:
                                    pos.phase = PositionPhase.CLOSED
                                    pos.quantity = 0
                                    continue
                            elif exit_state.pending_quantity > 0:
                                pos.quantity = exit_state.pending_quantity
                                pos.phase = PositionPhase.EXITING
                            elif hasattr(exit_state, "request") and exit_state.request and exit_state.request.quantity > 0:
                                rem = max(0, exit_state.request.quantity - exit_state.filled_quantity)
                                pos.quantity = rem
                                if pos.quantity == 0:
                                    pos.phase = PositionPhase.CLOSED
                                    continue
                                else:
                                    pos.phase = PositionPhase.EXITING
                        else:
                            logger.warning("Could not fetch exit order status for %s (order %s); deferring retry to next cycle", pos.symbol, pos.exit_order_id)
                            continue
                    except Exception as e_poll:
                        logger.error("Error polling exit order %s for %s: %s", pos.exit_order_id, pos.symbol, e_poll)

                if not pos.exit_order_id and pos.quantity > 0:
                    exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
                    try:
                        retry_exit = self._create_exit_order_request(
                            symbol=pos.symbol,
                            side=exit_side,
                            quantity=pos.quantity,
                            tag=make_broker_tag(self.session_tag, pos.symbol, "EXT"),
                            current_price=getattr(pos, "average_entry", 0.0),
                        )
                        retry_id = self.broker.place_order(retry_exit)
                        pos.exit_order_id = retry_id
                        self.journal.log_event(
                            event_type="EXIT_ORDER_RETRY",
                            symbol=pos.symbol,
                            side=exit_side.value,
                            data={"quantity": pos.quantity, "exit_order_id": retry_id},
                            explanation=f"Re-dispatched exit order {retry_id} for {pos.symbol} after exit failure",
                        )
                    except Exception as retry_err:
                        logger.critical("Failed to dispatch retry exit order for %s: %s", pos.symbol, retry_err)

                # Do not submit a second exit order while an exit order is pending or in-flight
                continue

            # Step 11.0: Reconcile broker stop-loss order status before target / manage checks
            if pos.protective_order_id:
                try:
                    sl_order = self.broker.get_order_status(pos.protective_order_id)
                    if sl_order and sl_order.status == OrderStatus.COMPLETE:
                        fill_px = sl_order.average_price or pos.current_stop_price
                        self.journal.log_event(
                            event_type="STOP_LOSS_HIT",
                            symbol=pos.symbol,
                            side=pos.side.value,
                            data={
                                "stop_price": pos.current_stop_price,
                                "sl_order_id": pos.protective_order_id,
                                "fill_price": fill_px,
                            },
                        )
                        try:
                            self.trade_logger.record_exit(
                                symbol=pos.symbol,
                                side=Side.SELL if pos.side == Side.BUY else Side.BUY,
                                quantity=pos.quantity,
                                exit_price=fill_px,
                                entry_price=pos.average_entry,
                                order_id=pos.protective_order_id,
                                trade_id=pos.protective_order_id,
                                exit_reason="STOP_LOSS_HIT",
                                timestamp=now,
                                entry_time=getattr(pos, "entry_time", None) or getattr(pos, "entry_timestamp", None),
                                initial_stop=pos.initial_stop_price,
                            )
                        except Exception as log_err:
                            logger.error("Error logging SL exit to trade_logger: %s", log_err)
                        pos.phase = PositionPhase.CLOSED
                        pos.quantity = 0
                        decisions.append(
                            StrategyDecision(
                                decision=Decision.EXIT,
                                symbol=pos.symbol,
                                side=pos.side,
                                timestamp_ist=now,
                                reason_codes=["STOP_LOSS_FILLED_AT_BROKER"],
                                explanation=f"Protective SL order {pos.protective_order_id} filled at broker",
                                broker_action="STOP_LOSS_FILLED",
                            )
                        )
                        continue
                    elif sl_order and sl_order.status == OrderStatus.OPEN:
                        # Order triggered but resting as unfilled limit order on the book
                        q_now = self.broker.get_quote(pos.symbol)
                        if not q_now or not getattr(q_now, "is_fresh", False) or not q_now.timestamp:
                            logger.warning("Gap exit check for %s: quote is stale or missing; skipping gap exit evaluation", pos.symbol)
                            continue
                        px = q_now.last_price if q_now.last_price > 0 else 0.0
                        limit_px = (
                            sl_order.request.price
                            if (getattr(sl_order, "request", None) and sl_order.request.price > 0)
                            else (getattr(sl_order, "price", 0.0) or pos.current_stop_price)
                        )
                        is_gapped = False
                        if pos.side == Side.BUY and px > 0 and px < limit_px * 0.999:
                            is_gapped = True
                        elif pos.side == Side.SELL and px > 0 and px > limit_px * 1.001:
                            is_gapped = True

                        if is_gapped:
                            logger.critical(
                                "EMERGENCY: SL order %s triggered but unfilled and market gapped through limit (%s). Executing market exit!",
                                pos.protective_order_id,
                                px,
                            )
                            sl_st = None
                            try:
                                self.broker.cancel_order(pos.protective_order_id)
                            except Exception as c_err:
                                logger.warning("Error cancelling stop during gap exit %s: %s", pos.symbol, c_err)
                            sl_st = self._confirm_order_terminal(pos.protective_order_id, timeout_sec=2.0)

                            if sl_st and sl_st.status == OrderStatus.COMPLETE:
                                self.journal.log_event(
                                    event_type="STOP_LOSS_HIT",
                                    symbol=pos.symbol,
                                    side=pos.side.value,
                                    data={
                                        "stop_price": pos.current_stop_price,
                                        "sl_order_id": pos.protective_order_id,
                                        "fill_price": sl_st.average_price or pos.current_stop_price,
                                    },
                                )
                                pos.phase = PositionPhase.CLOSED
                                pos.quantity = 0
                                decisions.append(
                                    StrategyDecision(
                                        decision=Decision.EXIT,
                                        symbol=pos.symbol,
                                        side=pos.side,
                                        timestamp_ist=now,
                                        reason_codes=["STOP_LOSS_FILLED_AT_BROKER"],
                                        explanation=f"Protective SL order {pos.protective_order_id} filled at broker during gap cancel",
                                        broker_action="STOP_LOSS_FILLED",
                                    )
                                )
                                continue

                            if not sl_st or sl_st.status not in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                                logger.warning(
                                    "Protective SL order %s for %s cancellation unconfirmed (%s); retrying gap exit next tick",
                                    pos.protective_order_id,
                                    pos.symbol,
                                    getattr(sl_st, "status", "UNKNOWN"),
                                )
                                continue

                            sl_filled = getattr(sl_st, "filled_quantity", 0) if sl_st else 0
                            rem_qty = max(0, pos.quantity - int(sl_filled if isinstance(sl_filled, (int, float)) else 0))
                            if rem_qty <= 0:
                                pos.phase = PositionPhase.CLOSED
                                pos.quantity = 0
                                continue

                            exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
                            emer_exit = self._create_exit_order_request(
                                symbol=pos.symbol,
                                side=exit_side,
                                quantity=rem_qty,
                                tag=make_broker_tag(self.session_tag, pos.symbol, "EMG"),
                                current_price=px,
                            )
                            emer_id = self.broker.place_order(emer_exit)
                            pos.exit_order_id = emer_id
                            pos.phase = PositionPhase.EXITING
                            decisions.append(
                                StrategyDecision(
                                    decision=Decision.EXIT,
                                    symbol=pos.symbol,
                                    side=pos.side,
                                    timestamp_ist=now,
                                    reason_codes=["STOP_TRIGGERED_UNFILLED_GAP_EXIT"],
                                    explanation=f"Protective SL limit order {pos.protective_order_id} triggered but unfilled as price gapped to {px}. Emergency exit dispatched for {rem_qty} shares.",
                                    broker_action="EMERGENCY_GAP_EXIT",
                                )
                            )
                            continue
                    elif sl_order and sl_order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                        sl_filled = getattr(sl_order, "filled_quantity", 0)
                        if isinstance(sl_filled, (int, float)) and sl_filled > 0:
                            pos.quantity = max(0, pos.quantity - int(sl_filled))
                        if pos.quantity <= 0:
                            pos.phase = PositionPhase.CLOSED
                            pos.quantity = 0
                            continue
                        logger.critical(
                            "EMERGENCY: Protective SL order %s for %s is %s! Marking OPEN_UNPROTECTED for immediate liquidation.",
                            pos.protective_order_id,
                            pos.symbol,
                            sl_order.status.value,
                        )
                        pos.phase = PositionPhase.OPEN_UNPROTECTED
                    elif sl_order is None or sl_order.status in (OrderStatus.UNKNOWN, OrderStatus.ERROR):
                        logger.critical(
                            "Protective SL order %s for %s status is %s. Flagging reconciliation alert.",
                            pos.protective_order_id,
                            pos.symbol,
                            getattr(sl_order, "status", "UNKNOWN"),
                        )
                        self.journal.log_warning(
                            severity="EMERGENCY",
                            code="PROTECTIVE_STOP_STATUS_UNKNOWN",
                            message=f"Protective SL order {pos.protective_order_id} for {pos.symbol} in UNKNOWN state.",
                            data={"symbol": pos.symbol, "sl_order_id": pos.protective_order_id},
                        )
                        continue
                except Exception as exc:
                    self.journal.log_warning(
                        severity="WARNING",
                        code="SL_STATUS_CHECK_FAILED",
                        message=f"Could not check SL order {pos.protective_order_id} for {pos.symbol}: {exc}",
                    )

            # Verify confirmed protective stop coverage
            if pos.phase not in (PositionPhase.OPEN_PROTECTED, PositionPhase.EXITING, PositionPhase.CLOSED) and pos.quantity > 0:
                logger.critical("EMERGENCY: Open position in %s has no verified protective stop (phase=%s)! Liquidating immediately.", pos.symbol, pos.phase)
                try:
                    exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
                    flat_req = self._create_exit_order_request(
                        symbol=pos.symbol,
                        side=exit_side,
                        quantity=pos.quantity,
                        tag=make_broker_tag(self.session_tag, pos.symbol, "EMG"),
                        current_price=getattr(pos, "average_entry", 0.0),
                    )
                    flat_id = self.broker.place_order(flat_req)
                    pos.exit_order_id = flat_id
                    pos.phase = PositionPhase.EXITING
                except Exception as unprot_err:
                    logger.critical("FATAL: Failed to liquidate unprotected position in %s: %s", pos.symbol, unprot_err)
                continue

            # Manage existing position stop or take profit
            try:
                quote = self.broker.get_quote(pos.symbol)
                if not quote or not quote.is_fresh or quote.last_price <= 0:
                    logger.warning("Skipping stop/target management for %s: quote is missing or stale", pos.symbol)
                    continue
                current_price = quote.last_price
                inst_info = self.broker.get_instrument_info(pos.symbol)

                # 0. Pre-News Protective Exit Check
                unbuffered_min = 3.0
                lock_be_threshold = 0.75
                if hasattr(self.config, "news") and self.config.news:
                    unbuffered_min = float(getattr(self.config.news, "pre_event_flatten_unbuffered_min", 3.0))
                    lock_be_threshold = float(getattr(self.config.news, "pre_event_lock_be_threshold_r", 0.75))

                mins_to_evt = getattr(macro_state, "minutes_to_next_high_event", None)
                if mins_to_evt is not None and 0.0 < mins_to_evt <= unbuffered_min:
                    unit_r = pos.r_value if pos.r_value > 0 else max(1.0, abs(pos.average_entry - pos.initial_stop_price))
                    fav = (current_price - pos.average_entry) if pos.side == Side.BUY else (pos.average_entry - current_price)
                    open_r = fav / unit_r if unit_r > 0 else 0.0

                    is_locked_be = getattr(pos, "fee_lock_applied", False) or (
                        pos.current_stop_price >= pos.average_entry if pos.side == Side.BUY else (pos.current_stop_price <= pos.average_entry and pos.current_stop_price > 0)
                    )

                    if open_r < lock_be_threshold and not is_locked_be and pos.phase == PositionPhase.OPEN_PROTECTED:
                        logger.warning(
                            "PRE_NEWS_PROTECTIVE_EXIT triggered for %s before %s: %.1fm to event, open_r=%.2fR < %.2fR",
                            pos.symbol,
                            getattr(macro_state, "active_event_title", "macro event"),
                            mins_to_evt,
                            open_r,
                            lock_be_threshold,
                        )
                        if pos.protective_order_id:
                            try:
                                self.broker.cancel_order(pos.protective_order_id)
                            except Exception as c_err:
                                logger.warning("Error cancelling stop before pre-news exit for %s: %s", pos.symbol, c_err)
                            self._confirm_order_terminal(pos.protective_order_id, timeout_sec=2.0)

                        nws_tag = make_broker_tag(self.session_tag, pos.symbol, "NWS")
                        exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
                        exit_req = self._create_exit_order_request(
                            symbol=pos.symbol,
                            side=exit_side,
                            quantity=pos.quantity,
                            tag=nws_tag,
                            current_price=current_price,
                        )
                        exit_order_id = self.broker.place_order(exit_req)
                        pos.exit_order_id = exit_order_id
                        pos.phase = PositionPhase.EXITING

                        exit_state = self.broker.get_order_status(exit_order_id)
                        if exit_state and exit_state.status == OrderStatus.COMPLETE:
                            pos.phase = PositionPhase.CLOSED
                            pos.quantity = 0
                            fill_px = exit_state.average_price or current_price
                            try:
                                self.trade_logger.record_exit(
                                    symbol=pos.symbol,
                                    side=exit_side,
                                    quantity=exit_state.filled_quantity or pos.quantity,
                                    exit_price=fill_px,
                                    entry_price=pos.average_entry,
                                    order_id=exit_order_id,
                                    trade_id=exit_order_id,
                                    exit_reason="PRE_NEWS_PROTECTIVE_EXIT",
                                    timestamp=now,
                                    entry_time=getattr(pos, "entry_time", None) or getattr(pos, "entry_timestamp", None),
                                    initial_stop=pos.initial_stop_price,
                                )
                            except Exception as log_err:
                                logger.error("Error logging pre-news exit to trade_logger: %s", log_err)
                        elif exit_state and exit_state.pending_quantity > 0:
                            pos.quantity = exit_state.pending_quantity
                            pos.phase = PositionPhase.EXITING

                        self.journal.log_event(
                            event_type="PRE_NEWS_PROTECTIVE_EXIT",
                            symbol=pos.symbol,
                            side=exit_side.value,
                            data={
                                "exit_order_id": exit_order_id,
                                "tag": nws_tag,
                                "minutes_to_event": mins_to_evt,
                                "open_r": open_r,
                            },
                            explanation=f"Pre-news protective exit for {pos.symbol} before event with tag {nws_tag}",
                        )

                        decisions.append(
                            StrategyDecision(
                                decision=Decision.EXIT,
                                symbol=pos.symbol,
                                side=pos.side,
                                timestamp_ist=now,
                                reason_codes=["PRE_NEWS_PROTECTIVE_EXIT"],
                                explanation=f"Pre-news protective exit before event ({mins_to_evt:.1f}m to event, open_r={open_r:.2f}R < {lock_be_threshold:.2f}R)",
                                broker_action=f"EXIT_NEWS_{nws_tag}",
                            )
                        )
                        continue

                # 1. Fixed Target check (Option A: 2.0R Target)
                if self.config.stops.use_fixed_target and pos.target_price > 0:
                    target_reached = False
                    if pos.side == Side.BUY and current_price >= pos.target_price:
                        target_reached = True
                    elif pos.side == Side.SELL and current_price <= pos.target_price:
                        target_reached = True

                    if target_reached:
                        exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
                        sl_filled = 0
                        sl_st = None
                        if pos.protective_order_id:
                            try:
                                self.broker.cancel_order(pos.protective_order_id)
                            except Exception as c_err:
                                logger.warning("Error cancelling stop before target exit %s: %s", pos.symbol, c_err)
                            sl_st = self._confirm_order_terminal(pos.protective_order_id, timeout_sec=2.0)
                            if sl_st and sl_st.status == OrderStatus.COMPLETE:
                                self.journal.log_event(
                                    event_type="STOP_LOSS_HIT",
                                    symbol=pos.symbol,
                                    side=pos.side.value,
                                    data={
                                        "stop_price": pos.current_stop_price,
                                        "sl_order_id": pos.protective_order_id,
                                        "fill_price": sl_st.average_price or pos.current_stop_price,
                                    },
                                )
                                pos.phase = PositionPhase.CLOSED
                                pos.quantity = 0
                                continue
                            elif not sl_st or sl_st.status not in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                                logger.warning(
                                    "Protective SL order %s for %s cancellation unconfirmed (%s); deferring target exit to next cycle",
                                    pos.protective_order_id,
                                    pos.symbol,
                                    getattr(sl_st, "status", "UNKNOWN"),
                                )
                                continue
                            else:
                                sl_filled = getattr(sl_st, "filled_quantity", 0)

                        remaining_qty = max(0, pos.quantity - int(sl_filled if isinstance(sl_filled, (int, float)) else 0))
                        if remaining_qty <= 0:
                            pos.phase = PositionPhase.CLOSED
                            pos.quantity = 0
                            continue

                        exit_req = self._create_exit_order_request(
                            symbol=pos.symbol,
                            side=exit_side,
                            quantity=remaining_qty,
                            tag=make_broker_tag(self.session_tag, pos.symbol, "TGT"),
                            current_price=current_price,
                        )
                        exit_order_id = self.broker.place_order(exit_req)
                        pos.exit_order_id = exit_order_id
                        pos.phase = PositionPhase.EXITING

                        # Confirm exit fill status before zeroing position quantity
                        exit_state = self.broker.get_order_status(exit_order_id)
                        if exit_state and exit_state.status == OrderStatus.COMPLETE:
                            pos.phase = PositionPhase.CLOSED
                            pos.quantity = 0
                            fill_px = exit_state.average_price or pos.average_entry
                            try:
                                self.trade_logger.record_exit(
                                    symbol=pos.symbol,
                                    side=exit_side,
                                    quantity=exit_state.filled_quantity or remaining_qty,
                                    exit_price=fill_px,
                                    entry_price=pos.average_entry,
                                    order_id=exit_order_id,
                                    trade_id=exit_order_id,
                                    exit_reason="TARGET_HIT",
                                    timestamp=now,
                                    entry_time=getattr(pos, "entry_time", None) or getattr(pos, "entry_timestamp", None),
                                    initial_stop=pos.initial_stop_price,
                                )
                            except Exception as log_err:
                                logger.error("Error logging immediate target exit to trade_logger: %s", log_err)
                        elif exit_state and exit_state.pending_quantity > 0:
                            pos.quantity = exit_state.pending_quantity
                            pos.phase = PositionPhase.EXITING
                        elif exit_state and hasattr(exit_state, "request") and exit_state.request and exit_state.request.quantity > 0:
                            rem = max(0, exit_state.request.quantity - exit_state.filled_quantity)
                            pos.quantity = rem
                            if pos.quantity == 0:
                                pos.phase = PositionPhase.CLOSED
                            else:
                                pos.phase = PositionPhase.EXITING
                        else:
                            pos.phase = PositionPhase.EXITING

                        self.journal.log_event(
                            event_type="TARGET_HIT_EXIT",
                            symbol=pos.symbol,
                            side=exit_side.value,
                            data={
                                "target_price": pos.target_price,
                                "exit_price": current_price,
                                "exit_order_id": exit_order_id,
                                "target_r": self.config.stops.target_r,
                            },
                            explanation=f"Fixed target of {self.config.stops.target_r}R reached at {current_price} for {pos.symbol}",
                        )
                        decisions.append(
                            StrategyDecision(
                                decision=Decision.EXIT,
                                symbol=pos.symbol,
                                side=pos.side,
                                timestamp_ist=now,
                                reason_codes=["TARGET_HIT"],
                                explanation=f"Fixed target of {self.config.stops.target_r}R reached at {current_price}",
                                broker_action=f"EXIT_TARGET_{pos.target_price}",
                            )
                        )
                        continue

                # 2. Trailing Stop Management (active only when trail_enabled is True)
                if self.config.stops.trail_enabled:
                    candles_5m = self.broker.get_historical(pos.symbol, "5minute", hist_from_5m, now)
                    candles_1m = self.broker.get_historical(pos.symbol, "minute", hist_from_1m, now)

                    atr_series = compute_atr(
                        [c.high for c in candles_5m],
                        [c.low for c in candles_5m],
                        [c.close for c in candles_5m],
                        period=self.config.stops.initial_atr_period,
                    )
                    atr_val = atr_series[-1] if atr_series else (current_price * 0.01)

                    if getattr(self.config, "exchange", "MCX") == "MCX":
                        from apex3.strategy.trail import evaluate_progressive_ratchet
                        trail_eval = evaluate_progressive_ratchet(
                            pos=pos.to_position_state(),
                            current_price=current_price,
                            candles_5m=candles_5m,
                            atr_14=atr_val,
                            config=self.config,
                            tick_size=inst_info.tick_size,
                            macro_state=macro_state,
                        )
                    else:
                        trail_eval = evaluate_trailing_stop(
                            position=pos.to_position_state(),
                            current_price=current_price,
                            atr_5m=atr_val,
                            candles_1m=candles_1m,
                            config=self.config.stops,
                            costs_config=self.config.costs,
                            current_ist_time=now.time(),
                            tick_size=inst_info.tick_size,
                        )

                    # Update High/Low water mark
                    pos.best_favorable_price = trail_eval.best_favorable_price
                    pos.trail_activated = trail_eval.trail_activated
                    pos.hwm_price = getattr(trail_eval, "hwm_price", trail_eval.best_favorable_price)
                    pos.hwm_r = getattr(trail_eval, "hwm_r", trail_eval.r_multiple)
                    pos.ratchet_stage = getattr(trail_eval, "ratchet_stage", 0)

                    broker_action = "NONE"
                    if trail_eval.should_modify:
                        headroom = self.config.stops.modification_headroom
                        max_mods = self.config.stops.max_modifications_per_order
                        if pos.stop_modification_count < (max_mods - headroom):
                            if pos.protective_order_id:
                                mod_ok = self.broker.modify_order(
                                    order_id=pos.protective_order_id,
                                    price=trail_eval.limit_price,
                                    trigger_price=trail_eval.trigger_price,
                                )
                                if mod_ok:
                                    pos.current_stop_price = trail_eval.proposed_stop
                                    pos.stop_modification_count += 1
                                    broker_action = f"MODIFIED_STOP_{trail_eval.proposed_stop}"
                                    self.journal.log_stop_change(
                                        symbol=pos.symbol,
                                        side=pos.side.value,
                                        previous_stop=pos.current_stop_price,
                                        new_stop=trail_eval.proposed_stop,
                                        reason=trail_eval.reason,
                                        r_multiple=trail_eval.r_multiple,
                                        best_favorable_price=pos.best_favorable_price,
                                        modification_number=pos.stop_modification_count,
                                    )

                    decisions.append(
                        StrategyDecision(
                            decision=Decision.MANAGE,
                            symbol=pos.symbol,
                            side=pos.side,
                            timestamp_ist=now,
                            reason_codes=["MANAGING_OPEN_POSITION"],
                            explanation=trail_eval.reason,
                            open_r=trail_eval.r_multiple,
                            best_favorable_price=pos.best_favorable_price,
                            current_stop=pos.current_stop_price,
                            proposed_stop=trail_eval.proposed_stop,
                            trail_reason=trail_eval.reason,
                            broker_action=broker_action,
                        )
                    )
                else:
                    # Trailing stop disabled (Option A): maintain structural stop towards fixed target
                    if pos.side == Side.BUY:
                        pos.best_favorable_price = max(pos.best_favorable_price, current_price)
                        fav_excursion = pos.best_favorable_price - pos.average_entry
                        curr_excursion = current_price - pos.average_entry
                    else:
                        pos.best_favorable_price = min(pos.best_favorable_price, current_price) if pos.best_favorable_price > 0 else current_price
                        fav_excursion = pos.average_entry - pos.best_favorable_price
                        curr_excursion = pos.average_entry - current_price

                    hwm_r = fav_excursion / pos.r_value if pos.r_value > 0 else 0.0
                    curr_r = curr_excursion / pos.r_value if pos.r_value > 0 else 0.0
                    open_r = hwm_r

                    # One-Time Profit-Preserving / Fee-Protected Step
                    st_cfg = {}
                    if hasattr(self.config, "raw_dict") and isinstance(self.config.raw_dict, dict):
                        st_cfg = self.config.raw_dict.get("stops_and_targets", {})
                    fee_lock_enabled = st_cfg.get("fee_lock_enabled", getattr(self.config.stops, "fee_lock_enabled", True))
                    trigger_r = float(st_cfg.get("fee_lock_trigger_r", getattr(self.config.stops, "fee_lock_trigger_r", 1.60)))
                    sched_cfg = self.config.raw_dict.get("session_schedule", self.config.raw_dict.get("schedule", {})) if hasattr(self.config, "raw_dict") and isinstance(self.config.raw_dict, dict) else {}
                    late_time = sched_cfg.get("late_day_protection_time", getattr(self.config.stops, "late_day_protection_time", "14:15"))
                    late_trigger_r = float(st_cfg.get("late_day_lock_trigger_r", getattr(self.config.stops, "late_day_lock_trigger_r", 1.15)))
                    offset_pct = float(st_cfg.get("fee_lock_offset_pct", getattr(self.config.stops, "fee_lock_offset_pct", 0.0015)))
                    profit_r = float(st_cfg.get("fee_lock_profit_r", getattr(self.config.stops, "fee_lock_profit_r", 0.60)))
                    late_profit_r = float(st_cfg.get("late_day_lock_profit_r", getattr(self.config.stops, "late_day_lock_profit_r", 0.35)))

                    now_time_str = now.strftime("%H:%M")
                    target_lock_r = None
                    if fee_lock_enabled and (not getattr(pos, "fee_lock_applied", False)):
                        if hwm_r >= trigger_r:
                            target_lock_r = profit_r
                        elif now_time_str >= late_time and curr_r >= late_trigger_r:
                            target_lock_r = late_profit_r

                    broker_action = "NONE"
                    reason_code = "HOLDING_FOR_TARGET"
                    reason_codes = [reason_code]
                    explanation = f"Holding with structural SL ({pos.current_stop_price:.2f}) towards 2.0R target ({pos.target_price:.2f})"

                    if target_lock_r is not None:
                        tick = inst_info.tick_size if inst_info and inst_info.tick_size > 0 else 0.05
                        if pos.side == Side.BUY:
                            cand = max(pos.average_entry * (1.0 + offset_pct), pos.average_entry + (target_lock_r * pos.r_value))
                            protected_stop = ceil_to_tick(cand, tick)
                            valid_move = protected_stop > pos.current_stop_price and protected_stop <= current_price
                        else:
                            cand = min(pos.average_entry * (1.0 - offset_pct), pos.average_entry - (target_lock_r * pos.r_value))
                            protected_stop = floor_to_tick(cand, tick)
                            valid_move = protected_stop < pos.current_stop_price and protected_stop >= current_price

                        if valid_move:
                            prev_stop = pos.current_stop_price
                            trigger, limit = calculate_stop_limit_prices(pos.side, protected_stop, self.config.stops, tick)
                            mod_confirmed = False
                            if pos.protective_order_id:
                                try:
                                    mod_ok = self.broker.modify_order(
                                        order_id=pos.protective_order_id,
                                        price=limit,
                                        trigger_price=trigger,
                                    )
                                    if mod_ok:
                                        mod_st = self.broker.get_order_status(pos.protective_order_id)
                                        if mod_st and mod_st.status in (OrderStatus.TRIGGER_PENDING, OrderStatus.OPEN):
                                            mod_confirmed = True
                                            if mod_st.status == OrderStatus.OPEN:
                                                logger.info(
                                                    "Stop order %s for %s triggered (OPEN) into resting limit during modification.",
                                                    pos.protective_order_id,
                                                    pos.symbol,
                                                )
                                        elif mod_st and mod_st.status == OrderStatus.COMPLETE:
                                            # Stop filled during modification
                                            self.journal.log_event(
                                                event_type="STOP_LOSS_HIT",
                                                symbol=pos.symbol,
                                                side=pos.side.value,
                                                data={
                                                    "stop_price": pos.current_stop_price,
                                                    "sl_order_id": pos.protective_order_id,
                                                    "fill_price": mod_st.average_price or pos.current_stop_price,
                                                },
                                            )
                                            pos.phase = PositionPhase.CLOSED
                                            pos.quantity = 0
                                            decisions.append(
                                                StrategyDecision(
                                                    decision=Decision.EXIT,
                                                    symbol=pos.symbol,
                                                    side=pos.side,
                                                    timestamp_ist=now,
                                                    reason_codes=["STOP_LOSS_FILLED_DURING_MODIFICATION"],
                                                    explanation=f"Protective SL order {pos.protective_order_id} filled at broker during modification",
                                                    broker_action="STOP_LOSS_FILLED",
                                                )
                                            )
                                            continue
                                except Exception as m_err:
                                    logger.error("Failed to modify stop order %s for %s: %s", pos.protective_order_id, pos.symbol, m_err)
                            else:
                                mod_confirmed = True

                            if mod_confirmed:
                                pos.current_stop_price = protected_stop
                                pos.fee_lock_applied = True
                                pos.stop_modification_count += 1
                                broker_action = f"MODIFIED_STOP_FEE_LOCK_{protected_stop}"
                                reason_codes = ["PROFIT_PRESERVING_STEP", "FEE_PROTECTED_BREAKEVEN_STEP"]
                                explanation = f"Moved stop to profit-preserving level ({protected_stop:.2f}, +{target_lock_r:.2f}R) at {open_r:.2f}R"
                                self.journal.log_stop_change(
                                    symbol=pos.symbol,
                                    side=pos.side.value,
                                    previous_stop=prev_stop,
                                    new_stop=protected_stop,
                                    reason="PROFIT_PRESERVING_STEP",
                                    r_multiple=round(open_r, 2),
                                    best_favorable_price=pos.best_favorable_price,
                                    modification_number=pos.stop_modification_count,
                                )
                            else:
                                logger.warning("Stop modification unconfirmed by broker for %s; internal stop not moved", pos.symbol)

                    decisions.append(
                        StrategyDecision(
                            decision=Decision.MANAGE,
                            symbol=pos.symbol,
                            side=pos.side,
                            timestamp_ist=now,
                            reason_codes=reason_codes if (target_lock_r is not None and valid_move and mod_confirmed) else [reason_code],
                            explanation=explanation,
                            open_r=round(open_r, 2),
                            best_favorable_price=pos.best_favorable_price,
                            current_stop=pos.current_stop_price,
                            proposed_stop=pos.current_stop_price,
                            broker_action=broker_action,
                        )
                    )
            except Exception as e:
                logger.error("Error managing position %s: %s", pos.symbol, e)

        # Daily P&L must include today's realized fills, estimated charges, and
        # broker-confirmed positions marked to current quotes. Keep this gate
        # ahead of every new-entry path so losses cannot disappear when a
        # position closes or the engine restarts.
        if self._daily_loss_halt_date == session_date:
            return decisions + [
                StrategyDecision(
                    decision=Decision.HALT,
                    timestamp_ist=now,
                    reason_codes=[self.pause_reason or "DAILY_LOSS_LIMIT_BREACHED"],
                    explanation="New entries remain paused for the rest of this trading session.",
                )
            ]

        try:
            broker_orders = self.broker.get_orders()
            broker_trades = self.broker.get_trades()
            broker_positions = self.broker.get_positions()
            risk_positions: list[PositionState] = []
            for position in broker_positions:
                if getattr(position, "product", Product.MIS) != Product.MIS:
                    continue
                qty = abs(position.quantity)
                if qty <= 0:
                    continue
                pos_side = Side.SELL if position.quantity < 0 else position.side
                mark = self.broker.get_quote(position.symbol)
                if not mark.is_fresh or mark.last_price <= 0:
                    raise ValueError(f"Missing or stale mark for {position.symbol}")
                unrealized = (
                    (mark.last_price - position.average_entry) * qty
                    if pos_side == Side.BUY
                    else (position.average_entry - mark.last_price) * qty
                )
                managed_pos = self.state_machine.positions.get(position.symbol)
                prot_stop = (
                    managed_pos.current_stop_price
                    if (managed_pos and managed_pos.current_stop_price > 0)
                    else position.protective_stop_price
                )
                init_stop = (
                    managed_pos.initial_stop_price
                    if (managed_pos and managed_pos.initial_stop_price > 0)
                    else position.initial_stop_price
                )
                risk_positions.append(
                    replace(
                        position,
                        side=pos_side,
                        quantity=qty,
                        current_price=mark.last_price,
                        protective_stop_price=prot_stop,
                        initial_stop_price=init_stop,
                        unrealized_pnl=unrealized,
                        realized_pnl=0.0,
                    )
                )

            session_pnl = build_session_pnl(
                trades=broker_trades,
                orders=broker_orders,
                open_positions=risk_positions,
                session_tag=self.session_tag,
                session_date=session_date,
                costs=self.config.costs,
            )
            self.latest_session_pnl = session_pnl
            margin_info = self.broker.get_margins()
            daily_risk_status = evaluate_portfolio_risk(
                open_positions=risk_positions,
                margin_info=margin_info,
                proposed_trade_risk_rupees=0.0,
                limits_config=self.config.limits,
                risk_config=self.config.risk,
                additional_realized_pnl=session_pnl.realized_pnl,
                estimated_charges=session_pnl.estimated_charges,
            )
            self._consecutive_pnl_failures = 0
        except Exception as exc:
            self._consecutive_pnl_failures += 1
            if self._consecutive_pnl_failures < 5:
                logger.warning(
                    "Transient P&L accounting error (%d/5): %s. Waiting for next tick.",
                    self._consecutive_pnl_failures,
                    exc,
                )
                return decisions + [
                    StrategyDecision(
                        decision=Decision.WAIT,
                        timestamp_ist=now,
                        reason_codes=["TRANSIENT_ACCOUNTING_WAIT"],
                        explanation=f"Waiting for broker accounting sync (attempt {self._consecutive_pnl_failures}/5): {exc}",
                    )
                ]

            self._daily_loss_halt_date = session_date
            self.is_paused = True
            self.pause_reason = "DAILY_PNL_ACCOUNTING_FAILED"
            self.journal.log_warning(
                severity="CRITICAL",
                code="DAILY_PNL_ACCOUNTING_FAILED",
                message=f"Could not safely calculate today's strategy P&L after 5 attempts: {exc}",
            )
            return decisions + [
                StrategyDecision(
                    decision=Decision.HALT,
                    timestamp_ist=now,
                    reason_codes=["DAILY_PNL_ACCOUNTING_FAILED"],
                    explanation="New entries blocked because daily P&L could not be verified from broker executions and position marks.",
                )
            ]

        if daily_risk_status.is_daily_loss_breached:
            self._daily_loss_halt_date = session_date
            self.is_paused = True
            self.pause_reason = "DAILY_LOSS_LIMIT_BREACHED"
            has_open = any(
                p.quantity > 0 or p.phase in (PositionPhase.ENTERING, PositionPhase.OPEN_UNPROTECTED, PositionPhase.EXITING)
                for p in self.state_machine.positions.values()
            )
            flatten_msg = ""
            if has_open:
                logger.critical("Daily loss limit breached with open positions! Squaring off immediately.")
                _, flatten_msg = self.reconciler.execute_flatten_sequence()

            self.journal.log_warning(
                severity="CRITICAL",
                code="DAILY_LOSS_LIMIT_BREACHED",
                message=f"{daily_risk_status.gate.reason}. {flatten_msg}".strip(),
                data={
                    "net_day_pnl": session_pnl.net_pnl,
                    "realized_pnl": session_pnl.realized_pnl,
                    "unrealized_pnl": session_pnl.unrealized_pnl,
                    "estimated_charges": session_pnl.estimated_charges,
                    "trade_count": session_pnl.trade_count,
                },
            )
            return decisions + [
                StrategyDecision(
                    decision=Decision.HALT,
                    timestamp_ist=now,
                    reason_codes=["DAILY_LOSS_LIMIT_BREACHED"],
                    explanation=f"{daily_risk_status.gate.reason}. {flatten_msg}".strip(),
                    gates=[daily_risk_status.gate],
                    broker_action="DAILY_LOSS_BREACH_FLATTEN" if has_open else "",
                )
            ]

        # Enforce engine pause state: if paused, block all new entries
        if self.is_paused:
            logger.info("Strategy engine is PAUSED (%s). New entries are blocked.", self.pause_reason or "Operator pause")
            if not decisions:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        timestamp_ist=now,
                        reason_codes=["ENGINE_PAUSED"],
                        explanation=f"New entries blocked because engine is paused: {self.pause_reason or 'Operator pause'}",
                    )
                )
            return decisions

        # Live mode compliance checks for NEW entries
        if self.config.mode == "live":
            if not self.config.secrets.live_trading_enabled:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.HALT,
                        timestamp_ist=now,
                        reason_codes=["LIVE_TRADING_FLAG_DISABLED"],
                        explanation="Live trading is configured in config.yaml but LIVE_TRADING_ENABLED env var is false",
                    )
                )
                return decisions
            if not self.config.secrets.static_ip_configured:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.HALT,
                        timestamp_ist=now,
                        reason_codes=["STATIC_IP_NOT_CONFIGURED"],
                        explanation="Live trading requires static IP confirmation (effective 1 April 2026)",
                    )
                )
                return decisions
            if not self.config.secrets.sebi_compliance_acknowledged:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.HALT,
                        timestamp_ist=now,
                        reason_codes=["SEBI_COMPLIANCE_NOT_ACKNOWLEDGED"],
                        explanation="SEBI retail algo compliance acknowledgement required for live trading",
                    )
                )
                return decisions

        if session_state == SessionState.SPECIAL_SESSION:
            decisions.append(
                StrategyDecision(
                    decision=Decision.WAIT,
                    timestamp_ist=now,
                    reason_codes=["SPECIAL_SESSION_NO_MIS"],
                    explanation="Special session active (e.g. Muhurat Trading). Intraday MIS trading blocked by policy.",
                )
            )
            return decisions

        if session_state == SessionState.PRE_OPEN:
            decisions.append(
                StrategyDecision(
                    decision=Decision.WAIT,
                    timestamp_ist=now,
                    reason_codes=["PRE_OPEN_SESSION"],
                    explanation="Pre-open session (before 09:15 IST). Observing and validating data.",
                )
            )
            return decisions

        # If already at max simultaneous positions, cannot take new entries
        active_managed_count = sum(
            1 for p in self.state_machine.positions.values()
            if p.quantity > 0 or p.phase in (PositionPhase.ENTERING, PositionPhase.OPEN_UNPROTECTED, PositionPhase.EXITING)
        )
        if active_managed_count >= self.config.limits.max_symbols:
            if not decisions:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        timestamp_ist=now,
                        reason_codes=["MAX_POSITIONS_REACHED"],
                        explanation=f"Already holding maximum simultaneous positions ({active_managed_count}/{self.config.limits.max_symbols})",
                    )
                )
            return decisions

        # Slot-2 Lock Gate: Only allow a 2nd position if Slot 1 is already in profit >= req_r (if req_r > 0)
        active_positions = [
            p for p in self.state_machine.positions.values()
            if p.quantity > 0 or p.phase in (PositionPhase.ENTERING, PositionPhase.OPEN_UNPROTECTED)
        ]
        if len(active_positions) == 1:
            risk_raw = self.config.raw_dict.get("risk_management", {}) if hasattr(self.config, "raw_dict") and isinstance(self.config.raw_dict, dict) else {}
            req_r = float(risk_raw.get("second_slot_requires_slot1_profit_r", 0.0))
            if req_r > 0.0:
                first_pos = active_positions[0]
                mark = self.broker.get_quote(first_pos.symbol)
                curr_px = mark.last_price if (mark and mark.is_fresh and mark.last_price > 0) else first_pos.average_entry
                r_val = first_pos.r_value if first_pos.r_value > 0 else abs(first_pos.average_entry - first_pos.initial_stop_price)
                unrealized_r = 0.0
                if r_val > 0:
                    unrealized_r = (curr_px - first_pos.average_entry) / r_val if first_pos.side == Side.BUY else (first_pos.average_entry - curr_px) / r_val

                if unrealized_r < req_r:
                    decisions.append(
                        StrategyDecision(
                            decision=Decision.WAIT,
                            timestamp_ist=now,
                            reason_codes=["SECOND_SLOT_LOCKED_UNTIL_0.5R", "SECOND_SLOT_LOCKED_UNTIL_PROFIT"],
                            explanation=f"Slot 2 locked: Active position in {first_pos.symbol} has unrealized R of {unrealized_r:.2f}R (requires >= {req_r:.2f}R)",
                        )
                    )
                    return decisions

        # -------------------------------------------------------------
        # STEP 6: Opening Range Period Check (09:15 - 09:30)
        # -------------------------------------------------------------
        if self.scheduler.is_opening_range_period(now):
            # In opening range build phase - update ranges
            session_date = (now if now.tzinfo else now.replace(tzinfo=IST)).astimezone(IST).date()
            for sym in symbols:
                candles_0915_0930 = self.broker.get_historical(sym, "5minute", hist_from_5m, now)
                self.opening_ranges[sym] = build_opening_range(sym, candles_0915_0930, session_date=session_date)

            return [
                StrategyDecision(
                    decision=Decision.WAIT,
                    timestamp_ist=now,
                    reason_codes=["BUILDING_OPENING_RANGE"],
                    explanation="Building first-15-minute opening range (09:15 - 09:30 IST). No entries permitted.",
                )
            ]

        # Check if outside entry window (e.g. after 14:15 IST)
        if not self.scheduler.can_enter(now):
            if not decisions:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        timestamp_ist=now,
                        reason_codes=["OUTSIDE_ENTRY_WINDOW"],
                        explanation="Entry window closed (14:15 IST cutoff passed). Managing existing exposure only.",
                    )
                )
            return decisions

        # -------------------------------------------------------------
        # STEP 4: Market Regime Filter Gate & Pre-Warming
        # -------------------------------------------------------------
        # 1. Pre-warm historical candles for benchmark and eligible symbols FIRST
        bench_candles = self.broker.get_historical(self.config.regime.benchmark_symbol, "5minute", hist_from_5m, now)

        stock_candles_map: dict[str, list[Candle]] = {}
        inst_map: dict[str, InstrumentInfo] = {}
        session_date = (now if now.tzinfo else now.replace(tzinfo=IST)).astimezone(IST).date()
        for sym in symbols:
            # Skip if already open or in flight
            if any(
                p.symbol == sym and (p.quantity > 0 or p.phase in (PositionPhase.ENTERING, PositionPhase.OPEN_UNPROTECTED, PositionPhase.EXITING))
                for p in self.state_machine.positions.values()
            ):
                continue
            inst_map[sym] = self.broker.get_instrument_info(sym)
            s_candles = self.broker.get_historical(sym, "5minute", hist_from_5m, now)
            stock_candles_map[sym] = s_candles
            # Pre-build/cache opening range so late starts (e.g. 09:50 IST) reconstruct immediately
            if s_candles and (sym not in self.opening_ranges or not self.opening_ranges[sym].is_complete):
                built_or = build_opening_range(sym, s_candles, session_date=session_date)
                if built_or.is_complete or built_or.candle_count > 0:
                    self.opening_ranges[sym] = built_or

        # 2. Batch-fetch quotes IMMEDIATELY AFTER historical pre-warm so quotes are ultra-fresh (< 0.2s old)
        all_symbols_to_quote = [self.config.regime.benchmark_symbol] + [s for s in symbols if s != self.config.regime.benchmark_symbol]
        batch_quotes: dict[str, Quote] = {}
        try:
            res = self.broker.get_quotes_batch(all_symbols_to_quote)
            if type(res) is dict:
                batch_quotes = res
        except Exception as bqe:
            logger.warning("Batch quote fetch failed; falling back to individual quotes: %s", bqe)

        bench_quote = batch_quotes.get(self.config.regime.benchmark_symbol) or self.broker.get_quote(self.config.regime.benchmark_symbol)
        regime_eval = evaluate_market_regime(bench_candles, self.config.regime, bench_quote)
        self.latest_regime_eval = regime_eval

        if regime_eval.regime == MarketRegime.NEUTRAL:
            if not decisions:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        timestamp_ist=now,
                        reason_codes=["REGIME_NEUTRAL_OR_MIXED"],
                        explanation=f"Broad-market regime is neutral: {regime_eval.gate.reason}",
                        gates=[regime_eval.gate],
                    )
                )
            return decisions

        # -------------------------------------------------------------
        # STEP 5, 7, 8: Candidate Signal Evaluation & Sizing
        # -------------------------------------------------------------
        qualified_candidates = []
        evaluated_signals = {}

        core_list = set()
        core_mult = 1.35
        if hasattr(self.config, "raw_dict") and isinstance(self.config.raw_dict, dict):
            radar_cfg = self.config.raw_dict.get("radar", {})
            core_list = set(radar_cfg.get("core_watchlist", []))
            core_mult = float(radar_cfg.get("core_conviction_multiplier", 1.35))

        for sym in symbols:
            # Skip if already open or in flight
            if any(
                p.symbol == sym and (p.quantity > 0 or p.phase in (PositionPhase.ENTERING, PositionPhase.OPEN_UNPROTECTED, PositionPhase.EXITING))
                for p in self.state_machine.positions.values()
            ):
                continue

            inst = inst_map.get(sym)
            if inst is None or (inst.upper_circuit <= 0.0 and inst.lower_circuit <= 0.0):
                inst = self.broker.get_instrument_info(sym)
                inst_map[sym] = inst
            quote = batch_quotes.get(sym) or self.broker.get_quote(sym)
            stock_candles = stock_candles_map.get(sym) or self.broker.get_historical(sym, "5minute", hist_from_5m, now)

            # Build/retrieve opening range
            or_range = self.opening_ranges.get(sym)
            if or_range is None or not or_range.is_complete:
                or_range = build_opening_range(sym, stock_candles, session_date=session_date)
                self.opening_ranges[sym] = or_range

            is_core = (sym in core_list) if core_list else True
            sig = evaluate_signal(
                symbol=sym,
                stock_candles_5m=stock_candles,
                benchmark_candles_5m=bench_candles,
                opening_range=or_range,
                market_regime=regime_eval.regime,
                instrument=inst,
                quote=quote,
                config=self.config.signal,
                is_entry_window=True,
                is_core_symbol=is_core,
            )
            eval_candles = stock_candles

            # Late-Start First-Scan Catch-Up Evaluation:
            # If the engine started late (e.g., 09:50 IST) and the latest bar is WAIT, scan earlier
            # post-OR bars of today once per session to catch valid earlier breakouts or reject stale ones.
            if sym not in self._late_start_catchup_done:
                self._late_start_catchup_done.add(sym)
                now_ist_time = self._ensure_ist(now).time()
                if sig.decision == Decision.WAIT and now_ist_time >= time(9, 40):
                    comp_c = [c for c in stock_candles if c.is_complete] or stock_candles
                    if len(comp_c) >= 2:
                        for idx in range(len(comp_c) - 2, -1, -1):
                            cand_bar = comp_c[idx]
                            cand_ts = self._ensure_ist(cand_bar.timestamp)
                            if cand_ts.date() != session_date or cand_ts.time() < time(9, 30):
                                break
                            sub_stock = comp_c[: idx + 1]
                            sub_bench = [bc for bc in bench_candles if self._ensure_ist(bc.timestamp) <= cand_ts] or bench_candles
                            catchup_sig = evaluate_signal(
                                symbol=sym,
                                stock_candles_5m=sub_stock,
                                benchmark_candles_5m=sub_bench,
                                opening_range=or_range,
                                market_regime=regime_eval.regime,
                                instrument=inst,
                                quote=quote,
                                config=self.config.signal,
                                is_entry_window=True,
                                is_core_symbol=is_core,
                            )
                            if catchup_sig.decision in (Decision.PROPOSED_LONG, Decision.PROPOSED_SHORT):
                                c_tick = inst.tick_size if inst and inst.tick_size > 0 else 0.05
                                slip_cfg = getattr(getattr(self.config, "entry", None), "max_slippage_pct", 0.15)
                                slip_pct = (slip_cfg / 100.0) if slip_cfg >= 0.01 else (slip_cfg if slip_cfg > 0 else 0.0015)
                                c_side = catchup_sig.side
                                c_price = catchup_sig.signal_price
                                if c_side == Side.BUY:
                                    c_entry = min(round_to_tick(c_price + c_tick, c_tick), round_to_tick(c_price * (1.0 + slip_pct), c_tick))
                                else:
                                    c_entry = max(round_to_tick(c_price - c_tick, c_tick), round_to_tick(c_price * (1.0 - slip_pct), c_tick))
                                c_atr_series = compute_atr(
                                    [c.high for c in sub_stock],
                                    [c.low for c in sub_stock],
                                    [c.close for c in sub_stock],
                                    period=self.config.stops.initial_atr_period,
                                )
                                c_atr = c_atr_series[-1] if c_atr_series else (c_entry * 0.01)
                                lb_bars = getattr(self.config.stops, "swing_lookback_bars", 5)
                                lb_c = sub_stock[-lb_bars:] if len(sub_stock) >= lb_bars else sub_stock
                                c_stop = calculate_initial_stop(
                                    side=c_side,
                                    entry_price=c_entry,
                                    signal_low=min(c.low for c in lb_c),
                                    signal_high=max(c.high for c in lb_c),
                                    atr_value=c_atr,
                                    config=self.config.stops,
                                    tick_size=c_tick,
                                )
                                c_target = calculate_target_price(
                                    side=c_side,
                                    entry_price=c_entry,
                                    initial_stop=c_stop,
                                    target_r=self.config.stops.target_r,
                                    tick_size=c_tick,
                                )
                                late_val = validate_late_start_signal(
                                    symbol=sym,
                                    side=c_side,
                                    signal_price=c_price,
                                    initial_stop=c_stop,
                                    target_price=c_target,
                                    or_high=or_range.high if or_range else 0.0,
                                    or_low=or_range.low if or_range else 0.0,
                                    trigger_bar_timestamp=cand_ts,
                                    subsequent_candles=stock_candles,
                                    quote=quote,
                                    now_ist=now,
                                    entry_start_hhmm=self.config.schedule.entry_start,
                                    entry_cutoff_hhmm=self.config.schedule.entry_cutoff,
                                    max_price_drift_pct=slip_pct,
                                )
                                if late_val.is_valid:
                                    sig = catchup_sig
                                    eval_candles = sub_stock
                                else:
                                    decisions.append(
                                        StrategyDecision(
                                            decision=Decision.WAIT,
                                            symbol=sym,
                                            side=c_side,
                                            timestamp_ist=now,
                                            reason_codes=[late_val.reason_code, "STALE_SIGNAL_SKIPPED"],
                                            explanation=late_val.explanation,
                                        )
                                    )
                                break

            evaluated_signals[sym] = (sig, inst, eval_candles, quote)
            if sig.decision in (Decision.PROPOSED_LONG, Decision.PROPOSED_SHORT):
                qualified_candidates.append(sym)

        if not qualified_candidates:
            # No symbol passed all signal gates
            reasons = [f"{s}: {evaluated_signals[s][0].explanation}" for s in evaluated_signals]
            if not decisions:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        timestamp_ist=now,
                        reason_codes=["NO_QUALIFYING_CANDIDATES"],
                        explanation="; ".join(reasons) if reasons else "Watchlist empty or no signals",
                    )
                )
            return decisions

        # Rank qualified candidates by Conviction Score: (|RS| * 100 * relative_volume) / max(spread_bps, 1.0)

        def compute_conviction(sym_name: str) -> float:
            sig_item, _, _, quote_item = evaluated_signals[sym_name]
            bid = getattr(quote_item, "bid", 0.0) if quote_item else 0.0
            ask = getattr(quote_item, "ask", 0.0) if quote_item else 0.0
            bid_val = float(bid) if isinstance(bid, (int, float)) and not isinstance(bid, bool) else 0.0
            ask_val = float(ask) if isinstance(ask, (int, float)) and not isinstance(ask, bool) else 0.0
            mid = (bid_val + ask_val) / 2.0 if (bid_val > 0 and ask_val > 0) else 0.0
            spread_bps = ((ask_val - bid_val) / mid) * 10000.0 if mid > 0 else 1.0
            spread_floor = max(spread_bps, 1.0)
            raw_rs = getattr(sig_item, "relative_strength", 0.0) if sig_item else 0.0
            rs = float(raw_rs) if isinstance(raw_rs, (int, float)) and not isinstance(raw_rs, bool) else 0.0
            raw_rvol = getattr(sig_item, "relative_volume", 1.0) if sig_item else 1.0
            rvol = float(raw_rvol) if isinstance(raw_rvol, (int, float)) and not isinstance(raw_rvol, bool) else 1.0
            score = (abs(rs) * 100.0 * rvol) / spread_floor
            if sym_name in core_list:
                score *= core_mult
            return score

        qualified_candidates.sort(key=compute_conviction, reverse=True)

        # Completed-bar deduplication: at most 1 entry per 5-minute candle
        if self._last_entry_bar_timestamp is not None:
            def _latest_completed_ts(c_list: list[Candle]):
                comp = [c for c in c_list if c.is_complete] or c_list
                return comp[-1].timestamp if comp else None

            qualified_candidates = [
                sym for sym in qualified_candidates
                if not (
                    evaluated_signals.get(sym)
                    and len(evaluated_signals[sym]) > 2
                    and evaluated_signals[sym][2]
                    and _latest_completed_ts(evaluated_signals[sym][2]) == self._last_entry_bar_timestamp
                )
            ]

        max_new = 1
        if hasattr(self.config, "raw_dict") and isinstance(self.config.raw_dict, dict):
            max_new = int(self.config.raw_dict.get("risk_management", {}).get("max_new_entries_per_bar", 1))

        # Sizing and Risk for qualified candidates
        num_qualified = len(qualified_candidates)
        entries_submitted = 0

        for sym in qualified_candidates:
            if entries_submitted >= max_new:
                break
            if len(risk_positions) >= self.config.limits.max_symbols:
                break
            sig, inst, stock_candles, quote = evaluated_signals[sym]
            completed_stock_candles = [c for c in stock_candles if c.is_complete] or stock_candles
            if self._last_entry_bar_timestamp is not None and completed_stock_candles and completed_stock_candles[-1].timestamp == self._last_entry_bar_timestamp:
                logger.debug("Bar %s already has an entry taken for %s. Skipping.", self._last_entry_bar_timestamp, sym)
                continue
            sig_price = sig.signal_price
            side = sig.side

            # Compute marketable limit entry price up-front before sizing & stops
            tick = inst.tick_size if inst and inst.tick_size > 0 else 0.05
            slip_cfg = getattr(getattr(self.config, "entry", None), "max_slippage_pct", 0.15)
            slip_pct = (slip_cfg / 100.0) if slip_cfg >= 0.01 else (slip_cfg if slip_cfg > 0 else 0.0015)

            if side == Side.BUY:
                cand_price = round_to_tick(sig_price + tick, tick_size=tick)
                cap_price = round_to_tick(sig_price * (1.0 + slip_pct), tick_size=tick)
                marketable_entry_price = min(cand_price, cap_price)
                if inst and inst.upper_circuit > 0:
                    marketable_entry_price = min(marketable_entry_price, round_to_tick(inst.upper_circuit, tick_size=tick))
            else:
                cand_price = round_to_tick(sig_price - tick, tick_size=tick)
                cap_price = round_to_tick(sig_price * (1.0 - slip_pct), tick_size=tick)
                marketable_entry_price = max(cand_price, cap_price)
                if inst and inst.lower_circuit > 0:
                    marketable_entry_price = max(marketable_entry_price, round_to_tick(inst.lower_circuit, tick_size=tick))
            entry_price = marketable_entry_price

            # Reject entry if live ask/bid has already gapped past cap_price
            if side == Side.BUY and quote and quote.ask > cap_price:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        symbol=sym,
                        side=side,
                        timestamp_ist=now,
                        reason_codes=["ENTRY_SLIPPAGE_EXCEEDED"],
                        explanation=f"Live ask {quote.ask:.2f} exceeded max entry cap {cap_price:.2f} (slippage > {slip_pct*100:.2f}%)",
                    )
                )
                continue
            if side == Side.SELL and quote and quote.bid > 0 and quote.bid < cap_price:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        symbol=sym,
                        side=side,
                        timestamp_ist=now,
                        reason_codes=["ENTRY_SLIPPAGE_EXCEEDED"],
                        explanation=f"Live bid {quote.bid:.2f} fell below min entry floor {cap_price:.2f} (slippage > {slip_pct*100:.2f}%)",
                    )
                )
                continue

            # Calculate ATR for initial stop (strictly on completed bars)
            atr_series = compute_atr(
                [c.high for c in completed_stock_candles],
                [c.low for c in completed_stock_candles],
                [c.close for c in completed_stock_candles],
                period=self.config.stops.initial_atr_period,
            )
            atr_val = atr_series[-1] if atr_series else (entry_price * 0.01)

            # Structural initial stop (true 5-bar structural swing anchor on completed bars)
            lookback_bars = getattr(self.config.stops, "swing_lookback_bars", 5)
            lookback_candles = completed_stock_candles[-lookback_bars:] if len(completed_stock_candles) >= lookback_bars else completed_stock_candles
            swing_low_5 = min(c.low for c in lookback_candles) if lookback_candles else completed_stock_candles[-1].low
            swing_high_5 = max(c.high for c in lookback_candles) if lookback_candles else completed_stock_candles[-1].high
            initial_stop = calculate_initial_stop(
                side=side,
                entry_price=entry_price,
                signal_low=swing_low_5,
                signal_high=swing_high_5,
                atr_value=atr_val,
                config=self.config.stops,
                tick_size=inst.tick_size,
            )

            # Gate 9 Stop-width bounds: 0.55% <= R% <= 0.90%
            min_stop_pct = getattr(self.config.signal, "min_stop_distance_pct", 0.0055)
            max_stop_pct = getattr(self.config.signal, "max_stop_distance_pct", 0.0090)
            stop_dist_pct = abs(entry_price - initial_stop) / entry_price if entry_price > 0 else 0.0
            if stop_dist_pct < min_stop_pct or stop_dist_pct > max_stop_pct:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        symbol=sym,
                        timestamp_ist=now,
                        reason_codes=["STOP_WIDTH_OUT_OF_BOUNDS"],
                        explanation=f"Stop width {stop_dist_pct*100:.2f}% is outside allowed bounds [{min_stop_pct*100:.2f}%, {max_stop_pct*100:.2f}%]",
                    )
                )
                continue

            # Late-Start & Stale-Signal Validity Gate:
            # If the trigger bar closed more than 5 minutes before `now`, validate that price is still
            # within slip_pct of signal_price, stop-loss was not hit, and target was not hit.
            if completed_stock_candles:
                trig_bar = completed_stock_candles[-1]
                trig_ts = self._ensure_ist(trig_bar.timestamp)
                bar_close_ts = trig_ts + timedelta(minutes=5)
                if (now - bar_close_ts).total_seconds() > 300.0:
                    cand_target = calculate_target_price(
                        side=side,
                        entry_price=entry_price,
                        initial_stop=initial_stop,
                        target_r=self.config.stops.target_r,
                        tick_size=tick,
                    )
                    or_obj = self.opening_ranges.get(sym)
                    late_val = validate_late_start_signal(
                        symbol=sym,
                        side=side,
                        signal_price=sig_price,
                        initial_stop=initial_stop,
                        target_price=cand_target,
                        or_high=or_obj.high if or_obj else 0.0,
                        or_low=or_obj.low if or_obj else 0.0,
                        trigger_bar_timestamp=trig_ts,
                        subsequent_candles=stock_candles_map.get(sym) or stock_candles,
                        quote=quote,
                        now_ist=now,
                        entry_start_hhmm=self.config.schedule.entry_start,
                        entry_cutoff_hhmm=self.config.schedule.entry_cutoff,
                        max_price_drift_pct=slip_pct,
                    )
                    if not late_val.is_valid:
                        decisions.append(
                            StrategyDecision(
                                decision=Decision.WAIT,
                                symbol=sym,
                                side=side,
                                timestamp_ist=now,
                                reason_codes=[late_val.reason_code, "STALE_SIGNAL_SKIPPED"],
                                explanation=late_val.explanation,
                            )
                        )
                        continue

            # Estimate order margin from broker
            dummy_req = OrderRequest(
                symbol=sym,
                exchange=getattr(self.config, "exchange", "MCX"),
                side=side,
                quantity=100,
                price=entry_price,
                product=Product.MIS,
                order_type=OrderType.LIMIT,
            )
            order_margin_est = self.broker.estimate_order_margin(dummy_req)

            # Sizing calculation
            sizing = size_position(
                symbol=sym,
                side=side,
                entry_price=entry_price,
                stop_price=initial_stop,
                num_qualified_symbols=num_qualified,
                margin_info=margin_info,
                instrument=inst,
                limits_config=self.config.limits,
                risk_config=self.config.risk,
                order_margin_estimate=order_margin_est,
            )

            if not sizing.is_valid or sizing.quantity <= 0:
                # Attempt automatic contract step-down for MCX Precious Metals
                stepped_down = False
                sym_root = getattr(inst, "symbol_root", "") or ""
                if not sym_root:
                    for r in ("SILVERMIC", "GOLDGUINEA", "GOLDPETAL", "GOLDTEN", "SILVERM", "GOLDM", "SILVER", "GOLD"):
                        if sym.upper().startswith(r):
                            sym_root = r
                            break
                step_down_map = {
                    "SILVER": ["SILVERM", "SILVERMIC"],
                    "SILVERM": ["SILVERMIC"],
                    "GOLD": ["GOLDM", "GOLDTEN", "GOLDGUINEA"],
                    "GOLDM": ["GOLDTEN", "GOLDGUINEA"],
                }
                candidates_to_try = step_down_map.get(sym_root, [])
                for fb_root in candidates_to_try:
                    try:
                        resolved = self.broker.resolve_active_mcx_futures([fb_root])
                        fb_inst = resolved.get(fb_root)
                        if fb_inst:
                            fb_sym = fb_inst.symbol
                            fb_dummy = OrderRequest(
                                symbol=fb_sym,
                                exchange=getattr(self.config, "exchange", "MCX"),
                                side=side,
                                quantity=1,
                                price=entry_price,
                                product=Product.MIS,
                                order_type=OrderType.LIMIT,
                            )
                            fb_margin_est = self.broker.estimate_order_margin(fb_dummy)
                            fb_sizing = size_position(
                                symbol=fb_sym,
                                side=side,
                                entry_price=entry_price,
                                stop_price=initial_stop,
                                num_qualified_symbols=num_qualified,
                                margin_info=margin_info,
                                instrument=fb_inst,
                                limits_config=self.config.limits,
                                risk_config=self.config.risk,
                                order_margin_estimate=fb_margin_est,
                            )
                            if fb_sizing.is_valid and fb_sizing.quantity > 0:
                                logger.info(
                                    "Automatic step-down: transitioned %s -> %s (lots: %d) due to %s",
                                    sym, fb_sym, fb_sizing.quantity, sizing.rejection_reason
                                )
                                sym = fb_sym
                                inst = fb_inst
                                sizing = fb_sizing
                                stepped_down = True
                                break
                    except Exception as sde:
                        logger.debug("Step-down attempt to %s failed: %s", fb_root, sde)

                if not stepped_down:
                    decisions.append(
                        StrategyDecision(
                            decision=Decision.WAIT,
                            symbol=sym,
                            side=side,
                            timestamp_ist=now,
                            reason_codes=[sizing.rejection_code or "SIZING_REJECTED"],
                            explanation=f"Sizing rejected for {sym}: {sizing.rejection_reason}",
                            gates=[sizing.gate],
                        )
                    )
                    continue

            # Portfolio Risk Gate
            risk_status = evaluate_portfolio_risk(
                open_positions=risk_positions,
                margin_info=margin_info,
                proposed_trade_risk_rupees=sizing.trade_risk_rupees,
                limits_config=self.config.limits,
                risk_config=self.config.risk,
                additional_realized_pnl=session_pnl.realized_pnl,
                estimated_charges=session_pnl.estimated_charges,
            )

            if not risk_status.can_open_new_trade:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        symbol=sym,
                        side=side,
                        timestamp_ist=now,
                        reason_codes=["PORTFOLIO_RISK_LIMIT_BREACHED"],
                        explanation=f"Portfolio risk check failed: {risk_status.gate.reason}",
                        gates=[risk_status.gate],
                    )
                )
                continue

            # -------------------------------------------------------------
            # STEP 9 & 10: Proposal Ready & Execution
            # -------------------------------------------------------------
            tag = make_broker_tag(self.session_tag, sym, f"E{now.strftime('%H%M')}")
            broker_action = "PLACE_ENTRY_AND_PROTECTION"

            # Duplicate-start / Order-book idempotency check before placing entry order:
            # Verify no active entry order or existing MIS position is already on the broker for `sym`
            dup_orders = [
                o for o in broker_orders
                if getattr(o, "request", None)
                and o.request.symbol == sym
                and (
                    (o.status in (OrderStatus.OPEN, OrderStatus.TRIGGER_PENDING, OrderStatus.PENDING) and o.request.order_type == OrderType.LIMIT)
                    or (o.request.tag == tag and o.status in (OrderStatus.OPEN, OrderStatus.TRIGGER_PENDING, OrderStatus.PENDING, OrderStatus.COMPLETE))
                )
            ]
            dup_positions = [
                p for p in broker_positions
                if p.symbol == sym and getattr(p, "product", Product.MIS) == Product.MIS and abs(p.quantity) > 0
            ]
            if dup_orders or dup_positions:
                decisions.append(
                    StrategyDecision(
                        decision=Decision.WAIT,
                        symbol=sym,
                        side=side,
                        timestamp_ist=now,
                        reason_codes=["DUPLICATE_ORDER_PREVENTED"],
                        explanation=f"Skipped duplicate entry for {sym}: active order or MIS position already exists in broker order book",
                    )
                )
                continue

            entry_req = OrderRequest(
                symbol=sym,
                exchange=getattr(self.config, "exchange", "MCX"),
                side=side,
                quantity=sizing.quantity,
                product=Product.MIS,
                order_type=OrderType.LIMIT,
                price=marketable_entry_price,
                tag=tag,
                strategy_id=self.session_tag,
            )

            try:
                entry_order_id = self.broker.place_order(entry_req)
                entry_state = self.broker.get_order_status(entry_order_id)
                if entry_state and entry_state.status in (OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.ERROR):
                    self._last_entry_bar_timestamp = None
                    rej_msg = getattr(entry_state, "rejection_reason", None) or getattr(entry_state, "status_message", "Order rejected")
                    self.journal.log_warning(
                        severity="ERROR",
                        code="ENTRY_ORDER_REJECTED",
                        message=f"Entry order {entry_order_id} for {sym} was {entry_state.status.value}: {rej_msg}",
                    )
                    decisions.append(
                        StrategyDecision(
                            decision=Decision.WAIT,
                            symbol=sym,
                            side=side,
                            timestamp_ist=now,
                            reason_codes=["ENTRY_REJECTED"],
                            explanation=f"Entry order {entry_order_id} {entry_state.status.value}: {rej_msg}",
                        )
                    )
                    continue

                managed = self.state_machine.get_or_create_position(sym)
                managed.side = side
                managed.entry_order_id = entry_order_id
                managed.strategy_tag = tag
                managed.entry_timestamp = now
                managed.requested_quantity = sizing.quantity
                managed.initial_stop_price = initial_stop
                managed.current_stop_price = initial_stop
                managed.protective_order_id = ""
                managed.exit_order_id = ""
                managed.fee_lock_applied = False
                managed.trail_activated = False
                managed.stop_modification_count = 0
                managed.target_price = calculate_target_price(
                    side=side,
                    entry_price=marketable_entry_price,
                    initial_stop=initial_stop,
                    target_r=self.config.stops.target_r,
                    tick_size=inst.tick_size,
                )

                # Record bar timestamp deduplication (using completed signal bar)
                self._last_entry_bar_timestamp = completed_stock_candles[-1].timestamp if completed_stock_candles else now

                entry_filled_qty = getattr(entry_state, "filled_quantity", 0)
                is_filled_qty_positive = isinstance(entry_filled_qty, (int, float)) and entry_filled_qty > 0
                is_complete = getattr(entry_state, "status", None) == OrderStatus.COMPLETE

                if entry_state and (is_complete or is_filled_qty_positive):
                    current_fill_qty = int(getattr(entry_state, "filled_quantity", 0) or 0)
                    if current_fill_qty <= 0:
                        managed.quantity = 0
                        managed.phase = PositionPhase.FLAT
                        continue

                    entry_avg_px = getattr(entry_state, "average_price", 0.0)
                    is_avg_px_positive = isinstance(entry_avg_px, (int, float)) and entry_avg_px > 0
                    actual_entry = float(entry_avg_px) if is_avg_px_positive else marketable_entry_price
                    managed.quantity = current_fill_qty
                    managed.average_entry = actual_entry
                    managed.best_favorable_price = actual_entry
                    managed.r_value = abs(actual_entry - initial_stop)
                    managed.target_price = calculate_target_price(
                        side=side,
                        entry_price=actual_entry,
                        initial_stop=initial_stop,
                        target_r=self.config.stops.target_r,
                        tick_size=inst.tick_size,
                    )

                    exit_side = Side.SELL if side == Side.BUY else Side.BUY
                    trigger, limit = calculate_stop_limit_prices(
                        side,
                        initial_stop,
                        self.config.stops,
                        inst.tick_size,
                        lower_circuit=inst.lower_circuit if inst else 0.0,
                        upper_circuit=inst.upper_circuit if inst else 0.0,
                    )

                    # Place protective stop immediately for filled quantity
                    sl_req = OrderRequest(
                        symbol=sym,
                        exchange=getattr(self.config, "exchange", "MCX"),
                        side=exit_side,
                        quantity=current_fill_qty,
                        product=Product.MIS,
                        order_type=OrderType.SL,
                        price=limit,
                        trigger_price=trigger,
                        tag=make_broker_tag(self.session_tag, sym, "SL"),
                        strategy_id=self.session_tag,
                    )
                    sl_order_id = ""
                    try:
                        sl_order_id = self.broker.place_order(sl_req)
                        managed.protective_order_id = sl_order_id
                    except Exception as sl_err:
                        logger.error("Failed to place protective stop for %s: %s", sym, sl_err)

                    # On partial fill, cancel the unfilled remainder and wait for terminal confirmation
                    if not is_complete:
                        try:
                            self.broker.cancel_order(entry_order_id)
                        except Exception as cancel_err:
                            logger.warning("Failed to cancel remaining entry for %s: %s", sym, cancel_err)
                        term_state = self._confirm_order_terminal(entry_order_id, timeout_sec=0.5)
                        if term_state:
                            term_filled_qty = int(getattr(term_state, "filled_quantity", 0) or 0)
                            if term_filled_qty > managed.quantity:
                                # Late fill occurred while cancelling!
                                logger.info("Late fill detected during cancel for %s: qty increased from %d to %d", sym, managed.quantity, term_filled_qty)
                                managed.quantity = term_filled_qty
                                if term_state.average_price > 0:
                                    managed.average_entry = term_state.average_price
                                if managed.protective_order_id:
                                    try:
                                        self.broker.modify_order(managed.protective_order_id, quantity=managed.quantity)
                                    except Exception as e:
                                        logger.error("Failed to modify stop after late fill for %s: %s", sym, e)

                            if term_state.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.COMPLETE):
                                entry_state = term_state
                            else:
                                logger.warning(
                                    "Entry order %s remainder cancel unconfirmed (%s); proceeding to verify protection",
                                    entry_order_id,
                                    getattr(term_state, "status", "UNKNOWN"),
                                )
                                entry_state = term_state
                        else:
                            logger.warning("Entry order %s remainder cancel timed out; proceeding to verify protection", entry_order_id)

                    confirmed_fill_qty = int(getattr(entry_state, "filled_quantity", 0) or 0)
                    if confirmed_fill_qty <= 0:
                        managed.quantity = 0
                        managed.phase = PositionPhase.FLAT
                        continue

                    # Reconcile confirmed fill quantity against live broker positions (strictly verify, no max() absorption)
                    actual_qty = confirmed_fill_qty
                    try:
                        fresh_pos = [p for p in self.broker.get_positions() if p.symbol == sym and getattr(p, "product", Product.MIS) == Product.MIS]
                        if fresh_pos:
                            broker_net_qty = abs(fresh_pos[0].quantity)
                            if broker_net_qty > 0 and broker_net_qty != confirmed_fill_qty:
                                logger.critical(
                                    "EMERGENCY: Broker MIS position (%d) != confirmed fill quantity (%d) for %s! Halting for manual reconciliation.",
                                    broker_net_qty, confirmed_fill_qty, sym,
                                )
                                self.journal.log_warning(
                                    severity="EMERGENCY",
                                    code="POSITION_QTY_DISAGREEMENT",
                                    message=f"Broker MIS position {broker_net_qty} does not match fill {confirmed_fill_qty} for {sym}",
                                    data={"broker_qty": broker_net_qty, "fill_qty": confirmed_fill_qty},
                                )
                                self.is_paused = True
                                self.pause_reason = f"QTY_MISMATCH_{sym}"
                                managed.phase = PositionPhase.ERROR
                                continue
                    except Exception as pe:
                        logger.warning("Error fetching broker positions for %s fill reconciliation: %s", sym, pe)

                    entry_avg_px = getattr(entry_state, "average_price", 0.0)
                    is_avg_px_positive = isinstance(entry_avg_px, (int, float)) and entry_avg_px > 0
                    actual_entry = float(entry_avg_px) if is_avg_px_positive else marketable_entry_price
                    managed.quantity = actual_qty
                    managed.average_entry = actual_entry
                    managed.best_favorable_price = actual_entry
                    managed.r_value = abs(actual_entry - initial_stop)
                    managed.target_price = calculate_target_price(
                        side=side,
                        entry_price=actual_entry,
                        initial_stop=initial_stop,
                        target_r=self.config.stops.target_r,
                        tick_size=inst.tick_size,
                    )

                    # Verify protective stop is active and matching confirmed fill quantity
                    sl_confirmed = False
                    if managed.protective_order_id:
                        try:
                            sl_confirmed = self._confirm_protective_sl(managed.protective_order_id, actual_qty, timeout_sec=2.0)
                            if sl_confirmed:
                                managed.phase = PositionPhase.OPEN_PROTECTED
                                managed.entry_timestamp = managed.entry_timestamp or now
                                managed.initial_stop_price = initial_stop
                                try:
                                    self.trade_logger.record_entry(
                                        symbol=sym,
                                        side=side,
                                        quantity=actual_qty,
                                        price=actual_entry,
                                        order_id=entry_order_id,
                                        trade_id=entry_order_id,
                                        timestamp=now,
                                        initial_stop=initial_stop,
                                        target_price=managed.target_price,
                                        risk_rupees=getattr(managed, "r_value", 0.0) * actual_qty,
                                        tag=getattr(managed, "strategy_tag", self.session_tag),
                                    )
                                except Exception as log_err:
                                    logger.error("Error logging immediate entry to trade_logger: %s", log_err)
                            else:
                                logger.error("Stop order %s in unconfirmed state: %s", managed.protective_order_id, getattr(self.broker.get_order_status(managed.protective_order_id), 'status', None))
                        except Exception as ce:
                            logger.error("Failed to verify stop status for %s: %s", sym, ce)

                    if not sl_confirmed:
                        self._last_entry_bar_timestamp = None
                        # Recovery path: liquidate filled position immediately to prevent naked risk
                        logger.critical("EMERGENCY: Protective stop placement failed/unconfirmed for %s! Liquidating immediately.", sym)
                        self.journal.log_warning(
                            severity="EMERGENCY",
                            code="PROTECTIVE_STOP_UNCONFIRMED",
                            message=f"Protective stop for {sym} failed or unconfirmed. Executing immediate market liquidation.",
                        )
                        try:
                            flat_req = self._create_exit_order_request(
                                symbol=sym,
                                side=exit_side,
                                quantity=actual_qty,
                                tag=make_broker_tag(self.session_tag, sym, "FLT"),
                                current_price=actual_entry,
                            )
                            emer_flat_id = self.broker.place_order(flat_req)
                            managed.exit_order_id = emer_flat_id
                            managed.phase = PositionPhase.EXITING
                        except Exception as flat_err:
                            logger.critical("FATAL: Emergency liquidation order failed for %s: %s", sym, flat_err)
                            managed.phase = PositionPhase.ERROR
                        self.is_paused = True
                        self.pause_reason = f"PROTECTIVE_STOP_FAILED_{sym}"
                        continue

                    risk_positions.append(
                        PositionState(
                            symbol=sym,
                            side=side,
                            quantity=actual_qty,
                            average_entry=actual_entry,
                            current_price=actual_entry,
                            protective_stop_price=initial_stop,
                            initial_stop_price=initial_stop,
                            is_protected=True,
                            strategy_tag=tag,
                        )
                    )

                    self.journal.log_event(
                        event_type="ENTRY_AND_PROTECTION_PLACED",
                        symbol=sym,
                        side=side.value,
                        data={
                            "entry_order_id": entry_order_id,
                            "sl_order_id": sl_order_id,
                            "quantity": actual_qty,
                            "entry_price": actual_entry,
                            "initial_stop": initial_stop,
                            "target_price": managed.target_price,
                        },
                        explanation=f"Placed entry, target ({managed.target_price:.2f}), and protective stop for {sym} ({actual_qty} shares)",
                    )
                    broker_action = f"BUY_ENTRY_AND_SL_{initial_stop}" if side == Side.BUY else f"SELL_ENTRY_AND_SL_{initial_stop}"
                else:
                    # Accepted on orderbook but unfilled yet — do NOT place naked SL
                    managed.quantity = 0
                    managed.average_entry = marketable_entry_price
                    managed.best_favorable_price = marketable_entry_price
                    managed.r_value = abs(marketable_entry_price - initial_stop)
                    managed.phase = PositionPhase.ENTERING
                    actual_qty = sizing.quantity
                    actual_entry = marketable_entry_price

                    risk_positions.append(
                        PositionState(
                            symbol=sym,
                            side=side,
                            quantity=sizing.quantity,
                            average_entry=marketable_entry_price,
                            current_price=marketable_entry_price,
                            protective_stop_price=initial_stop,
                            initial_stop_price=initial_stop,
                            is_protected=False,
                            strategy_tag=tag,
                        )
                    )

                    self.journal.log_event(
                        event_type="ENTRY_ORDER_PENDING",
                        symbol=sym,
                        side=side.value,
                        data={
                            "entry_order_id": entry_order_id,
                            "quantity": sizing.quantity,
                            "order_price": marketable_entry_price,
                            "initial_stop": initial_stop,
                            "target_price": managed.target_price,
                        },
                        explanation=f"Submitted {side.value} entry {sizing.quantity} {sym} @ {marketable_entry_price:.2f}; awaiting fill before placing SL",
                    )
                    broker_action = f"SUBMITTED_ENTRY_{marketable_entry_price}"

                decisions.append(
                    StrategyDecision(
                        decision=sig.decision,
                        symbol=sym,
                        side=side,
                        timestamp_ist=now,
                        reason_codes=["ENTRY_SUBMITTED"],
                        explanation=f"Submitted {sig.decision.value} for {actual_qty} shares of {sym}",
                        gates=sig.gates + [sizing.gate, risk_status.gate],
                        quantity=actual_qty,
                        entry_limit=actual_entry,
                        initial_stop=initial_stop,
                        risk_rupees=sizing.trade_risk_rupees,
                        estimated_margin=sizing.notional_value / sizing.effective_leverage,
                        max_notional=sizing.per_symbol_budget,
                        broker_action=broker_action,
                    )
                )
                entries_submitted += 1
                if entries_submitted >= max_new:
                    break
            except Exception as e:
                logger.error("Failed to place order for %s: %s", sym, e)
                decisions.append(
                    StrategyDecision(
                        decision=Decision.HALT,
                        symbol=sym,
                        side=side,
                        timestamp_ist=now,
                        reason_codes=["ORDER_PLACEMENT_EXCEPTION"],
                        explanation=f"Order placement failed: {e}",
                    )
                )

        return decisions


# ---------------------------------------------------------------------------
# Cross-Sectional Radar Merger & Conviction Ranker Loop
# ---------------------------------------------------------------------------




def select_best_entries_for_bar(
    benchmark_bar: BenchmarkSnapshot,
    candidate_bars: dict[str, StockCandidateBar],
    open_positions: dict[str, dict],
    realized_daily_pnl_net: float,
    cfg: dict,
    kite_client=None,
) -> list[tuple[TradeSignal, PositionSizeResult]]:
    """
    Evaluates all stocks on the active radar at 5-minute bar close, ranks passing
    candidates by Conviction Score, and returns the #1 trade to execute.
    """
    from apex3.strategy.regime import BenchmarkSnapshot, MarketRegime, evaluate_market_regime
    from apex3.strategy.signal import StockCandidateBar, TradeSignal, evaluate_candidate_signal
    from apex3.strategy.sizer import PositionSizeResult, calculate_position_size

    acc_cfg = cfg.get("account", {})
    risk_cfg = cfg.get("risk_management", cfg.get("risk", {}))

    # Check Daily Loss Limit (-₹240 on ₹15,000 capital)
    starting_cap = float(acc_cfg.get("starting_capital", 15000.0))
    daily_loss_pct = float(risk_cfg.get("daily_loss_limit_pct", 0.0160))
    if daily_loss_pct > 0.1:
        daily_loss_pct /= 100.0
    max_daily_loss = -1.0 * starting_cap * daily_loss_pct
    if realized_daily_pnl_net <= max_daily_loss:
        return []

    max_slots = int(risk_cfg.get("max_concurrent_positions", 2))
    if len(open_positions) >= max_slots:
        return []

    # Slot 2 Gate: If req_r > 0, only allow 2nd position when Slot 1 is >= req_r in profit
    if len(open_positions) == 1:
        req_r = float(risk_cfg.get("second_slot_requires_slot1_profit_r", 0.0))
        if req_r > 0.0:
            first_pos = next(iter(open_positions.values()))
            if first_pos.get("unrealized_r", 0.0) < req_r:
                return []

    # Step 1: Evaluate NIFTY 50 Regime
    regime = evaluate_market_regime(benchmark_bar, cfg)
    regime_val = regime.value if hasattr(regime, "value") else str(regime)
    if regime_val == "NEUTRAL":
        return []

    # Step 2: Scan all symbols in merged radar
    active_radar = build_active_radar(cfg)
    passing_signals: list[TradeSignal] = []

    for symbol in active_radar:
        if symbol in open_positions or symbol not in candidate_bars:
            continue
        sig = evaluate_candidate_signal(
            bar=candidate_bars[symbol],
            regime=regime,
            active_positions_count=len(open_positions),
            cfg=cfg,
        )
        if sig is not None:
            passing_signals.append(sig)

    if not passing_signals:
        return []

    # Step 3: Sort by highest Conviction Score and take at most max_new_entries_per_bar (1)
    passing_signals.sort(key=lambda s: s.conviction_score, reverse=True)
    max_new = int(risk_cfg.get("max_new_entries_per_bar", 1))

    selected_orders = []
    for sig in passing_signals[:max_new]:
        size_res = calculate_position_size(
            entry_price=sig.entry_price,
            risk_per_share=sig.levels.risk_per_share,
            target_price=sig.levels.target_price,
            cfg=cfg,
            kite_client=kite_client,
        )
        if size_res is not None:
            selected_orders.append((sig, size_res))

    return selected_orders
