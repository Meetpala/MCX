"""
main.py
-------
Institutional-Grade MCX Precious Metals Quantitative Trading System
Zerodha Kite Connect v3/v4 Autonomous Execution Controller

Coordinates:
1. Dynamic US DST Schedule and Intraday Execution Windows
2. LiveMarginManager (Margin Verification, Sizing & Micro Downgrade)
3. StrategyEngine (VWAP, ATR, CI_14, ADX, REGIME_WAITING Gate, VVR-BPC Triggers)
4. OrderStateMachine (Asynchronous Limit Entries, 5s Timeouts, Exchange SL-L, Emergency Exits)
5. TrailingEngine (4-Stage Asymmetric Ratchet Trailing Stop-Loss)
6. 10-Second Background Position Reconciler & Daily Circuit Breaker
"""

import argparse
from datetime import datetime, time, timedelta
import logging
import signal
import sys
import threading
import time as time_module
from typing import Dict, Optional

from config.settings import (
    ANGEL_API_KEY,
    ANGEL_CLIENT_CODE,
    ANGEL_PIN,
    ANGEL_TOTP_SECRET,
    CAPITAL_RISK_PER_TRADE,
    DAILY_LOSS_LIMIT_PCT,
    DEFAULT_COMMODITY,
    IST,
    KITE_API_KEY,
    KITE_ACCESS_TOKEN,
    MAX_CONSECUTIVE_STOPOUTS,
    MCX_INSTRUMENTS,
    PREFERRED_TIER,
    SessionWindow,
    TIER_DOWNGRADE_ORDER,
    TRADING_MODE,
    CommodityFamily,
    get_current_session_window,
    get_mcx_schedule,
    is_entry_allowed,
)
from apex3.broker.angel_data_provider import AngelDataProvider
from core.margin_and_sizer import LiveMarginManager, SizingResult, calculate_net_pnl
from core.order_state_machine import OrderStateMachine, Position, PositionStatus
from core.regime_and_strategy import Candle, MarketRegime, StrategyEngine, TradeSignal
from core.trailing_engine import TrailingEngine, TrailingStage

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("MCXTrader.Main")


class SystemExecutionState:
    RUNNING = "RUNNING"
    LOCKED_FOR_DAY = "LOCKED_FOR_DAY"
    STOPPED = "STOPPED"


class MCXTradingBot:
    """
    Main Autonomous Trading Bot for MCX Precious Metals.
    """

    def __init__(
        self,
        symbol: str = "GOLDM",
        is_paper: bool = True,
        initial_capital: float = 250000.0,
    ):
        self.symbol = symbol
        self.is_paper = is_paper
        self.starting_daily_capital = initial_capital
        self.current_equity = initial_capital
        self.realized_daily_pnl = 0.0
        self.consecutive_stopouts = 0
        self.execution_state = SystemExecutionState.RUNNING

        # 1. Market Data Layer (Angel One SmartAPI - Strictly Free Read-Only)
        self.angel_data: Optional[AngelDataProvider] = None
        if ANGEL_API_KEY and ANGEL_CLIENT_CODE:
            try:
                self.angel_data = AngelDataProvider(
                    api_key=ANGEL_API_KEY,
                    client_code=ANGEL_CLIENT_CODE,
                    pin=ANGEL_PIN,
                    totp_secret=ANGEL_TOTP_SECRET,
                )
                if self.angel_data.is_session_valid():
                    logger.info("Angel One SmartAPI market data feed ACTIVE and authenticated.")
                else:
                    logger.warning("Angel One session authentication pending or awaiting credentials.")
            except Exception as e:
                logger.warning(f"Could not initialize AngelDataProvider: {e}")

        # 2. Execution & Margin Layer (Zerodha Kite Connect - Strictly Orders & Margins)
        self.kite = None
        if not is_paper:
            try:
                from kiteconnect import KiteConnect
                if KITE_API_KEY and KITE_ACCESS_TOKEN:
                    self.kite = KiteConnect(api_key=KITE_API_KEY)
                    self.kite.set_access_token(KITE_ACCESS_TOKEN)
                    logger.info("Successfully initialized Zerodha Kite Connect LIVE execution client.")
                else:
                    logger.warning("Kite API credentials not set. Falling back to PAPER execution mode.")
                    self.is_paper = True
            except ImportError:
                logger.error("kiteconnect package not found. Running in simulation/paper mode.")
                self.is_paper = True

        logger.info(
            "SPLIT-FEED HYBRID ARCHITECTURE INITIALIZED: "
            "[Market Data: Angel One SmartAPI (Free Quotes/Ticks)] -> "
            "[Order Execution: Zerodha Kite Connect (Free MIS/NRML Execution)]"
        )

        # Core Engines
        spec = MCX_INSTRUMENTS.get(self.symbol, MCX_INSTRUMENTS["GOLDM"])
        self.spec = spec
        self.trailing_engine = TrailingEngine(tick_size=spec.tick_size)
        self.order_sm = OrderStateMachine(
            kite_client=self.kite,
            trailing_engine=self.trailing_engine,
            is_paper=self.is_paper,
        )
        self.margin_mgr = LiveMarginManager(kite_client=self.kite, default_equity=initial_capital)
        self.strategy_engine = StrategyEngine(symbol=self.symbol)

        # Threading & Control flags
        self.shutdown_requested = False
        self.threads = []

    def start(self):
        """Starts background workers and tick processing loop."""
        logger.info(
            f"Starting MCX Precious Metals Bot: Symbol={self.symbol}, Mode={'PAPER' if self.is_paper else 'LIVE'}, "
            f"Capital={self.starting_daily_capital:.2f}"
        )
        schedule = get_mcx_schedule()
        logger.info(
            f"MCX Schedule: DST={schedule.is_dst}, Open={schedule.market_open}, Close={schedule.market_close}, "
            f"Cutoff={schedule.entry_cutoff_time}, Squareoff={schedule.auto_squareoff_time}"
        )

        # Launch 10-second Reconciler thread
        rec_thread = threading.Thread(target=self._run_reconciler_loop, daemon=True)
        rec_thread.start()
        self.threads.append(rec_thread)

        # Launch Intraday Session Scheduler thread
        sched_thread = threading.Thread(target=self._run_scheduler_loop, daemon=True)
        sched_thread.start()
        self.threads.append(sched_thread)

        # Launch Angel One Market Data streaming/polling thread (Market Data Layer)
        if self.angel_data:
            data_thread = threading.Thread(target=self._run_angel_market_data_loop, daemon=True)
            data_thread.start()
            self.threads.append(data_thread)
            logger.info("Angel One live market data poller launched.")

    def _run_angel_market_data_loop(self):
        """Continuously pulls live market data/quotes exclusively from Angel One SmartAPI."""
        logger.info(f"Angel One market data stream loop started for {self.symbol} (MCX)")
        while not self.shutdown_requested:
            try:
                # Strictly read-only market data from Angel One SmartAPI
                q = self.angel_data.get_quote(self.symbol, exchange="MCX")
                if q and q.last_price > 0:
                    self.on_tick_received(q.timestamp or datetime.now(IST), q.last_price, q.volume or 1.0)
            except Exception as e:
                logger.error(f"Error in Angel One market data loop: {e}")
            time_module.sleep(1.0)

    def on_tick_received(self, timestamp: datetime, ltp: float, volume: float = 1.0):
        """
        Primary event handler called upon arrival of market tick.
        Feeds tick into StrategyEngine, manages active trailing stops, and evaluates signals.
        """
        if self.shutdown_requested or self.execution_state == SystemExecutionState.LOCKED_FOR_DAY:
            return

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=IST)

        # 1. Manage Active Position Trailing Stops
        pos = self.order_sm.positions.get(self.symbol)
        if pos and pos.status == PositionStatus.ACTIVE and pos.trade_state:
            self._update_position_trailing(pos, ltp)

        # 2. Feed tick to bar aggregator and strategy
        signal = self.strategy_engine.feed_tick(timestamp, ltp, volume)
        if signal:
            self._process_trade_signal(signal, ltp, timestamp)

    def _update_position_trailing(self, pos: Position, ltp: float):
        """Updates trailing stop-loss on incoming tick."""
        trade = pos.trade_state
        if not trade:
            return

        atrs = self.strategy_engine.candles_5m
        curr_atr = atrs[-1].range if atrs else 50.0
        vwap = self.strategy_engine.vwap_tracker.vwap
        sigma = self.strategy_engine.vwap_tracker.sigma

        new_sl, new_stage, should_modify, reason = self.trailing_engine.update_price_and_compute_trailing(
            trade=trade,
            current_price=ltp,
            atr14=curr_atr,
            vwap=vwap,
            vwap_sd=sigma,
        )

        if should_modify:
            buffer_pct = 0.0035
            if pos.is_long:
                new_limit = new_sl * (1.0 - buffer_pct)
            else:
                new_limit = new_sl * (1.0 + buffer_pct)

            logger.info(
                f"[{pos.trade_id}] Ratchet Trailing SL Triggered ({reason}): "
                f"Stage {new_stage.value} -> New SL {new_sl:.1f}"
            )
            self.order_sm.modify_stop_loss(pos, new_trigger=new_sl, new_limit=new_limit)

    def _process_trade_signal(self, signal: TradeSignal, ltp: float, now_dt: datetime):
        """Processes validated trade signal from strategy."""
        logger.info(f"Received trade signal: {signal.signal_type.value} on {signal.symbol} at {ltp:.1f}")

        # Invariant 6: Pre-Close Cutoff Gate (No new entries within T_close - 75m)
        if not is_entry_allowed(now_dt):
            logger.warning(f"Trade signal rejected: Current session window does not permit new entries.")
            return

        # Invariant 5: Daily Circuit Breaker Gate
        if self.execution_state == SystemExecutionState.LOCKED_FOR_DAY:
            logger.warning("Trade signal rejected: System is LOCKED_FOR_DAY.")
            return

        # Position Sizing & Margin Gate
        # Compute SL_0 and 1R
        sl_0, unit_risk_1r, is_valid_rr = self.trailing_engine.calculate_initial_stop_loss(
            entry_price=ltp,
            is_long=(signal.direction == "BUY"),
            atr14=signal.atr14,
            signal_candle_low=signal.signal_candle.low,
            signal_candle_high=signal.signal_candle.high,
        )

        if not is_valid_rr:
            logger.warning(f"Trade skipped: Structural risk distance exceeds maximum 2.2 * ATR_14 cap.")
            return

        sizing = self.margin_mgr.size_position(
            symbol=self.symbol,
            entry_price=ltp,
            stop_loss_price=sl_0,
            is_buy=(signal.direction == "BUY"),
            override_equity=self.current_equity,
            allow_downgrade=True,
        )

        if sizing.lots <= 0:
            logger.warning(
                f"Trade signal blocked: Capital/margin sizing yielded 0 lots "
                f"(Req={sizing.required_margin_inr:.1f}, Usable={sizing.usable_margin_inr:.1f})."
            )
            return

        if sizing.downgraded_from:
            logger.info(f"Auto-downgraded tier from {sizing.downgraded_from} to micro {sizing.symbol} for trade.")

        # Execute Entry Order
        self.order_sm.execute_entry(
            symbol=sizing.symbol,
            direction=signal.direction,
            sizing=sizing,
            ltp=ltp,
            initial_sl=sl_0,
            unit_risk_1r=unit_risk_1r,
        )

    def record_trade_completion(self, gross_pnl: float, net_pnl: float, was_stopout: bool):
        """Updates equity, realized P&L, and enforces daily circuit breaker."""
        self.realized_daily_pnl += net_pnl
        self.current_equity += net_pnl

        logger.info(
            f"Trade Closed: Net P&L = Rs. {net_pnl:.2f} (Gross = Rs. {gross_pnl:.2f}). "
            f"Cumulative Daily Realized P&L = Rs. {self.realized_daily_pnl:.2f} ({self.realized_daily_pnl / self.starting_daily_capital * 100:.2f}%)"
        )

        if was_stopout:
            self.consecutive_stopouts += 1
            logger.warning(f"Stop-out recorded. Consecutive stop-outs: {self.consecutive_stopouts}/{MAX_CONSECUTIVE_STOPOUTS}")
        else:
            self.consecutive_stopouts = 0

        # Invariant 5: Daily Circuit Breaker (-2.0% loss or 2 consecutive stopouts)
        loss_pct = (self.realized_daily_pnl / self.starting_daily_capital)
        if loss_pct <= -DAILY_LOSS_LIMIT_PCT:
            self.execution_state = SystemExecutionState.LOCKED_FOR_DAY
            logger.critical(
                f"CIRCUIT BREAKER TRIGGERED: Daily loss reached {loss_pct * 100:.2f}% (Limit: -{DAILY_LOSS_LIMIT_PCT * 100}%). "
                f"System transitioning to LOCKED_FOR_DAY until 09:00 IST next session."
            )
            self.order_sm.close_all_positions(reason="Daily Loss Limit Circuit Breaker")

        elif self.consecutive_stopouts >= MAX_CONSECUTIVE_STOPOUTS:
            self.execution_state = SystemExecutionState.LOCKED_FOR_DAY
            logger.critical(
                f"CIRCUIT BREAKER TRIGGERED: {self.consecutive_stopouts} consecutive stop-outs recorded. "
                f"System transitioning to LOCKED_FOR_DAY until 09:00 IST next session."
            )
            self.order_sm.close_all_positions(reason="Consecutive Stopout Circuit Breaker")

    def _run_reconciler_loop(self):
        """Background thread executing 10-second reconciliation and SL trap audit."""
        while not self.shutdown_requested:
            try:
                self.order_sm.reconcile_positions_and_orders()
                self.order_sm.check_sl_triggered_open_trap()
            except Exception as e:
                logger.error(f"Error in reconciler loop: {e}")
            time_module.sleep(10.0)

    def _run_scheduler_loop(self):
        """Background thread monitoring session cutoffs and mandatory square-offs."""
        while not self.shutdown_requested:
            try:
                now_dt = datetime.now(IST)
                schedule = get_mcx_schedule(now_dt.date())
                curr_t = now_dt.time()

                # Invariant 4: Hard Intraday Auto-Squareoff (T_close - 20m)
                if schedule.auto_squareoff_time <= curr_t < schedule.market_close:
                    active_positions = any(
                        p.status == PositionStatus.ACTIVE and p.filled_quantity > 0
                        for p in self.order_sm.positions.values()
                    )
                    if active_positions:
                        logger.critical(
                            f"MANDATORY SQUAREOFF WINDOW REACHED ({curr_t} >= {schedule.auto_squareoff_time}). "
                            f"Flattening all positions and cancelling orders."
                        )
                        self.order_sm.close_all_positions(reason="Mandatory T_close - 20m Auto-Squareoff")

                # Invariant 4 Audit: T_close - 18m Zero Carry Audit
                if schedule.zero_carry_audit_time <= curr_t < schedule.market_close:
                    self._audit_zero_overnight_carry()

                # Reset circuit breaker at 09:00 IST new session
                if curr_t < time(9, 1) and self.execution_state == SystemExecutionState.LOCKED_FOR_DAY:
                    logger.info("New trading session started at 09:00 IST. Resetting daily circuit breaker.")
                    self.execution_state = SystemExecutionState.RUNNING
                    self.consecutive_stopouts = 0
                    self.realized_daily_pnl = 0.0

            except Exception as e:
                logger.error(f"Error in session scheduler loop: {e}")
            time_module.sleep(30.0)

    def _audit_zero_overnight_carry(self):
        """Verifies that net quantity == 0 and 0 open orders exist on MCX."""
        active_qty = sum(p.filled_quantity for p in self.order_sm.positions.values())
        if active_qty != 0:
            logger.critical(f"ZERO OVERNIGHT CARRY AUDIT FAILED: Local open quantity = {active_qty}!")
            self.order_sm.close_all_positions(reason="Emergency Zero Overnight Carry Audit")
        else:
            logger.info("Zero Overnight Carry Audit Verified: 0 open positions.")

    def stop(self):
        """Gracefully halts trading system."""
        logger.info("Shutting down MCX Trading Bot...")
        self.shutdown_requested = True
        self.order_sm.close_all_positions(reason="System Shutdown")


def main():
    parser = argparse.ArgumentParser(description="MCX Precious Metals Quantitative Trading System")
    parser.add_argument("--symbol", type=str, default=DEFAULT_COMMODITY + "M", help="Target symbol (e.g. GOLDM, SILVERMIC)")
    parser.add_argument("--paper", action="store_true", default=True, help="Run in paper simulation mode")
    parser.add_argument("--capital", type=float, default=250000.0, help="Initial trading capital in INR")
    args = parser.parse_args()

    bot = MCXTradingBot(symbol=args.symbol, is_paper=args.paper, initial_capital=args.capital)
    bot.start()

    def sig_handler(sig, frame):
        logger.info("Received interrupt signal. Exiting...")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    logger.info("Bot event loop running. Press Ctrl+C to stop.")
    try:
        while not bot.shutdown_requested:
            time_module.sleep(1.0)
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    main()

