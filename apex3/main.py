"""APEX-3 Main Entrypoint and Orchestrator.

Starts the strategy engine in paper mode by default, boots the local
operator dashboard on 127.0.0.1:5000, and manages the simulation loop.

Usage:
    python -m apex3.main
    python -m apex3.main --mode paper --port 5000
    python -m apex3.main --simulate
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from apex3.alerts.notifier import AlertNotifier
from apex3.broker.paper import PaperBroker
from apex3.config import load_config
from apex3.dashboard.app import create_app
from apex3.journal.sqlite_journal import Journal
from apex3.journal.trade_logger import TradeLogger
from apex3.strategy.engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("apex3.main")

_LOCK_HANDLE = None


def acquire_instance_lock(lock_path: str = "apex3_engine.lock") -> bool:
    """Acquire an OS-level exclusive file lock to prevent duplicate engine instances."""
    global _LOCK_HANDLE
    try:
        fh = open(lock_path, "w", encoding="utf-8")
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        _LOCK_HANDLE = fh
        return True
    except Exception as e:
        logger.critical("DUPLICATE INSTANCE PREVENTED: Could not acquire lock on %s (%s). Another APEX-3 instance is already running!", lock_path, e)
        return False


def release_instance_lock() -> None:
    """Release the OS-level process lock if held."""
    global _LOCK_HANDLE
    if _LOCK_HANDLE is not None:
        try:
            if os.name == "nt":
                import msvcrt
                _LOCK_HANDLE.seek(0)
                msvcrt.locking(_LOCK_HANDLE.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            _LOCK_HANDLE.close()
        except Exception:
            pass
        _LOCK_HANDLE = None


def run_app():
    parser = argparse.ArgumentParser(description="APEX-3 Zerodha Intraday System")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=["paper", "live"], help="Operating mode (paper or live)")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode")
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode")
    parser.add_argument("--simulate", action="store_true", help="Run a simulated trading day and print report")
    parser.add_argument("--sync-trades", action="store_true", help="Sync broker trades to info.log and exit")
    parser.add_argument("--host", type=str, default=None, help="Dashboard host interface (e.g. 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Local operator dashboard port")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.live:
        config.mode = "live"
    elif args.paper:
        config.mode = "paper"
    elif args.mode:
        config.mode = args.mode
    logger.info("Initializing APEX-3 version 0.1.0 (Mode: %s)...", config.mode.upper())
    notifier = AlertNotifier()

    if not getattr(args, "sync_trades", False) and not getattr(args, "simulate", False):
        if not acquire_instance_lock():
            sys.exit(1)

    # Initialize Journal & Trade Audit Logger
    journal = Journal(db_path=config.journal.db_path)
    trade_logger = TradeLogger(log_path="info.log", start_date="2026-10-29")

    # Initialize Broker Adapter
    if config.mode == "live":
        from apex3.broker.kite_adapter import KiteAdapter
        from apex3.broker.rate_limiter import RateLimiter
        logger.warning("LIVE TRADING MODE CONFIGURED — Checking safety gates...")
        if not config.secrets.live_trading_enabled:
            logger.critical("LIVE TRADING REQUIRES EXPLICIT FLAG: Set LIVE_TRADING_ENABLED=true in .env")
            if not getattr(args, "simulate", False):
                release_instance_lock()
                sys.exit(1)
        rl_rates = {
            "orders": float(getattr(config.rate_limits, "orders_per_second", 10.0)),
            "quotes": float(getattr(config.rate_limits, "quotes_per_second", 1.0)),
            "historical": float(getattr(config.rate_limits, "historical_per_second", 3.0)),
        }
        rate_limiter = RateLimiter(rates=rl_rates)
        kite_broker = KiteAdapter(
            api_key=config.secrets.kite_api_key,
            access_token=config.secrets.kite_access_token,
            live_enabled=config.secrets.live_trading_enabled,
            rate_limiter=rate_limiter,
        )
        if config.secrets.angel_api_key:
            from apex3.broker.angel_data_provider import AngelDataProvider
            from apex3.broker.hybrid_adapter import HybridBrokerAdapter
            logger.info("Configuring HYBRID MODE: Angel One SmartAPI (Data) + Zerodha (Orders)")
            angel_data = AngelDataProvider(
                api_key=config.secrets.angel_api_key,
                client_code=config.secrets.angel_client_code,
                pin=config.secrets.angel_pin,
                totp_secret=config.secrets.angel_totp_secret,
                rate_limiter=rate_limiter,
            )
            broker = HybridBrokerAdapter(data_provider=angel_data, execution_broker=kite_broker)
        else:
            broker = kite_broker

        # Pre-flight session check
        session_valid_at_boot = broker.is_session_valid()
        if session_valid_at_boot:
            logger.info("SUCCESS: Broker live session(s) ACTIVE and authenticated.")
        else:
            msg = (
                "STANDBY MODE: Zerodha daily session token is expired or awaiting login. "
                "Starting Operator Dashboard & /login/kite OAuth listener in standby mode..."
            )
            logger.warning(msg)
            notifier.notify_connection_lost("Zerodha/AngelOne", msg)
    else:
        logger.info("Operating in PAPER TRADING mode. Real orders are impossible.")
        paper_broker = PaperBroker(starting_margin=config.paper_starting_capital)
        if config.secrets.angel_api_key and not getattr(args, "simulate", False):
            from apex3.broker.angel_data_provider import AngelDataProvider
            from apex3.broker.hybrid_adapter import HybridBrokerAdapter
            from apex3.broker.rate_limiter import RateLimiter
            rl_rates = {
                "orders": float(getattr(config.rate_limits, "orders_per_second", 10.0)),
                "quotes": float(getattr(config.rate_limits, "quotes_per_second", 5.0)),
                "historical": float(getattr(config.rate_limits, "historical_per_second", 3.0)),
            }
            rate_limiter = RateLimiter(rates=rl_rates)
            angel_data = AngelDataProvider(
                api_key=config.secrets.angel_api_key,
                client_code=config.secrets.angel_client_code,
                pin=config.secrets.angel_pin,
                totp_secret=config.secrets.angel_totp_secret,
                rate_limiter=rate_limiter,
            )
            if angel_data.is_session_valid():
                logger.info("LIVE-DATA PAPER MODE ACTIVE: Streaming live NSE feed from Angel One into PaperBroker.")
                broker = HybridBrokerAdapter(data_provider=angel_data, execution_broker=paper_broker)
            else:
                logger.warning("Angel One live feed unavailable; falling back to standalone offline PaperBroker.")
                broker = paper_broker
        else:
            broker = paper_broker

    # Sync offline/startup trades from broker into info.log
    try:
        new_synced = trade_logger.sync_from_broker(broker)
        if new_synced > 0:
            logger.info("Trade audit log synced: %d new trade execution(s) recorded in info.log", new_synced)
        else:
            logger.info("Trade audit log (info.log) is up to date.")
    except Exception as e:
        logger.warning("Could not sync trade audit log on startup: %s", e)

    if getattr(args, "sync_trades", False):
        print(f"Trade sync complete: info.log is up to date. ({len(trade_logger._recorded_trade_ids)} IDs tracked)")
        return

    # Initialize Strategy Engine
    engine = StrategyEngine(config=config, broker=broker, journal=journal, trade_logger=trade_logger)

    if config.mode == "live":
        if broker.is_session_valid():
            from apex3.reconciliation.reconciler import Reconciler
            logger.info("Executing mandatory pre-flight account state and position reconciliation audit...")
            reconciler = Reconciler(
                broker=broker,
                state_machine=engine.state_machine,
                journal=journal,
                session_tag=engine.session_tag,
                config=config,
            )
            radar_cfg = getattr(config, "raw_dict", {}).get("radar", {}) if hasattr(config, "raw_dict") else {}
            watched = list(radar_cfg.get("core_watchlist", [])) + list(radar_cfg.get("extended_pool", []))
            if not watched:
                watched = list(config.watchlist.symbols) + list(config.watchlist.scanner_universe)
            rec_status = reconciler.audit_startup_state(watched_symbols=watched, reattach_stops=True)
            if not rec_status.is_clean:
                msg = (
                    f"CRITICAL PRE-FLIGHT AUDIT WARNING: {rec_status.message}. "
                    "Engine paused until conflicting external positions/orders are resolved."
                )
                logger.critical(msg)
                engine.is_paused = True
                engine.pause_reason = "UNKNOWN_MANUAL_POSITIONS"
            else:
                logger.info("PRE-FLIGHT AUDIT PASSED: Account state verified. Flat=%s", rec_status.is_flat)
                try:
                    backfill_res = engine.reconcile_and_backfill_late_start()
                    if backfill_res.get("recovered_positions"):
                        notifier.notify_unexpected_restart(config.mode, backfill_res["recovered_positions"])
                except Exception as bfe:
                    logger.warning("Late-start backfill warning: %s", bfe)
        else:
            engine.is_paused = True
            engine.pause_reason = "BROKER_SESSION_INVALID"
            logger.info("Engine paused in STANDBY (BROKER_SESSION_INVALID). Log in via /login/kite to activate live trading.")

    if args.simulate:
        logger.info("Running simulated trading session...")
        # Run a simulated sequence through the 12-step pipeline
        sim_time = datetime.now(IST).replace(hour=9, minute=35, second=0)
        decisions = engine.evaluate_step(sim_time)
        print("\n=== SIMULATION CYCLE RESULT ===")
        for d in decisions:
            print(f"Symbol: {d.symbol} | Decision: {d.decision.value} | Reason: {d.explanation}")
            print(f"Broker Action: {d.broker_action}")
        print("===============================\n")
        return

    # Start Autonomous Engine Loop Thread (runs strategy independently of UI)
    import threading

    def _trading_loop():
        logger.info("Autonomous Strategy Execution Thread started (Interval: dynamic 0.5s in-flight / 3.0s idle).")
        while True:
            try:
                now = datetime.now(IST)
                engine.evaluate_step(now)
            except Exception as e:
                logger.error("Error in autonomous trading loop: %s", e)
            sleep_time = 0.5 if getattr(engine, "has_in_flight_orders", lambda: False)() else 3.0
            time.sleep(sleep_time)

    loop_thread = threading.Thread(target=_trading_loop, daemon=True, name="Apex3TradingLoop")
    loop_thread.start()

    # Create & Run Flask Local/Cloud Dashboard
    host = args.host or os.environ.get("APEX3_DASHBOARD_HOST") or config.dashboard.host
    port = args.port or config.dashboard.port
    try:
        object.__setattr__(config.dashboard, "host", host)
        object.__setattr__(config.dashboard, "port", port)
    except Exception:
        pass
    dashboard_app = create_app(engine, config, journal)
    logger.info(
        "Starting APEX-3 Operator Dashboard at http://%s:%d...",
        host,
        port,
    )
    dashboard_app.run(
        host=host,
        port=port,
        debug=False,
    )


if __name__ == "__main__":
    run_app()
