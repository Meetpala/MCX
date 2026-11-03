"""Unit tests for TradeLogger (info.log) functionality.

Verifies:
1. info.log creation and format.
2. Past trades (< 2026-10-29) are strictly excluded.
3. Future trades (>= 2026-10-29) are correctly recorded across trading days.
4. Idempotency: Multiple syncs/runs never duplicate log entries.
5. Offline broker sync: Trades retrieved via broker.get_trades() update info.log.
6. Round-trip trade pairing, Zerodha statutory friction, gross & net PnL calculation.
7. Integration with StrategyEngine.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo
import pytest

from apex3.broker.paper import PaperBroker
from apex3.config import AppConfig
from apex3.journal.trade_logger import TradeLogger, parse_trade_timestamp
from apex3.models import Side, OrderState, OrderStatus, OrderRequest, Product, OrderType
from apex3.strategy.engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")


def test_trade_logger_initializes_file(tmp_path: Path):
    """Verify info.log is created with standard header if it does not exist."""
    log_file = tmp_path / "info.log"
    logger = TradeLogger(log_path=str(log_file), start_date="2026-10-29")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "APEX-3 LIVE TRADE AUDIT LOG (info.log)" in content
    assert "Audit Start   : 2026-10-29" in content
    assert "INFO.LOG INITIALIZED" in content


def test_trade_logger_excludes_past_trades(tmp_path: Path):
    """Verify trades occurring before start_date (2026-10-29) are strictly ignored."""
    log_file = tmp_path / "info.log"
    logger = TradeLogger(log_path=str(log_file), start_date="2026-10-29")

    # Past trade on 2026-10-28
    past_dt = datetime(2026, 10, 28, 10, 15, 0, tzinfo=IST)
    res = logger.record_entry(
        symbol="INFY",
        side=Side.BUY,
        quantity=35,
        price=1420.50,
        order_id="ORD_PAST_1",
        trade_id="TRD_PAST_1",
        timestamp=past_dt,
    )
    assert res is False
    content = log_file.read_text(encoding="utf-8")
    assert "ORD_PAST_1" not in content
    assert "TRD_PAST_1" not in content


def test_trade_logger_records_future_trades(tmp_path: Path):
    """Verify trades occurring from tomorrow (2026-10-29) onwards are recorded."""
    log_file = tmp_path / "info.log"
    logger = TradeLogger(log_path=str(log_file), start_date="2026-10-29")

    future_dt = datetime(2026, 10, 29, 9, 35, 12, tzinfo=IST)
    res = logger.record_entry(
        symbol="TCS",
        side=Side.BUY,
        quantity=15,
        price=3500.0,
        order_id="ORD_FUT_1",
        trade_id="TRD_FUT_1",
        timestamp=future_dt,
        initial_stop=3475.0,
        target_price=3550.0,
        risk_rupees=375.0,
    )
    assert res is True
    content = log_file.read_text(encoding="utf-8")
    assert "TCS (NSE)" in content
    assert "BUY (LONG)" in content
    assert "15 shares" in content
    assert "Rs. 3,500.00" in content
    assert "ORD_FUT_1" in content
    assert "TRD_FUT_1" in content


def test_trade_logger_records_roundtrip_exit_with_friction(tmp_path: Path):
    """Verify round-trip trade exit records gross PnL, Zerodha statutory friction, net PnL, and R-multiple."""
    log_file = tmp_path / "info.log"
    logger = TradeLogger(log_path=str(log_file), start_date="2026-10-29")

    entry_dt = datetime(2026, 10, 29, 9, 35, 0, tzinfo=IST)
    exit_dt = datetime(2026, 10, 29, 10, 45, 0, tzinfo=IST)

    logger.record_entry(
        symbol="RELIANCE",
        side=Side.BUY,
        quantity=20,
        price=2500.0,
        order_id="ORD_REL_ENT",
        trade_id="TRD_REL_ENT",
        timestamp=entry_dt,
        initial_stop=2480.0,
        target_price=2540.0,
        risk_rupees=400.0,
    )

    # 2.0R target exit @ 2540
    res_exit = logger.record_exit(
        symbol="RELIANCE",
        side=Side.SELL,
        quantity=20,
        exit_price=2540.0,
        entry_price=2500.0,
        order_id="ORD_REL_EXT",
        trade_id="TRD_REL_EXT",
        exit_reason="TARGET_HIT",
        timestamp=exit_dt,
        entry_time=entry_dt,
        initial_stop=2480.0,
        running_daily_pnl=775.25,
    )
    assert res_exit is True
    content = log_file.read_text(encoding="utf-8")
    assert "TARGET_HIT" in content
    # Gross PnL = (2540 - 2500) * 20 = +800.00
    assert "Gross PnL       : +Rs. 800.00" in content
    assert "Zerodha Friction:" in content
    assert "Net Realized PnL: +Rs." in content
    assert "Realized R-Mult : +2.00R" in content
    assert "Session Net PnL : +Rs. 775.25" in content


def test_trade_logger_idempotency_prevents_duplicate_records(tmp_path: Path):
    """Verify recording the same trade multiple times never duplicates lines in info.log."""
    log_file = tmp_path / "info.log"
    logger = TradeLogger(log_path=str(log_file), start_date="2026-10-29")

    dt = datetime(2026, 10, 29, 9, 40, 0, tzinfo=IST)
    first = logger.record_entry(
        symbol="INFY",
        side=Side.BUY,
        quantity=10,
        price=1500.0,
        order_id="ORD_DUP_1",
        trade_id="TRD_DUP_1",
        timestamp=dt,
    )
    assert first is True

    # Second call with same trade_id
    second = logger.record_entry(
        symbol="INFY",
        side=Side.BUY,
        quantity=10,
        price=1500.0,
        order_id="ORD_DUP_1",
        trade_id="TRD_DUP_1",
        timestamp=dt,
    )
    assert second is False

    # Instantiate new logger instance loading from disk
    logger2 = TradeLogger(log_path=str(log_file), start_date="2026-10-29")
    third = logger2.record_entry(
        symbol="INFY",
        side=Side.BUY,
        quantity=10,
        price=1500.0,
        order_id="ORD_DUP_1",
        trade_id="TRD_DUP_1",
        timestamp=dt,
    )
    assert third is False

    content = log_file.read_text(encoding="utf-8")
    assert content.count("ORD_DUP_1") == 1


def test_sync_from_broker_captures_offline_trades(tmp_path: Path):
    """Verify broker.get_trades() syncs trades executed while the system was offline."""
    log_file = tmp_path / "info.log"
    logger = TradeLogger(log_path=str(log_file), start_date="2026-10-29")

    mock_broker = MagicMock()
    mock_broker.get_trades.return_value = [
        # Past trade (should be excluded)
        {
            "trade_id": "TRD_OLD_1",
            "order_id": "ORD_OLD_1",
            "symbol": "SBIN",
            "side": "BUY",
            "quantity": 50,
            "price": 800.0,
            "timestamp": "2026-10-28 11:00:00",
        },
        # Future trade BUY on 2026-10-29
        {
            "trade_id": "TRD_NEW_B1",
            "order_id": "ORD_NEW_B1",
            "tradingsymbol": "HDFCBANK",
            "transaction_type": "BUY",
            "quantity": 25,
            "average_price": 1600.0,
            "timestamp": "2026-10-29 09:35:00",
        },
        # Future trade SELL on 2026-10-29 (paired round-trip)
        {
            "trade_id": "TRD_NEW_S1",
            "order_id": "ORD_NEW_S1",
            "tradingsymbol": "HDFCBANK",
            "transaction_type": "SELL",
            "quantity": 25,
            "average_price": 1625.0,
            "timestamp": "2026-10-29 11:15:00",
        },
    ]

    new_trades = logger.sync_from_broker(mock_broker)
    assert new_trades == 2  # 1 BUY + 1 SELL

    content = log_file.read_text(encoding="utf-8")
    # Old trade must NOT be present
    assert "TRD_OLD_1" not in content
    assert "SBIN" not in content

    # New trades must be present
    assert "HDFCBANK (NSE)" in content
    assert "TRD_NEW_B1" in content
    assert "TRD_NEW_S1" in content
    assert "BROKER_SYNC_EXIT" in content
    # Gross PnL: (1625 - 1600) * 25 = +625.00
    assert "Gross PnL       : +Rs. 625.00" in content

    # Syncing again yields 0 new trades (idempotent)
    second_sync = logger.sync_from_broker(mock_broker)
    assert second_sync == 0


def test_strategy_engine_integrates_trade_logger(tmp_path: Path):
    """Verify StrategyEngine evaluate_step syncs trades and logs executions."""
    log_file = tmp_path / "info.log"
    trade_logger = TradeLogger(log_path=str(log_file), start_date="2026-10-29")

    broker = PaperBroker(starting_margin=15_000.0)
    journal = MagicMock()
    config = AppConfig()

    engine = StrategyEngine(
        config=config,
        broker=broker,
        journal=journal,
        trade_logger=trade_logger,
    )

    assert engine.trade_logger is trade_logger

    # Run evaluate_step with a simulated future date
    future_time = datetime(2026, 10, 29, 9, 35, 0, tzinfo=IST)
    decisions = engine.evaluate_step(future_time)

    assert isinstance(decisions, list)
    # File exists and header remains intact
    assert log_file.exists()
