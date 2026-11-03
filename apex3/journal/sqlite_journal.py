"""APEX-3 SQLite Event and Trade Journal.

Records all strategy decisions, orders, fills, stop changes, warnings,
and daily summaries for reproducibility and audit.

CRITICAL: Never log credentials, access tokens, API secrets, or TOTP codes.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class Journal:
    """SQLite-backed event and trade journal.

    All events are written with IST timestamps. The journal is append-only
    during normal operation — events are never deleted or modified.
    """

    def __init__(self, db_path: str = "apex3_journal.db"):
        """Initialize journal and create tables if needed.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_tables(self) -> None:
        """Create journal tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ist TEXT NOT NULL,
                session_date TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                side TEXT,
                data TEXT,
                reason_codes TEXT,
                explanation TEXT
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ist TEXT NOT NULL,
                session_date TEXT NOT NULL,
                order_id TEXT,
                strategy_tag TEXT,
                symbol TEXT NOT NULL,
                exchange TEXT DEFAULT 'NSE',
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                product TEXT DEFAULT 'MIS',
                price REAL,
                trigger_price REAL,
                status TEXT NOT NULL,
                filled_quantity INTEGER DEFAULT 0,
                average_price REAL DEFAULT 0,
                modification_count INTEGER DEFAULT 0,
                status_message TEXT,
                parent_order_id TEXT
            );

            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ist TEXT NOT NULL,
                session_date TEXT NOT NULL,
                order_id TEXT NOT NULL,
                trade_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                is_partial INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS stop_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ist TEXT NOT NULL,
                session_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                previous_stop REAL,
                new_stop REAL,
                reason TEXT,
                r_multiple REAL,
                best_favorable_price REAL,
                modification_number INTEGER
            );

            CREATE TABLE IF NOT EXISTS warnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ist TEXT NOT NULL,
                session_date TEXT NOT NULL,
                severity TEXT NOT NULL,
                code TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_date TEXT NOT NULL UNIQUE,
                mode TEXT NOT NULL,
                symbols_watched TEXT,
                trades_taken INTEGER DEFAULT 0,
                trades_won INTEGER DEFAULT 0,
                trades_lost INTEGER DEFAULT 0,
                gross_pnl REAL DEFAULT 0,
                total_costs REAL DEFAULT 0,
                net_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                max_favorable_excursion REAL DEFAULT 0,
                max_adverse_excursion REAL DEFAULT 0,
                risk_events TEXT,
                halts INTEGER DEFAULT 0,
                data TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_events_session
                ON events(session_date, event_type);
            CREATE INDEX IF NOT EXISTS idx_orders_session
                ON orders(session_date, symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_order_id
                ON orders(order_id);
            CREATE INDEX IF NOT EXISTS idx_fills_order
                ON fills(order_id);
            CREATE INDEX IF NOT EXISTS idx_stops_session
                ON stop_changes(session_date, symbol);
        """)
        conn.commit()

    def _now_ist(self) -> str:
        """Current IST timestamp as ISO string."""
        return datetime.now(IST).isoformat()

    def _today_ist(self) -> str:
        """Current IST date as YYYY-MM-DD string."""
        return datetime.now(IST).date().isoformat()

    # ----- Event logging -----

    def log_event(
        self,
        event_type: str,
        symbol: str = "",
        side: str = "",
        data: Optional[dict[str, Any]] = None,
        reason_codes: Optional[list[str]] = None,
        explanation: str = "",
    ) -> int:
        """Log a strategy event (decision, state change, etc.).

        Args:
            event_type: e.g., DECISION, REGIME_CHECK, SIGNAL, GATE_FAIL, HALT
            symbol: Trading symbol, if applicable.
            side: BUY/SELL, if applicable.
            data: Arbitrary structured data for the event.
            reason_codes: Machine-readable reason codes.
            explanation: Human-readable explanation.

        Returns:
            Row ID of the inserted event.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO events
               (timestamp_ist, session_date, event_type, symbol, side,
                data, reason_codes, explanation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._now_ist(),
                self._today_ist(),
                event_type,
                symbol,
                side,
                json.dumps(data) if data else None,
                json.dumps(reason_codes) if reason_codes else None,
                explanation,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_order(
        self,
        order_id: str,
        strategy_tag: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        status: str,
        price: float = 0.0,
        trigger_price: float = 0.0,
        filled_quantity: int = 0,
        average_price: float = 0.0,
        modification_count: int = 0,
        status_message: str = "",
        parent_order_id: str = "",
    ) -> int:
        """Log an order event (placement, modification, status change).

        WARNING: Order acceptance does NOT prove exchange execution.
        Always reconcile via order history and trades.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO orders
               (timestamp_ist, session_date, order_id, strategy_tag, symbol,
                side, quantity, order_type, price, trigger_price, status,
                filled_quantity, average_price, modification_count,
                status_message, parent_order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._now_ist(),
                self._today_ist(),
                order_id,
                strategy_tag,
                symbol,
                side,
                quantity,
                order_type,
                price,
                trigger_price,
                status,
                filled_quantity,
                average_price,
                modification_count,
                status_message,
                parent_order_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        trade_id: str = "",
        is_partial: bool = False,
    ) -> int:
        """Log a confirmed fill (from broker trade data, not order acceptance)."""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO fills
               (timestamp_ist, session_date, order_id, trade_id, symbol,
                side, quantity, price, is_partial)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._now_ist(),
                self._today_ist(),
                order_id,
                trade_id,
                symbol,
                side,
                quantity,
                price,
                1 if is_partial else 0,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_stop_change(
        self,
        symbol: str,
        side: str,
        previous_stop: float,
        new_stop: float,
        reason: str = "",
        r_multiple: float = 0.0,
        best_favorable_price: float = 0.0,
        modification_number: int = 0,
    ) -> int:
        """Log a protective stop change.

        Trail monotonicity is enforced by the strategy, not the journal.
        The journal records what happened for audit.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO stop_changes
               (timestamp_ist, session_date, symbol, side, previous_stop,
                new_stop, reason, r_multiple, best_favorable_price,
                modification_number)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._now_ist(),
                self._today_ist(),
                symbol,
                side,
                previous_stop,
                new_stop,
                reason,
                r_multiple,
                best_favorable_price,
                modification_number,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_warning(
        self,
        severity: str,
        code: str,
        message: str,
        data: Optional[dict[str, Any]] = None,
    ) -> int:
        """Log a warning or error event.

        Args:
            severity: INFO, WARNING, CRITICAL, EMERGENCY
            code: Machine-readable code (e.g., STALE_FEED, MARGIN_ERROR)
            message: Human-readable description.
            data: Additional context data.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO warnings
               (timestamp_ist, session_date, severity, code, message, data)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                self._now_ist(),
                self._today_ist(),
                severity,
                code,
                message,
                json.dumps(data) if data else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_daily_summary(
        self,
        mode: str,
        symbols_watched: list[str],
        trades_taken: int = 0,
        trades_won: int = 0,
        trades_lost: int = 0,
        gross_pnl: float = 0.0,
        total_costs: float = 0.0,
        net_pnl: float = 0.0,
        max_drawdown: float = 0.0,
        max_favorable_excursion: float = 0.0,
        max_adverse_excursion: float = 0.0,
        risk_events: Optional[list[str]] = None,
        halts: int = 0,
        data: Optional[dict[str, Any]] = None,
    ) -> int:
        """Log end-of-day summary. One row per session date."""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT OR REPLACE INTO daily_summaries
               (session_date, mode, symbols_watched, trades_taken, trades_won,
                trades_lost, gross_pnl, total_costs, net_pnl, max_drawdown,
                max_favorable_excursion, max_adverse_excursion, risk_events,
                halts, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._today_ist(),
                mode,
                json.dumps(symbols_watched),
                trades_taken,
                trades_won,
                trades_lost,
                gross_pnl,
                total_costs,
                net_pnl,
                max_drawdown,
                max_favorable_excursion,
                max_adverse_excursion,
                json.dumps(risk_events) if risk_events else None,
                halts,
                json.dumps(data) if data else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # ----- Query helpers -----

    def get_events(
        self,
        session_date: Optional[str] = None,
        event_type: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query events by date, type, and/or symbol."""
        conn = self._get_conn()
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if session_date:
            query += " AND session_date = ?"
            params.append(session_date)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_warnings(
        self,
        session_date: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query warning and critical error events by date and/or severity."""
        conn = self._get_conn()
        query = "SELECT * FROM warnings WHERE 1=1"
        params: list[Any] = []
        if session_date:
            query += " AND session_date = ?"
            params.append(session_date)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_orders(
        self,
        session_date: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query orders by date and/or symbol."""
        conn = self._get_conn()
        query = "SELECT * FROM orders WHERE 1=1"
        params: list[Any] = []
        if session_date:
            query += " AND session_date = ?"
            params.append(session_date)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_daily_summary(self, session_date: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Get daily summary for a date (default: today)."""
        conn = self._get_conn()
        date_str = session_date or self._today_ist()
        row = conn.execute(
            "SELECT * FROM daily_summaries WHERE session_date = ?",
            (date_str,),
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
