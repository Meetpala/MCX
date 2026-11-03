"""Telegram and Email Alert Notifier for APEX-3 Production Events.

Sends real-time operator alerts on critical execution events:
1. Order Filled (Entry / Target Exit)
2. Order Rejected
3. Stop-Loss Hit
4. Daily Loss Limit Hit (-Rs. 240.00 hard stop)
5. Broker Connection / Session Lost
6. Unexpected Restart / Crash Recovery

Configuration is read strictly from environment variables (.env):
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
- ALERT_SMTP_HOST
- ALERT_SMTP_PORT (default: 587)
- ALERT_SMTP_USER
- ALERT_SMTP_PASSWORD
- ALERT_EMAIL_TO
"""
from __future__ import annotations

import logging
import os
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage
from typing import Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger("apex3.alerts")
IST = ZoneInfo("Asia/Kolkata")


class AlertNotifier:
    """Dispatches critical trading alerts via Telegram and/or SMTP Email."""

    def __init__(
        self,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        email_to: Optional[str] = None,
        timeout_sec: float = 4.0,
    ) -> None:
        self.telegram_bot_token = (
            telegram_bot_token if telegram_bot_token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        )
        self.telegram_chat_id = (
            telegram_chat_id if telegram_chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        )
        self.smtp_host = smtp_host if smtp_host is not None else os.environ.get("ALERT_SMTP_HOST", "").strip()
        self.smtp_port = int(smtp_port if smtp_port is not None else (os.environ.get("ALERT_SMTP_PORT", "587") or 587))
        self.smtp_user = smtp_user if smtp_user is not None else os.environ.get("ALERT_SMTP_USER", "").strip()
        self.smtp_password = (
            smtp_password if smtp_password is not None else os.environ.get("ALERT_SMTP_PASSWORD", "").strip()
        )
        self.email_to = email_to if email_to is not None else os.environ.get("ALERT_EMAIL_TO", "").strip()
        self.timeout_sec = timeout_sec

    @property
    def is_telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def is_email_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_password and self.email_to)

    def send_alert(self, event_type: str, message: str, async_send: bool = False) -> bool:
        """Send an alert across all configured channels (Telegram and/or Email)."""
        ts_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        formatted = f"[APEX-3 {event_type}] ({ts_str})\n{message}"

        if async_send:
            threading.Thread(
                target=self._dispatch_sync,
                args=(event_type, formatted),
                daemon=True,
            ).start()
            return True

        return self._dispatch_sync(event_type, formatted)

    def _dispatch_sync(self, event_type: str, formatted: str) -> bool:
        sent_any = False
        if self.is_telegram_configured:
            if self._send_telegram(formatted):
                sent_any = True
        if self.is_email_configured:
            if self._send_email(subject=f"[APEX-3 ALERT] {event_type}", body=formatted):
                sent_any = True
        if not self.is_telegram_configured and not self.is_email_configured:
            logger.info("Alert (%s): %s", event_type, formatted.replace("\n", " | "))
        return sent_any

    def _send_telegram(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        try:
            resp = requests.post(
                url,
                json={"chat_id": self.telegram_chat_id, "text": text},
                timeout=self.timeout_sec,
            )
            if resp.status_code == 200:
                return True
            logger.warning("Telegram alert returned HTTP %d", resp.status_code)
        except Exception as e:
            logger.warning("Telegram alert delivery failed: %s", e)
        return False

    def _send_email(self, subject: str, body: str) -> bool:
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = self.email_to
            msg.set_content(body)
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout_sec) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            return True
        except Exception as e:
            logger.warning("Email alert delivery failed: %s", e)
            return False

    # ── High-Level Event Helpers ──────────────────────────────────────────

    def notify_order_filled(
        self, symbol: str, side: str, quantity: int, price: float, order_id: str, async_send: bool = False
    ) -> bool:
        return self.send_alert(
            "ORDER_FILLED",
            f"Symbol: {symbol} | Side: {side} | Qty: {quantity} @ Rs.{price:,.2f} | OrderID: {order_id}",
            async_send=async_send,
        )

    def notify_order_rejected(
        self, symbol: str, side: str, order_id: str, reason: str, async_send: bool = False
    ) -> bool:
        return self.send_alert(
            "ORDER_REJECTED",
            f"Symbol: {symbol} | Side: {side} | OrderID: {order_id} | Reason: {reason}",
            async_send=async_send,
        )

    def notify_sl_hit(
        self, symbol: str, quantity: int, fill_price: float, net_pnl: float, async_send: bool = False
    ) -> bool:
        return self.send_alert(
            "SL_HIT",
            f"Symbol: {symbol} | Qty: {quantity} | Stop Fill: Rs.{fill_price:,.2f} | Net PnL: Rs.{net_pnl:+,.2f}",
            async_send=async_send,
        )

    def notify_daily_loss_limit(
        self, realized_pnl: float, limit_rupees: float, async_send: bool = False
    ) -> bool:
        return self.send_alert(
            "DAILY_LOSS_LIMIT_HIT",
            f"EMERGENCY: Session PnL (Rs.{realized_pnl:+,.2f}) breached daily limit (Rs.{limit_rupees:,.2f}). Flattening & halting.",
            async_send=async_send,
        )

    def notify_connection_lost(
        self, broker_name: str, details: str, async_send: bool = False
    ) -> bool:
        return self.send_alert(
            "CONNECTION_LOST",
            f"Broker: {broker_name} | Details: {details}",
            async_send=async_send,
        )

    def notify_unexpected_restart(
        self, mode: str, recovered_positions: list[str], async_send: bool = False
    ) -> bool:
        pos_str = ", ".join(recovered_positions) if recovered_positions else "None (Flat)"
        return self.send_alert(
            "ENGINE_RESTART_RECONCILED",
            f"Mode: {mode.upper()} | Recovered Open Positions: {pos_str}",
            async_send=async_send,
        )

