"""APEX-3 IST Session Scheduler.

Manages trading session windows, exchange holidays, and time-based
state transitions for the intraday strategy.

All times are in Asia/Kolkata (IST). Uses exchange session calendar,
not the computer's local wall clock.

Session states:
- PRE_OPEN: Before 09:15 — system startup, data validation
- OPENING_RANGE: 09:15–09:30 — observe, build opening range, no entries
- ENTRY_WINDOW: 09:30–14:15 — entries allowed when all gates pass
- MANAGE_ONLY: 14:15–14:55 — no new entries, manage existing positions
- FLATTEN: 14:55–15:00 — exit all strategy MIS positions
- POST_CLOSE: After 15:00 — reconcile and report
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from enum import Enum, auto
from typing import Optional
from zoneinfo import ZoneInfo

from apex3.config import ScheduleConfig

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class SessionState(Enum):
    """Current state of the trading session."""
    PRE_OPEN = auto()
    OPENING_RANGE = auto()
    ENTRY_WINDOW = auto()
    MANAGE_ONLY = auto()
    FLATTEN = auto()
    POST_CLOSE = auto()
    HOLIDAY = auto()
    WEEKEND = auto()
    SPECIAL_SESSION = auto()


class SessionScheduler:
    """Determines the current session state based on IST time.

    This scheduler uses the exchange session calendar and configured
    holidays. It does NOT rely on the computer's timezone — all
    comparisons use explicit IST conversion.
    """

    def __init__(self, config: ScheduleConfig):
        """Initialize with schedule configuration.

        Args:
            config: Schedule configuration from config.yaml.
        """
        self._config = config

        # Parse configured times
        self._market_open = config.get_time("market_open")
        self._or_end = config.get_time("opening_range_end")
        self._entry_start = config.get_time("entry_start")
        self._entry_cutoff = config.get_time("entry_cutoff")
        self._flatten_start = config.get_time("flatten_start")
        self._flat_target = config.get_time("flat_target")
        self._hard_deadline = config.get_time("hard_deadline")
        self._market_close = config.get_time("market_close")

    def now_ist(self) -> datetime:
        """Get the current time in IST.

        Returns:
            Current datetime with Asia/Kolkata timezone.
        """
        return datetime.now(IST)

    def today_ist(self) -> date:
        """Get today's date in IST."""
        return self.now_ist().date()

    def validate_calendar_freshness(self, target_year: Optional[int] = None) -> tuple[bool, str]:
        """Verify that the configured exchange holiday calendar includes holidays for the operating year.

        Prevents running against a live exchange when the static holiday calendar is out of date.
        """
        yr = target_year or self.today_ist().year
        year_str = str(yr)
        has_year = any(str(h).startswith(year_str) for h in getattr(self._config, "holidays", []))
        if not has_year:
            msg = f"Stale holiday calendar: No exchange holidays configured for year {yr}. Annual calendar update required."
            logger.critical(msg)
            return False, msg
        return True, f"Calendar validated for year {yr}"

    def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        """Check if a date is a valid trading day.

        A trading day is a weekday that is not in the holiday list.
        NOTE: This uses a static holiday list from config. The user
        must update holidays annually from the NSE calendar.

        Args:
            check_date: Date to check. Defaults to today IST.

        Returns:
            True if the date is a trading day.
        """
        d = check_date or self.today_ist()
        # Weekday check: Monday=0 through Friday=4
        if d.weekday() >= 5:
            return False
        # Holiday check
        return not self._config.is_holiday(d)

    def get_session_state(self, at_time: Optional[datetime] = None) -> SessionState:
        """Determine the current session state.

        Args:
            at_time: Time to evaluate. Defaults to now IST.

        Returns:
            Current SessionState.
        """
        now = at_time or self.now_ist()
        now_date = now.date()
        now_time = now.timetz().replace(tzinfo=None)

        # Special session (e.g. Muhurat Trading)
        is_special, _ = self._config.is_special_session(now_date)
        if is_special:
            return SessionState.SPECIAL_SESSION

        # Weekend
        if now_date.weekday() >= 5:
            return SessionState.WEEKEND

        # Holiday
        if self._config.is_holiday(now_date):
            return SessionState.HOLIDAY

        # Pre-open
        if now_time < self._market_open:
            return SessionState.PRE_OPEN

        # Opening range: 09:15 to entry_start (09:30 / 09:35)
        if self._market_open <= now_time < max(self._or_end, self._entry_start):
            return SessionState.OPENING_RANGE

        # Entry window: 09:30 to 14:15
        if self._entry_start <= now_time < self._entry_cutoff:
            return SessionState.ENTRY_WINDOW

        # Manage only: 14:15 to 14:55
        if self._entry_cutoff <= now_time < self._flatten_start:
            return SessionState.MANAGE_ONLY

        # Flatten: 14:55 to market close
        if self._flatten_start <= now_time < self._market_close:
            return SessionState.FLATTEN

        # Post close
        return SessionState.POST_CLOSE

    def can_enter(self, at_time: Optional[datetime] = None) -> bool:
        """Check if new entries are permitted at this time.

        Entries are only allowed during ENTRY_WINDOW state.

        Args:
            at_time: Time to check. Defaults to now IST.
        """
        return self.get_session_state(at_time) == SessionState.ENTRY_WINDOW

    def should_flatten(self, at_time: Optional[datetime] = None) -> bool:
        """Check if the system should begin flattening positions.

        Args:
            at_time: Time to check. Defaults to now IST.
        """
        return self.get_session_state(at_time) == SessionState.FLATTEN

    def is_past_flat_target(self, at_time: Optional[datetime] = None) -> bool:
        """Check if we're past the target time to be flat.

        This is 14:58 IST by default. If not flat by this time,
        the system should escalate alerts.
        """
        now = at_time or self.now_ist()
        now_time = now.timetz().replace(tzinfo=None)
        return now_time >= self._flat_target

    def is_past_hard_deadline(self, at_time: Optional[datetime] = None) -> bool:
        """Check if we're past the hard deadline (15:00 IST).

        Any remaining strategy position at this point is a critical
        failure requiring operator intervention.

        WARNING: Never claim the broker will guarantee an exit.
        The strategy's own earlier exit is the control.
        """
        now = at_time or self.now_ist()
        now_time = now.timetz().replace(tzinfo=None)
        return now_time >= self._hard_deadline

    def is_opening_range_period(self, at_time: Optional[datetime] = None) -> bool:
        """Check if we're in the opening range build period (09:15–09:30)."""
        return self.get_session_state(at_time) == SessionState.OPENING_RANGE

    def seconds_until_flatten(self, at_time: Optional[datetime] = None) -> float:
        """Seconds remaining until flatten time.

        Returns negative if already past flatten time.
        """
        now = at_time or self.now_ist()
        today = now.date()
        flatten_dt = datetime.combine(today, self._flatten_start, tzinfo=IST)
        return (flatten_dt - now).total_seconds()

    def seconds_until_cutoff(self, at_time: Optional[datetime] = None) -> float:
        """Seconds remaining until entry cutoff.

        Returns negative if already past cutoff.
        """
        now = at_time or self.now_ist()
        today = now.date()
        cutoff_dt = datetime.combine(today, self._entry_cutoff, tzinfo=IST)
        return (cutoff_dt - now).total_seconds()

    def get_next_trading_day(self, after_date: Optional[date] = None) -> date:
        """Find the next valid trading day after a given date.

        Args:
            after_date: Start searching after this date. Defaults to today.

        Returns:
            Next trading day date.
        """
        d = after_date or self.today_ist()
        for _ in range(10):  # Safety limit to prevent infinite loop
            d += timedelta(days=1)
            if self.is_trading_day(d):
                return d
        # Fallback — should never happen with reasonable holiday lists
        return d

    def get_state_display(self, at_time: Optional[datetime] = None) -> dict[str, str]:
        """Get a human-readable display of the current session state.

        Returns:
            Dict with keys: state, description, entries_allowed, time_ist
        """
        state = self.get_session_state(at_time)
        now = at_time or self.now_ist()

        descriptions = {
            SessionState.PRE_OPEN: "Pre-open — validating data and account",
            SessionState.OPENING_RANGE: "Building opening range (09:15–09:30) — no entries",
            SessionState.ENTRY_WINDOW: "Entry window open — evaluating signals",
            SessionState.MANAGE_ONLY: "Entry cutoff passed — managing existing positions only",
            SessionState.FLATTEN: "FLATTEN — exiting all strategy MIS positions",
            SessionState.POST_CLOSE: "Post-close — reconciling and reporting",
            SessionState.HOLIDAY: "Exchange holiday — no trading",
            SessionState.WEEKEND: "Weekend — no trading",
            SessionState.SPECIAL_SESSION: "Special session — intraday MIS entries disabled",
        }

        return {
            "state": state.name,
            "description": descriptions[state],
            "entries_allowed": str(state == SessionState.ENTRY_WINDOW),
            "time_ist": now.strftime("%H:%M:%S %Z"),
            "date": now.date().isoformat(),
        }
