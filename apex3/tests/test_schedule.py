"""Tests for IST Session Scheduler and Holiday Logic."""
from datetime import date, datetime
from zoneinfo import ZoneInfo
import pytest

from apex3.config import ScheduleConfig
from apex3.strategy.schedule import SessionScheduler, SessionState

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def scheduler() -> SessionScheduler:
    config = ScheduleConfig(
        market_open="09:15",
        opening_range_end="09:30",
        entry_start="09:30",
        entry_cutoff="14:15",
        flatten_start="14:55",
        flat_target="14:58",
        market_close="15:30",
        hard_deadline="15:00",
        holidays=["2026-01-26", "2026-10-02"],
    )
    return SessionScheduler(config)


def test_opening_range_window(scheduler):
    """09:15 to 09:29:59 IST must be OPENING_RANGE, can_enter=False."""
    t = datetime(2026, 10, 21, 9, 20, tzinfo=IST)  # Wednesday
    assert scheduler.get_session_state(t) == SessionState.OPENING_RANGE
    assert not scheduler.can_enter(t)
    assert scheduler.is_opening_range_period(t)


def test_entry_window_active(scheduler):
    """09:30 to 14:14:59 IST must be ENTRY_WINDOW, can_enter=True."""
    t = datetime(2026, 10, 21, 10, 15, tzinfo=IST)
    assert scheduler.get_session_state(t) == SessionState.ENTRY_WINDOW
    assert scheduler.can_enter(t)


def test_entry_cutoff_passed(scheduler):
    """14:15 to 14:54:59 IST must be MANAGE_ONLY, can_enter=False."""
    t = datetime(2026, 10, 21, 14, 20, tzinfo=IST)
    assert scheduler.get_session_state(t) == SessionState.MANAGE_ONLY
    assert not scheduler.can_enter(t)
    assert not scheduler.should_flatten(t)


def test_flatten_window(scheduler):
    """14:55 to 15:00 IST must trigger should_flatten=True."""
    t = datetime(2026, 10, 21, 14, 56, tzinfo=IST)
    assert scheduler.get_session_state(t) == SessionState.FLATTEN
    assert scheduler.should_flatten(t)
    assert not scheduler.can_enter(t)


def test_flat_target_and_hard_deadline(scheduler):
    """Verify target flat by 14:58 and hard deadline at 15:00."""
    t_1458 = datetime(2026, 10, 21, 14, 58, 30, tzinfo=IST)
    assert scheduler.is_past_flat_target(t_1458)
    assert not scheduler.is_past_hard_deadline(t_1458)

    t_1501 = datetime(2026, 10, 21, 15, 1, 0, tzinfo=IST)
    assert scheduler.is_past_hard_deadline(t_1501)


def test_holiday_and_weekend(scheduler):
    """Holidays and weekends must return appropriate non-trading states."""
    # Weekend (Saturday)
    t_weekend = datetime(2026, 10, 24, 11, 0, tzinfo=IST)
    assert scheduler.get_session_state(t_weekend) == SessionState.WEEKEND
    assert not scheduler.is_trading_day(t_weekend.date())

    # Holiday
    t_holiday = datetime(2026, 1, 26, 11, 0, tzinfo=IST)
    assert scheduler.get_session_state(t_holiday) == SessionState.HOLIDAY
    assert not scheduler.is_trading_day(t_holiday.date())
