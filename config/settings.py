"""
config/settings.py
------------------
System configuration, seasonal US Daylight Saving Time (DST) detector,
MCX trading session schedule, and Precious Metals contract specifications.
"""

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional
import zoneinfo
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# IST Timezone
IST = zoneinfo.ZoneInfo("Asia/Kolkata")
UTC = zoneinfo.ZoneInfo("UTC")

# Zerodha Kite Connect Credentials
KITE_API_KEY: str = os.getenv("KITE_API_KEY", "")
KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

# Risk Management Parameters
CAPITAL_RISK_PER_TRADE: float = float(os.getenv("CAPITAL_RISK_PER_TRADE", "0.01"))  # 1.0% equity
MAX_MARGIN_UTILIZATION: float = float(os.getenv("MAX_MARGIN_UTILIZATION", "0.65"))  # 65% usable margin
DAILY_LOSS_LIMIT_PCT: float = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.02"))      # 2.0% daily circuit breaker
MAX_CONSECUTIVE_STOPOUTS: int = int(os.getenv("MAX_CONSECUTIVE_STOPOUTS", "2"))

# Execution & Trading Controls
TRADING_MODE: str = os.getenv("TRADING_MODE", "PAPER").upper()  # 'PAPER' or 'LIVE'
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_COMMODITY: str = os.getenv("DEFAULT_COMMODITY", "GOLD").upper()
PREFERRED_TIER: str = os.getenv("PREFERRED_TIER", "MINI").upper()  # 'STANDARD', 'MINI', 'MICRO'


class CommodityFamily(str, Enum):
    GOLD = "GOLD"
    SILVER = "SILVER"


class ContractTier(str, Enum):
    STANDARD = "STANDARD"
    MINI = "MINI"
    MICRO = "MICRO"


class SessionWindow(str, Enum):
    PRE_MARKET = "PRE_MARKET"
    OPENING_STABILIZATION = "OPENING_STABILIZATION"   # 09:00 - 09:15
    WINDOW_1_ASIA_INDIA = "WINDOW_1_ASIA_INDIA"       # 09:15 - 11:30
    MIDDAY_CHOP_GUARD = "MIDDAY_CHOP_GUARD"           # 11:30 - 13:30
    WINDOW_2_LONDON_OPEN = "WINDOW_2_LONDON_OPEN"     # 13:30 - 16:30
    PRE_NY_CONSOLIDATION = "PRE_NY_CONSOLIDATION"     # 16:30 - 17:45 / 18:45
    WINDOW_3_COMEX_PRIME = "WINDOW_3_COMEX_PRIME"     # 17:45/18:45 - 21:45/22:15
    NO_NEW_ENTRY_CUTOFF = "NO_NEW_ENTRY_CUTOFF"       # T_close - 75m to T_close - 20m
    HARD_AUTO_SQUAREOFF = "HARD_AUTO_SQUAREOFF"       # T_close - 20m to T_close
    MARKET_CLOSED = "MARKET_CLOSED"


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    family: CommodityFamily
    tier: ContractTier
    lot_size: int          # Quantity per contract lot
    multiplier: float      # Point multiplier to rupees (Delta Rs 1 P&L)
    tick_size: float       # Tick size in rupees
    approx_nrml_margin: float # Typical margin %
    min_capital: float     # Recommended capital
    exchange: str = "MCX"


# Instrument Matrix according to specification Section 2
MCX_INSTRUMENTS: Dict[str, InstrumentSpec] = {
    # Gold Hierarchy
    "GOLDTEN": InstrumentSpec(
        symbol="GOLDTEN",
        family=CommodityFamily.GOLD,
        tier=ContractTier.MICRO,
        lot_size=1,        # 10 grams traded as 1 unit
        multiplier=1.0,    # Delta Rs 1 = Rs 1
        tick_size=1.0,
        approx_nrml_margin=0.0925,
        min_capital=35000.0,
    ),
    "GOLDM": InstrumentSpec(
        symbol="GOLDM",
        family=CommodityFamily.GOLD,
        tier=ContractTier.MINI,
        lot_size=1,        # 100 grams traded as 1 unit (quoted per 10g => 10x)
        multiplier=10.0,   # Delta Rs 1 = Rs 10
        tick_size=1.0,
        approx_nrml_margin=0.0925,
        min_capital=250000.0,
    ),
    "GOLD": InstrumentSpec(
        symbol="GOLD",
        family=CommodityFamily.GOLD,
        tier=ContractTier.STANDARD,
        lot_size=1,        # 1 kg traded as 1 unit (quoted per 10g => 100x)
        multiplier=100.0,  # Delta Rs 1 = Rs 100
        tick_size=1.0,
        approx_nrml_margin=0.0925,
        min_capital=2500000.0,
    ),
    # Silver Hierarchy
    "SILVERMIC": InstrumentSpec(
        symbol="SILVERMIC",
        family=CommodityFamily.SILVER,
        tier=ContractTier.MICRO,
        lot_size=1,        # 1 kg traded as 1 unit (quoted per 1 kg => 1x)
        multiplier=1.0,    # Delta Rs 1 = Rs 1
        tick_size=1.0,
        approx_nrml_margin=0.1552,
        min_capital=50000.0,
    ),
    "SILVERM": InstrumentSpec(
        symbol="SILVERM",
        family=CommodityFamily.SILVER,
        tier=ContractTier.MINI,
        lot_size=1,        # 5 kg traded as 1 unit (quoted per 1 kg => 5x)
        multiplier=5.0,    # Delta Rs 1 = Rs 5
        tick_size=1.0,
        approx_nrml_margin=0.1552,
        min_capital=300000.0,
    ),
    "SILVER": InstrumentSpec(
        symbol="SILVER",
        family=CommodityFamily.SILVER,
        tier=ContractTier.STANDARD,
        lot_size=1,        # 30 kg traded as 1 unit (quoted per 1 kg => 30x)
        multiplier=30.0,   # Delta Rs 1 = Rs 30
        tick_size=1.0,
        approx_nrml_margin=0.1583,
        min_capital=2000000.0,
    ),
}

# Tier downgrade order when sizing falls below 1 lot
TIER_DOWNGRADE_ORDER: Dict[CommodityFamily, List[str]] = {
    CommodityFamily.GOLD: ["GOLD", "GOLDM", "GOLDTEN"],
    CommodityFamily.SILVER: ["SILVER", "SILVERM", "SILVERMIC"],
}


def is_us_dst(target_dt: Optional[datetime | date] = None) -> bool:
    """
    US Daylight Saving Time Schedule Detector.
    US DST starts at 02:00 local on the 2nd Sunday in March,
    and ends at 02:00 local on the 1st Sunday in November.

    Summer Time (DST active): MCX closes at 23:30 IST.
    Winter Time (DST inactive): MCX closes at 23:55 IST.
    """
    if target_dt is None:
        target_date = datetime.now(IST).date()
    elif isinstance(target_dt, datetime):
        target_date = target_dt.date()
    else:
        target_date = target_dt

    year = target_date.year

    # 2nd Sunday in March
    # March 1 is day of week weekday() (0=Monday, 6=Sunday)
    first_of_march = date(year, 3, 1)
    # Days until first Sunday: (6 - weekday) % 7
    days_to_first_sun = (6 - first_of_march.weekday()) % 7
    first_sun_march = first_of_march + timedelta(days=days_to_first_sun)
    second_sun_march = first_sun_march + timedelta(days=7)

    # 1st Sunday in November
    first_of_nov = date(year, 11, 1)
    days_to_nov_sun = (6 - first_of_nov.weekday()) % 7
    first_sun_nov = first_of_nov + timedelta(days=days_to_nov_sun)

    # In the US, DST is active between 2nd Sunday in March and 1st Sunday in Nov
    return second_sun_march <= target_date < first_sun_nov


@dataclass(frozen=True)
class MCXSchedule:
    target_date: date
    is_dst: bool
    market_open: time          # 09:00 IST
    market_close: time         # 23:30 (Summer) or 23:55 (Winter)
    entry_cutoff_time: time    # T_close - 75m: 22:15 or 22:40
    auto_squareoff_time: time  # T_close - 20m: 23:10 or 23:35
    zero_carry_audit_time: time# T_close - 18m: 23:12 or 23:37
    pre_ny_start: time         # 16:30
    comex_prime_start: time    # 17:45 (Summer) or 18:45 (Winter)
    comex_prime_end: time      # 21:45 (Summer) or 22:15 (Winter)


def get_mcx_schedule(target: Optional[datetime | date] = None) -> MCXSchedule:
    """Computes the full MCX schedule for a given date, factoring in US DST."""
    if target is None:
        target_d = datetime.now(IST).date()
    elif isinstance(target, datetime):
        target_d = target.date()
    else:
        target_d = target

    dst = is_us_dst(target_d)

    if dst:
        # US Summer Time: Close at 23:30 IST
        return MCXSchedule(
            target_date=target_d,
            is_dst=True,
            market_open=time(9, 0),
            market_close=time(23, 30),
            entry_cutoff_time=time(22, 15),     # 23:30 - 75 min
            auto_squareoff_time=time(23, 10),   # 23:30 - 20 min
            zero_carry_audit_time=time(23, 12), # 23:30 - 18 min
            pre_ny_start=time(16, 30),
            comex_prime_start=time(17, 45),
            comex_prime_end=time(21, 45),
        )
    else:
        # US Winter Time: Close at 23:55 IST
        return MCXSchedule(
            target_date=target_d,
            is_dst=False,
            market_open=time(9, 0),
            market_close=time(23, 55),
            entry_cutoff_time=time(22, 40),     # 23:55 - 75 min
            auto_squareoff_time=time(23, 35),   # 23:55 - 20 min
            zero_carry_audit_time=time(23, 37), # 23:55 - 18 min
            pre_ny_start=time(16, 30),
            comex_prime_start=time(18, 45),
            comex_prime_end=time(22, 15),
        )


def get_current_session_window(now_dt: Optional[datetime] = None) -> SessionWindow:
    """Identifies the active intraday operational window."""
    if now_dt is None:
        now_dt = datetime.now(IST)
    elif now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=IST)

    schedule = get_mcx_schedule(now_dt.date())
    curr_t = now_dt.time()

    if curr_t < schedule.market_open or curr_t >= schedule.market_close:
        return SessionWindow.MARKET_CLOSED
    if curr_t < time(9, 15):
        return SessionWindow.OPENING_STABILIZATION
    if curr_t < time(11, 30):
        return SessionWindow.WINDOW_1_ASIA_INDIA
    if curr_t < time(13, 30):
        return SessionWindow.MIDDAY_CHOP_GUARD
    if curr_t < time(16, 30):
        return SessionWindow.WINDOW_2_LONDON_OPEN
    if curr_t < schedule.comex_prime_start:
        return SessionWindow.PRE_NY_CONSOLIDATION
    if curr_t < schedule.entry_cutoff_time:
        return SessionWindow.WINDOW_3_COMEX_PRIME
    if curr_t < schedule.auto_squareoff_time:
        return SessionWindow.NO_NEW_ENTRY_CUTOFF
    if curr_t < schedule.market_close:
        return SessionWindow.HARD_AUTO_SQUAREOFF

    return SessionWindow.MARKET_CLOSED


def is_entry_allowed(now_dt: Optional[datetime] = None) -> bool:
    """
    Checks if new trade entries are allowed based on time cutoffs.
    Strictly forbids entries when time >= T_close - 75m or during non-entry windows.
    """
    if now_dt is None:
        now_dt = datetime.now(IST)
    elif now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=IST)

    window = get_current_session_window(now_dt)
    return window in (
        SessionWindow.WINDOW_1_ASIA_INDIA,
        SessionWindow.MIDDAY_CHOP_GUARD,   # Permitted only under high volume/ADX conditional rule
        SessionWindow.WINDOW_2_LONDON_OPEN,
        SessionWindow.WINDOW_3_COMEX_PRIME,
    )


def should_rollover(current_date: date, expiry_date: date, rollover_days: int = 7) -> bool:
    """
    Checks if contract should be rolled over to next month's contract.
    Mandatory rollover occurs 7 calendar days before expiry to avoid
    Zerodha's 5-day pre-expiry physical delivery compulsory square-off.
    """
    days_to_expiry = (expiry_date - current_date).days
    return days_to_expiry <= rollover_days


def select_active_contract(
    instruments: List[Dict],
    symbol_name: str,
    as_of_date: Optional[date] = None,
    rollover_days: int = 7,
) -> Optional[Dict]:
    """
    Selects the active non-tender contract for symbol_name (e.g. 'GOLDM', 'SILVERMIC').
    Filters instruments where days_to_expiry > 7, sorted by expiry ascending.
    """
    if as_of_date is None:
        as_of_date = datetime.now(IST).date()

    matching = []
    for inst in instruments:
        # Check symbol and exchange
        if inst.get("segment") == "MCX" or inst.get("exchange") == "MCX":
            name = inst.get("name") or inst.get("tradingsymbol", "")
            if name.startswith(symbol_name):
                expiry = inst.get("expiry")
                if isinstance(expiry, str):
                    expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
                elif isinstance(expiry, datetime):
                    expiry = expiry.date()

                if expiry and not should_rollover(as_of_date, expiry, rollover_days):
                    matching.append((expiry, inst))

    if not matching:
        return None

    # Sort by expiry ascending (nearest eligible contract)
    matching.sort(key=lambda x: x[0])
    return matching[0][1]

