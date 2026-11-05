"""APEX-3 Market Regime Filter.

Evaluates broad-market regime using NIFTY 50 OHLC trend data (EMA-20 slope, ADX, VWAP alignment).
Supports both BenchmarkSnapshot (high-speed radar scanning) and Candle series.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo

from apex3.config import RegimeConfig
from apex3.indicators.adx import compute_adx
from apex3.indicators.ema import compute_ema, ema_slope
from apex3.models import Candle, GateResult, GateStatus, Quote, Side

IST = ZoneInfo("Asia/Kolkata")


def is_us_dst(dt: Union[date, datetime]) -> bool:
    """Return True if the given date falls within US Daylight Saving Time (EDT).

    US DST begins on the 2nd Sunday of March and ends on the 1st Sunday of November.
    """
    d = dt.date() if isinstance(dt, datetime) else dt
    year = d.year

    # 2nd Sunday of March
    first_mar = date(year, 3, 1)
    first_sun_mar = 1 + (6 - first_mar.weekday()) % 7
    second_sun_mar = first_sun_mar + 7
    dst_start = date(year, 3, second_sun_mar)

    # 1st Sunday of November
    first_nov = date(year, 11, 1)
    first_sun_nov = 1 + (6 - first_nov.weekday()) % 7
    dst_end = date(year, 11, first_sun_nov)

    return dst_start <= d < dst_end


def get_active_mcx_session_windows(dt: Union[date, datetime]) -> dict[str, Any]:
    """Calculate active MCX session timing and cutoff windows for a given date.

    - US Daylight Saving Time (Summer / EDT):
      Market Closes: 23:30 IST
      Pre-Close Entry Cutoff (T_close - 75m): 22:15 IST
      Mandatory Intraday Flatten (T_close - 20m): 23:10 IST
      COMEX Opening Range: 17:00 - 17:30 IST
    - US Standard Time (Winter / EST):
      Market Closes: 23:55 IST
      Pre-Close Entry Cutoff (T_close - 75m): 22:40 IST
      Mandatory Intraday Flatten (T_close - 20m): 23:35 IST
      COMEX Opening Range: 18:00 - 18:30 IST
    - Common:
      Morning Session Open: 09:00 IST
      Morning Opening Range: 09:00 - 09:15 IST
    """
    is_dst = is_us_dst(dt)
    if is_dst:
        close_t = time(23, 30)
        cutoff_t = time(22, 15)
        flatten_t = time(23, 10)
        comex_start = time(17, 0)
        comex_end = time(17, 30)
    else:
        close_t = time(23, 55)
        cutoff_t = time(22, 40)
        flatten_t = time(23, 35)
        comex_start = time(18, 0)
        comex_end = time(18, 30)

    return {
        "is_us_dst": is_dst,
        "market_open": time(9, 0),
        "market_close": close_t,
        "entry_cutoff": cutoff_t,
        "intraday_flatten": flatten_t,
        "or_morning_start": time(9, 0),
        "or_morning_end": time(9, 15),
        "or_comex_start": comex_start,
        "or_comex_end": comex_end,
    }


def calculate_session_anchored_vwap(
    candles: list[Candle],
    anchor_time: Optional[time] = None,
) -> tuple[float, float]:
    """Compute Session-Anchored VWAP and standard deviation (sigma_vwap).

    If anchor_time is None, defaults to 09:00 IST.
    Returns (vwap, sigma_vwap).
    """
    if not candles:
        return 0.0, 0.0

    target_d = candles[-1].timestamp.date() if hasattr(candles[-1].timestamp, "date") else None
    anc_t = anchor_time or time(9, 0)

    filtered = []
    for c in candles:
        c_ts = c.timestamp if c.timestamp.tzinfo else c.timestamp.replace(tzinfo=IST)
        c_ist = c_ts.astimezone(IST)
        if target_d is not None and c_ist.date() != target_d:
            continue
        if c_ist.time() >= anc_t:
            filtered.append(c)

    if not filtered:
        filtered = candles

    total_vol = sum(c.volume for c in filtered)
    if total_vol > 0:
        typical_prices = [(c.high + c.low + c.close) / 3.0 for c in filtered]
        vwap = sum(tp * c.volume for tp, c in zip(typical_prices, filtered)) / total_vol
        variance = sum(c.volume * ((tp - vwap) ** 2) for tp, c in zip(typical_prices, filtered)) / total_vol
        sigma_vwap = math.sqrt(max(0.0, variance))
        return vwap, sigma_vwap
    else:
        typical_prices = [(c.high + c.low + c.close) / 3.0 for c in filtered]
        vwap = sum(typical_prices) / len(typical_prices)
        variance = sum((tp - vwap) ** 2 for tp in typical_prices) / len(typical_prices)
        sigma_vwap = math.sqrt(max(0.0, variance))
        return vwap, sigma_vwap


def detect_smt_divergence(
    lead_candles: list[Candle],
    sibling_candles: list[Candle],
    lookback: int = 6,
) -> tuple[Optional[str], str]:
    """Detect Smart Money Technique (SMT) divergence between Gold and Silver.

    - Bullish SMT Divergence ("BULLISH_SMT"):
      One precious metal makes a lower low while the sibling holds a higher low.
    - Bearish SMT Divergence ("BEARISH_SMT"):
      One precious metal makes a higher high while the sibling holds a lower high.
    """
    comp_lead = [c for c in lead_candles if getattr(c, "is_complete", True)]
    comp_sib = [c for c in sibling_candles if getattr(c, "is_complete", True)]

    if len(comp_lead) < lookback + 1 or len(comp_sib) < lookback + 1:
        return None, "Insufficient completed candles to evaluate SMT divergence"

    curr_l = comp_lead[-1]
    curr_s = comp_sib[-1]

    prior_lead = comp_lead[-lookback - 1 : -1]
    prior_sib = comp_sib[-lookback - 1 : -1]

    lead_min_l = min(c.low for c in prior_lead)
    lead_max_h = max(c.high for c in prior_lead)
    sib_min_l = min(c.low for c in prior_sib)
    sib_max_h = max(c.high for c in prior_sib)

    lead_ll = curr_l.low < lead_min_l
    sib_ll = curr_s.low < sib_min_l

    lead_hh = curr_l.high > lead_max_h
    sib_hh = curr_s.high > sib_max_h

    # Bullish SMT: One makes lower low, other holds higher low
    if lead_ll and not sib_ll:
        return (
            "BULLISH_SMT",
            f"Bullish SMT: Lead swept low ({curr_l.low:.2f} < {lead_min_l:.2f}) while sibling held ({curr_s.low:.2f} >= {sib_min_l:.2f})",
        )
    if sib_ll and not lead_ll:
        return (
            "BULLISH_SMT",
            f"Bullish SMT: Sibling swept low ({curr_s.low:.2f} < {sib_min_l:.2f}) while lead held ({curr_l.low:.2f} >= {lead_min_l:.2f})",
        )

    # Bearish SMT: One makes higher high, other holds lower high
    if lead_hh and not sib_hh:
        return (
            "BEARISH_SMT",
            f"Bearish SMT: Lead broke high ({curr_l.high:.2f} > {lead_max_h:.2f}) while sibling failed ({curr_s.high:.2f} <= {sib_max_h:.2f})",
        )
    if sib_hh and not lead_hh:
        return (
            "BEARISH_SMT",
            f"Bearish SMT: Sibling broke high ({curr_s.high:.2f} > {sib_max_h:.2f}) while lead held ({curr_l.high:.2f} <= {lead_max_h:.2f})",
        )

    return None, "No SMT divergence detected"


class MarketRegime(str, Enum):
    """Broad-market trend regime."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    WAITING = "WAITING"


Regime = MarketRegime


def calculate_choppiness_index(candles_15m: list[Candle], period: int = 14) -> float:
    """Computes Choppiness Index (CI_14) on 15m candles:
    CI = 100 * LOG10(Sum(TR_14) / (MaxHigh_14 - MinLow_14)) / LOG10(period)
    """
    if len(candles_15m) < period + 1:
        return 50.0

    window = candles_15m[-period:]
    true_ranges = []
    for i in range(len(window)):
        curr = window[i]
        idx_in_all = len(candles_15m) - period + i
        prev_c = candles_15m[idx_in_all - 1].close
        tr = max(curr.high - curr.low, abs(curr.high - prev_c), abs(curr.low - prev_c))
        true_ranges.append(tr)

    sum_tr = sum(true_ranges)
    max_h = max(c.high for c in window)
    min_l = min(c.low for c in window)
    high_low_span = max_h - min_l

    if high_low_span <= 1e-6 or sum_tr <= 1e-6:
        return 50.0

    ci = 100.0 * (math.log10(sum_tr / high_low_span) / math.log10(period))
    return min(max(ci, 0.0), 100.0)


compute_choppiness_index = calculate_choppiness_index


def evaluate_bullion_alignment(
    gold_candles: list[Candle],
    silver_candles: list[Candle],
    intended_side: Optional[Side] = None,
) -> tuple[bool, str]:
    """Verify that MCX Gold and Silver directionally agree on EMA_20 and Session VWAP."""
    if len(gold_candles) < 20 or len(silver_candles) < 20:
        return True, "Insufficient candles to verify bullion alignment; defaulting to pass"

    gold_closes = [c.close for c in gold_candles]
    silver_closes = [c.close for c in silver_candles]

    gold_ema20 = compute_ema(gold_closes, 20)[-1]
    silver_ema20 = compute_ema(silver_closes, 20)[-1]

    from apex3.indicators.vwap import compute_session_vwap

    today = gold_candles[-1].timestamp.date()
    gold_today = [c for c in gold_candles if c.timestamp.date() == today] or gold_candles[-20:]
    silver_today = [c for c in silver_candles if c.timestamp.date() == today] or silver_candles[-20:]

    gold_vol = sum(c.volume for c in gold_today)
    silver_vol = sum(c.volume for c in silver_today)

    gold_vwap = (
        compute_session_vwap(
            [c.high for c in gold_today],
            [c.low for c in gold_today],
            [c.close for c in gold_today],
            [c.volume for c in gold_today],
        )
        if gold_vol > 0
        else (sum(c.close for c in gold_today) / len(gold_today))
    )
    silver_vwap = (
        compute_session_vwap(
            [c.high for c in silver_today],
            [c.low for c in silver_today],
            [c.close for c in silver_today],
            [c.volume for c in silver_today],
        )
        if silver_vol > 0
        else (sum(c.close for c in silver_today) / len(silver_today))
    )

    gold_close = gold_closes[-1]
    silver_close = silver_closes[-1]

    gold_bull = (gold_close > gold_ema20) and (gold_close > gold_vwap)
    gold_bear = (gold_close < gold_ema20) and (gold_close < gold_vwap)

    silver_bull = (silver_close > silver_ema20) and (silver_close > silver_vwap)
    silver_bear = (silver_close < silver_ema20) and (silver_close < silver_vwap)

    if intended_side == Side.BUY:
        if not gold_bull:
            return False, f"Gold not bullish: Close={gold_close} (EMA20={gold_ema20:.1f}, VWAP={gold_vwap:.1f})"
        if not silver_bull:
            return False, f"Silver not bullish: Close={silver_close} (EMA20={silver_ema20:.1f}, VWAP={silver_vwap:.1f})"
        return True, "Bullion alignment confirmed (Gold & Silver Bullish)"

    if intended_side == Side.SELL:
        if not gold_bear:
            return False, f"Gold not bearish: Close={gold_close} (EMA20={gold_ema20:.1f}, VWAP={gold_vwap:.1f})"
        if not silver_bear:
            return False, f"Silver not bearish: Close={silver_close} (EMA20={silver_ema20:.1f}, VWAP={silver_vwap:.1f})"
        return True, "Bullion alignment confirmed (Gold & Silver Bearish)"

    if (gold_bull and silver_bull) or (gold_bear and silver_bear):
        return True, "Bullion alignment confirmed"

    return False, f"Bullion cross-metal divergence: Gold(bull={gold_bull}, bear={gold_bear}) vs Silver(bull={silver_bull}, bear={silver_bear})"


@dataclass
class BenchmarkSnapshot:
    close: float
    ema20: float
    prev_ema20: float
    session_vwap: float
    adx14: float
    prev_adx14: float
    ema50: float = 0.0
    ci14: Optional[float] = None
    sigma_vwap: Optional[float] = None


@dataclass(frozen=True)
class RegimeEvaluation:
    """Detailed evaluation result for market regime."""
    regime: MarketRegime
    gate: GateResult
    current_close: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_slope_val: Optional[float] = None
    adx_val: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None


def evaluate_market_regime(
    target: Union[BenchmarkSnapshot, list[Candle]],
    cfg_or_config: Union[dict, RegimeConfig],
    benchmark_quote: Optional[Quote] = None,
    stale_threshold_sec: float = 2.0,
    candles_15m: Optional[list[Candle]] = None,
) -> Union[MarketRegime, RegimeEvaluation]:
    """Evaluate market regime.

    Supports:
      1. evaluate_market_regime(snapshot: BenchmarkSnapshot, cfg: dict) -> MarketRegime
      2. evaluate_market_regime(candles_5m: list[Candle], config: RegimeConfig) -> RegimeEvaluation
    """
    if isinstance(target, BenchmarkSnapshot):
        cfg = cfg_or_config if isinstance(cfg_or_config, dict) else {}
        regime_cfg = cfg.get("regime_filter", cfg.get("regime", {}))
        min_adx = float(regime_cfg.get("min_adx", 22.0))
        require_vwap = bool(regime_cfg.get("require_vwap_alignment", True))
        require_rising_adx = bool(regime_cfg.get("require_rising_adx", True))

        # Choppiness check on snapshot if present
        ci_val = getattr(target, "ci14", None) or getattr(target, "choppiness_index", None)
        if ci_val is not None and float(ci_val) >= 50.0:
            return MarketRegime.WAITING

        # Low ADX check (< 21.0 indicates congestion / waiting)
        if getattr(target, "adx14", 25.0) < 21.0:
            return MarketRegime.WAITING

        # VWAP +/- 0.30 SD equilibrium check
        sigma_vwap = getattr(target, "sigma_vwap", None) or getattr(target, "vwap_sd", None)
        if sigma_vwap is not None and float(sigma_vwap) > 0 and target.session_vwap > 0:
            if abs(target.close - target.session_vwap) <= 0.30 * float(sigma_vwap):
                return MarketRegime.WAITING

        snapshot_slope = target.ema20 - target.prev_ema20
        adx_strong = target.adx14 >= min_adx
        adx_rising = (target.adx14 > target.prev_adx14) if require_rising_adx else True

        if not (adx_strong and adx_rising):
            return MarketRegime.NEUTRAL

        # Bullish Regime Check: Close > EMA20 > EMA50, EMA20 Slope > 0, VWAP aligned
        vwap_bull_ok = (target.close > target.session_vwap) if (require_vwap and target.session_vwap > 0) else True
        ema50_bull_ok = (target.ema20 > target.ema50) if target.ema50 > 0 else True
        if target.close > target.ema20 and ema50_bull_ok and snapshot_slope > 0 and vwap_bull_ok:
            return MarketRegime.BULLISH

        # Bearish Regime Check: Close < EMA20 < EMA50, EMA20 Slope < 0, VWAP aligned
        vwap_bear_ok = (target.close < target.session_vwap) if (require_vwap and target.session_vwap > 0) else True
        ema50_bear_ok = (target.ema20 < target.ema50) if target.ema50 > 0 else True
        if target.close < target.ema20 and ema50_bear_ok and snapshot_slope < 0 and vwap_bear_ok:
            return MarketRegime.BEARISH

        return MarketRegime.NEUTRAL

    # Candle series processing
    candles_5m = target
    config = cfg_or_config if isinstance(cfg_or_config, RegimeConfig) else RegimeConfig()

    # Enforce Choppiness Index Gate (< 50.0) if 15m candles provided
    if candles_15m:
        completed_15m = [c for c in candles_15m if c.is_complete]
        if len(completed_15m) >= 14:
            ci_14 = calculate_choppiness_index(completed_15m, period=14)
            if ci_14 >= 50.0:
                return RegimeEvaluation(
                    regime=MarketRegime.WAITING,
                    gate=GateResult(
                        name="CHOPPINESS_INDEX",
                        status=GateStatus.FAIL,
                        reason=f"Choppiness Index ({ci_14:.1f}) >= 50.0 indicates choppy/sideways regime; entering REGIME.WAITING",
                        data={"ci_14": round(ci_14, 2), "max_allowed": 50.0},
                    ),
                )

    # Check quote freshness if provided
    if benchmark_quote is not None:
        if not benchmark_quote.is_fresh:
            return RegimeEvaluation(
                regime=MarketRegime.NEUTRAL,
                gate=GateResult(
                    name="MARKET_REGIME",
                    status=GateStatus.FAIL,
                    reason="Benchmark quote is marked stale",
                    data={"code": "STALE_BENCHMARK_DATA"},
                ),
            )

    min_required_bars = max(config.ema_period + config.ema_slope_lookback + 2, config.adx_period * 2 + 5)
    completed_candles = [c for c in candles_5m if c.is_complete]

    if len(completed_candles) < min_required_bars:
        return RegimeEvaluation(
            regime=MarketRegime.NEUTRAL,
            gate=GateResult(
                name="MARKET_REGIME",
                status=GateStatus.FAIL,
                reason=f"Insufficient completed 5m benchmark candles ({len(completed_candles)}/{min_required_bars})",
                data={"code": "INSUFFICIENT_BENCHMARK_DATA", "count": len(completed_candles)},
            ),
        )

    closes = [c.close for c in completed_candles]
    highs = [c.high for c in completed_candles]
    lows = [c.low for c in completed_candles]

    ema_series = compute_ema(closes, period=config.ema_period)
    if not ema_series or len(ema_series) < config.ema_slope_lookback:
        return RegimeEvaluation(
            regime=MarketRegime.NEUTRAL,
            gate=GateResult(
                name="MARKET_REGIME",
                status=GateStatus.FAIL,
                reason="Could not compute EMA series",
                data={"code": "EMA_CALCULATION_ERROR"},
            ),
        )

    latest_close = closes[-1]
    latest_ema = ema_series[-1]
    slope = ema_slope(ema_series, lookback=config.ema_slope_lookback)

    # Compute EMA50 if configured and sufficient candles are available
    slow_period = getattr(config, "ema_slow_period", 50)
    latest_ema50: Optional[float] = None
    dual_ema_bull = True
    dual_ema_bear = True
    if len(closes) >= slow_period:
        ema_slow_series = compute_ema(closes, period=slow_period)
        if ema_slow_series:
            latest_ema50 = ema_slow_series[-1]
            dual_ema_bull = (latest_ema > latest_ema50)
            dual_ema_bear = (latest_ema < latest_ema50)

    adx_dict = compute_adx(highs, lows, closes, period=config.adx_period)
    adx_series = adx_dict.get("adx", [])
    plus_di_series = adx_dict.get("plus_di", [])
    minus_di_series = adx_dict.get("minus_di", [])

    if not adx_series:
        return RegimeEvaluation(
            regime=MarketRegime.NEUTRAL,
            gate=GateResult(
                name="MARKET_REGIME",
                status=GateStatus.FAIL,
                reason="Could not compute ADX series",
                data={"code": "ADX_CALCULATION_ERROR"},
            ),
        )

    latest_adx = adx_series[-1]
    prev_adx = adx_series[-2] if len(adx_series) >= 2 else latest_adx
    latest_plus_di = plus_di_series[-1] if plus_di_series else None
    latest_minus_di = minus_di_series[-1] if minus_di_series else None

    # Congestion filter: ADX < 21.0 triggers WAITING
    if latest_adx < 21.0:
        return RegimeEvaluation(
            regime=MarketRegime.WAITING,
            gate=GateResult(
                name="MARKET_REGIME",
                status=GateStatus.FAIL,
                reason=f"ADX ({latest_adx:.1f}) < 21.0 indicates congestion/absence of trend; entering REGIME.WAITING",
                data={
                    "code": "ADX_BELOW_21",
                    "adx": round(latest_adx, 2),
                    "min_adx": 21.0,
                    "close": latest_close,
                },
            ),
            current_close=latest_close,
            ema_20=latest_ema,
            ema_50=latest_ema50,
            ema_slope_val=slope,
            adx_val=latest_adx,
            plus_di=latest_plus_di,
            minus_di=latest_minus_di,
        )

    if getattr(config, "require_rising_adx", True):
        adx_rising = (latest_adx > prev_adx) if latest_adx < 28.0 else True
    else:
        adx_rising = True
    is_trending = (latest_adx >= config.min_adx) and adx_rising

    is_bullish = (latest_close > latest_ema) and dual_ema_bull and (slope > 0) and is_trending
    is_bearish = (latest_close < latest_ema) and dual_ema_bear and (slope < 0) and is_trending

    # Directional Indicator check (+DI vs -DI)
    if latest_plus_di is not None and latest_minus_di is not None:
        if is_bullish and latest_plus_di <= latest_minus_di:
            is_bullish = False
        if is_bearish and latest_minus_di <= latest_plus_di:
            is_bearish = False

    # Session VWAP / TWAP alignment
    if getattr(config, "require_vwap_alignment", True) and len(completed_candles) >= 3:
        target_date = completed_candles[-1].timestamp.date()
        today_candles = [c for c in completed_candles if c.timestamp.date() == target_date]
        if today_candles:
            tot_vol = sum(c.volume for c in today_candles)
            if tot_vol > 0:
                from apex3.indicators.vwap import compute_session_vwap
                vwap_val = compute_session_vwap(
                    [c.high for c in today_candles],
                    [c.low for c in today_candles],
                    [c.close for c in today_candles],
                    [c.volume for c in today_candles],
                )
            else:
                # Zero-volume index spot candles: compute Typical-Price Session TWAP
                vwap_val = sum((c.high + c.low + c.close) / 3.0 for c in today_candles) / len(today_candles)

            # Check VWAP +/- 0.30 SD equilibrium band
            _, sigma_vwap = calculate_session_anchored_vwap(today_candles)
            if sigma_vwap > 0 and vwap_val > 0 and abs(latest_close - vwap_val) <= 0.30 * sigma_vwap:
                return RegimeEvaluation(
                    regime=MarketRegime.WAITING,
                    gate=GateResult(
                        name="MARKET_REGIME",
                        status=GateStatus.FAIL,
                        reason=f"Price ({latest_close:.2f}) inside equilibrium band VWAP +/- 0.30 SD ([{vwap_val - 0.30*sigma_vwap:.2f}, {vwap_val + 0.30*sigma_vwap:.2f}]); entering REGIME.WAITING",
                        data={
                            "code": "VWAP_EQUILIBRIUM",
                            "close": latest_close,
                            "vwap": round(vwap_val, 2),
                            "sigma_vwap": round(sigma_vwap, 2),
                        },
                    ),
                    current_close=latest_close,
                    ema_20=latest_ema,
                    ema_50=latest_ema50,
                    ema_slope_val=slope,
                    adx_val=latest_adx,
                    plus_di=latest_plus_di,
                    minus_di=latest_minus_di,
                )

            if vwap_val > 0:
                if is_bullish and latest_close <= vwap_val:
                    is_bullish = False
                if is_bearish and latest_close >= vwap_val:
                    is_bearish = False

    details = {
        "close": latest_close,
        "ema": round(latest_ema, 2),
        "ema50": round(latest_ema50, 2) if latest_ema50 is not None else None,
        "slope": slope,
        "adx": round(latest_adx, 2),
        "prev_adx": round(prev_adx, 2),
        "min_adx": config.min_adx,
    }

    if is_bullish:
        ema50_str = f" > EMA-50 ({latest_ema50:.2f})" if latest_ema50 is not None else ""
        return RegimeEvaluation(
            regime=MarketRegime.BULLISH,
            gate=GateResult(
                name="MARKET_REGIME",
                status=GateStatus.PASS,
                reason=f"Bullish regime: Close ({latest_close}) > EMA-20 ({latest_ema:.2f}){ema50_str}, rising slope, ADX ({latest_adx:.1f}) >= {config.min_adx}",
                data=details,
            ),
            current_close=latest_close,
            ema_20=latest_ema,
            ema_50=latest_ema50,
            ema_slope_val=slope,
            adx_val=latest_adx,
            plus_di=latest_plus_di,
            minus_di=latest_minus_di,
        )

    if is_bearish:
        ema50_str = f" < EMA-50 ({latest_ema50:.2f})" if latest_ema50 is not None else ""
        return RegimeEvaluation(
            regime=MarketRegime.BEARISH,
            gate=GateResult(
                name="MARKET_REGIME",
                status=GateStatus.PASS,
                reason=f"Bearish regime: Close ({latest_close}) < EMA-20 ({latest_ema:.2f}){ema50_str}, falling slope, ADX ({latest_adx:.1f}) >= {config.min_adx}",
                data=details,
            ),
            current_close=latest_close,
            ema_20=latest_ema,
            ema_50=latest_ema50,
            ema_slope_val=slope,
            adx_val=latest_adx,
            plus_di=latest_plus_di,
            minus_di=latest_minus_di,
        )

    return RegimeEvaluation(
        regime=MarketRegime.NEUTRAL,
        gate=GateResult(
            name="MARKET_REGIME",
            status=GateStatus.FAIL,
            reason="Market regime is neutral (mixed signals or weak trend)",
            data=details,
        ),
        current_close=latest_close,
        ema_20=latest_ema,
        ema_50=latest_ema50,
        ema_slope_val=slope,
        adx_val=latest_adx,
        plus_di=latest_plus_di,
        minus_di=latest_minus_di,
    )
