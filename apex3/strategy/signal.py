"""APEX-3 Stock Signal Generator.

Evaluates individual stock candidates against opening range, session VWAP,
volume confirmation, relative strength vs NIFTY 50, spread, and circuit distance.

Exact mirror conditions for Long and Short:
- Long setup:
  1. Completed 5m close > Opening Range High (09:15-09:30 IST)
  2. Completed 5m close > Stock Session VWAP
  3. Market regime is BULLISH
  4. Relative strength > benchmark over lookback period (e.g. 20 bars)
  5. Signal bar volume >= 1.5x (min_relative_volume) 20-bar volume average
  6. Bid-Ask spread <= max_spread_bps (e.g. 8 bps)
  7. Price is at least min_circuit_distance_pct away from Upper Circuit limit
  8. Data is fresh and candle is completed

- Short setup:
  1. Completed 5m close < Opening Range Low (09:15-09:30 IST)
  2. Completed 5m close < Stock Session VWAP
  3. Market regime is BEARISH
  4. Relative weakness < benchmark over lookback period
  5. Signal bar volume >= 1.5x 20-bar volume average
  6. Bid-Ask spread <= max_spread_bps
  7. Price is at least min_circuit_distance_pct away from Lower Circuit limit
  8. Data is fresh and candle is completed
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

from apex3.config import SignalConfig
from apex3.indicators.atr import compute_atr
from apex3.indicators.vwap import compute_session_vwap
from apex3.models import (
    Candle,
    Decision,
    GateResult,
    GateStatus,
    InstrumentInfo,
    OpeningRange,
    Quote,
    Side,
    round_to_tick,
)
from apex3.strategy.regime import MarketRegime


@dataclass
class SignalEvaluation:
    """Result of signal evaluation for one symbol."""
    symbol: str
    decision: Decision  # PROPOSED_LONG, PROPOSED_SHORT, or WAIT
    side: Optional[Side] = None
    gates: list[GateResult] = None  # type: ignore[assignment]
    reason_codes: list[str] = None  # type: ignore[assignment]
    explanation: str = ""
    signal_price: float = 0.0
    or_high: float = 0.0
    or_low: float = 0.0
    vwap: float = 0.0
    relative_volume: float = 0.0
    relative_strength: float = 0.0
    circuit_distance_pct: float = 0.0

    def __post_init__(self):
        if self.gates is None:
            self.gates = []
        if self.reason_codes is None:
            self.reason_codes = []


from datetime import date, datetime, time
from zoneinfo import ZoneInfo

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


def build_opening_range(
    symbol: str,
    candles_0915_0930: list[Candle],
    session_date: Optional[date] = None,
    session_type: Literal["MORNING", "COMEX"] = "MORNING",
) -> OpeningRange:
    """Build opening range from completed candles for MORNING or COMEX session.

    - MORNING: 09:00 - 09:15 IST (or 09:15 - 09:30 for legacy equity)
    - COMEX: 17:00 - 17:30 IST (EDT) or 18:00 - 18:30 IST (EST)
    """
    completed = [c for c in candles_0915_0930 if c.is_complete]
    if not completed:
        return OpeningRange(
            symbol=symbol,
            high=0.0,
            low=0.0,
            is_complete=False,
            candle_count=0,
            session_type=session_type,
        )

    target_d = session_date or (
        completed[-1].timestamp if completed[-1].timestamp.tzinfo else completed[-1].timestamp.replace(tzinfo=IST)
    ).astimezone(IST).date()

    if session_type == "COMEX":
        if is_us_dst(target_d):
            start_t = time(17, 0)
            end_t = time(17, 30)
        else:
            start_t = time(18, 0)
            end_t = time(18, 30)
        expected_bars = 6
    else:  # MORNING
        has_0900 = any(
            (c.timestamp if c.timestamp.tzinfo else c.timestamp.replace(tzinfo=IST)).astimezone(IST).time() < time(9, 15)
            for c in completed
        )
        if has_0900:
            start_t = time(9, 0)
            end_t = time(9, 15)
        else:
            start_t = time(9, 15)
            end_t = time(9, 30)
        expected_bars = 3

    filtered = []
    for c in completed:
        c_dt = (c.timestamp if c.timestamp.tzinfo else c.timestamp.replace(tzinfo=IST)).astimezone(IST)
        if c_dt.date() == target_d and start_t <= c_dt.time() < end_t:
            filtered.append(c)

    if not filtered:
        return OpeningRange(
            symbol=symbol,
            high=0.0,
            low=0.0,
            is_complete=False,
            candle_count=0,
            session_type=session_type,
        )

    high = max(c.high for c in filtered)
    low = min(c.low for c in filtered)
    is_complete = len(filtered) >= expected_bars

    return OpeningRange(
        symbol=symbol,
        high=high,
        low=low,
        is_complete=is_complete,
        candle_count=len(filtered),
        session_type=session_type,
    )


build_mcx_opening_range = build_opening_range


def compute_relative_strength(
    stock_closes: list[float],
    benchmark_closes: list[float],
    lookback: int = 20,
) -> float:
    """Calculate relative performance of stock vs benchmark over lookback bars.

    Returns:
        stock_pct_return - benchmark_pct_return (positive = stock outperforming).
    """
    if len(stock_closes) < lookback or len(benchmark_closes) < lookback:
        return 0.0

    stock_ret = (stock_closes[-1] - stock_closes[-lookback]) / stock_closes[-lookback]
    bench_ret = (benchmark_closes[-1] - benchmark_closes[-lookback]) / benchmark_closes[-lookback]
    return (stock_ret - bench_ret) * 100.0


def evaluate_signal(
    symbol: str,
    stock_candles_5m: list[Candle],
    benchmark_candles_5m: list[Candle],
    opening_range: OpeningRange,
    market_regime: MarketRegime,
    instrument: InstrumentInfo,
    quote: Optional[Quote],
    config: SignalConfig,
    is_entry_window: bool = True,
    is_core_symbol: bool = True,
) -> SignalEvaluation:
    """Evaluate a single stock for entry signal across all 8 gates.

    Fails closed: Any failed gate or missing/stale data results in WAIT.
    """
    gates: list[GateResult] = []
    reason_codes: list[str] = []

    # 0. Session Entry Window Gate
    if not is_entry_window:
        gates.append(
            GateResult(
                name="ENTRY_WINDOW",
                status=GateStatus.FAIL,
                reason="Current session state does not permit new entries",
                data={"code": "OUTSIDE_ENTRY_WINDOW"},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["OUTSIDE_ENTRY_WINDOW"],
            explanation="Outside active entry window (09:30 - 14:15 IST)",
        )
    gates.append(GateResult(name="ENTRY_WINDOW", status=GateStatus.PASS, reason="Inside entry window"))

    # 1. Opening Range Gate
    if not opening_range.is_complete or opening_range.high <= opening_range.low:
        gates.append(
            GateResult(
                name="OPENING_RANGE",
                status=GateStatus.FAIL,
                reason=f"Opening range incomplete (count={opening_range.candle_count})",
                data={"code": "INCOMPLETE_OPENING_RANGE"},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["INCOMPLETE_OPENING_RANGE"],
            explanation=f"Opening range for {symbol} is incomplete or invalid",
        )
    gates.append(
        GateResult(
            name="OPENING_RANGE",
            status=GateStatus.PASS,
            reason=f"Opening range complete: H={opening_range.high:.2f}, L={opening_range.low:.2f}",
        )
    )

    # 2. Candle Data Freshness Gate
    completed_candles = [c for c in stock_candles_5m if c.is_complete]
    if not completed_candles:
        gates.append(
            GateResult(
                name="CANDLE_DATA",
                status=GateStatus.FAIL,
                reason="No completed 5m candles available",
                data={"code": "NO_COMPLETED_CANDLES"},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["NO_COMPLETED_CANDLES"],
            explanation="No completed 5m bars available for evaluation",
        )

    latest_bar = completed_candles[-1]
    signal_price = latest_bar.close

    # Check Quote Freshness, Source Timestamp, and Spread
    if quote is None:
        gates.append(
            GateResult(
                name="QUOTE_FRESHNESS",
                status=GateStatus.FAIL,
                reason="Live quote is missing; failing closed",
                data={"code": "MISSING_QUOTE"},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["MISSING_QUOTE"],
            explanation="Live quote required for signal evaluation",
        )

    # 0. Product Eligibility & Instrument Metadata Verification
    if not getattr(instrument, "mis_available", True):
        gates.append(
            GateResult(
                name="PRODUCT_ELIGIBILITY",
                status=GateStatus.FAIL,
                reason=f"MIS product is not permitted/available for {symbol}",
                data={"mis_available": False},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["MIS_NOT_AVAILABLE"],
            explanation=f"MIS product is not permitted for {symbol}",
        )

    if not quote.is_fresh:
        gates.append(
            GateResult(
                name="QUOTE_FRESHNESS",
                status=GateStatus.FAIL,
                reason="Quote data is marked stale or lacks depth",
                data={"code": "STALE_QUOTE"},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["STALE_QUOTE"],
            explanation="Quote data exceeds staleness threshold or lacks depth",
        )

    if quote.timestamp is None:
        gates.append(
            GateResult(
                name="QUOTE_FRESHNESS",
                status=GateStatus.FAIL,
                reason="Quote source timestamp is missing; failing closed",
                data={"code": "MISSING_QUOTE_TIMESTAMP"},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["STALE_QUOTE", "MISSING_QUOTE_TIMESTAMP"],
            explanation="Live quote timestamp required for exchange clock validation",
        )

    # Validate source quote timestamp age against exchange clock
    now_ist = datetime.now(IST)
    q_ts = quote.timestamp if quote.timestamp.tzinfo else quote.timestamp.replace(tzinfo=IST)
    eval_clock = now_ist if (now_ist.date() == q_ts.date()) else (latest_bar.timestamp if latest_bar.timestamp.tzinfo else latest_bar.timestamp.replace(tzinfo=IST))
    clock_diff = (eval_clock - q_ts).total_seconds()
    age_sec = clock_diff
    if clock_diff < -15.0:
        gates.append(
            GateResult(
                name="QUOTE_FRESHNESS",
                status=GateStatus.FAIL,
                reason=f"Quote timestamp ({q_ts}) is in the future relative to exchange clock ({clock_diff:.1f}s)",
                data={"code": "FUTURE_QUOTE_TIMESTAMP", "diff_sec": clock_diff},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["FUTURE_QUOTE_TIMESTAMP"],
            explanation=f"Quote timestamp is in the future by {-clock_diff:.1f}s",
        )
    max_quote_age = float(getattr(config, "max_quote_age_sec", 15.0))
    if clock_diff > max_quote_age:
        gates.append(
            GateResult(
                name="QUOTE_FRESHNESS",
                status=GateStatus.FAIL,
                reason=f"Quote timestamp age ({clock_diff:.1f}s) exceeds max threshold ({max_quote_age:.1f}s)",
                data={"code": "STALE_QUOTE", "age_sec": clock_diff, "threshold_sec": max_quote_age},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["STALE_QUOTE"],
            explanation=f"Quote timestamp is {clock_diff:.1f}s old (max allowed: {max_quote_age:.1f}s)",
        )

    # Spread and market depth check
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        gates.append(
            GateResult(
                name="SPREAD",
                status=GateStatus.FAIL,
                reason=f"Invalid market depth: bid={quote.bid:.2f}, ask={quote.ask:.2f}",
                data={"bid": quote.bid, "ask": quote.ask},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["INVALID_DEPTH"],
            explanation=f"Invalid quote depth: bid={quote.bid:.2f}, ask={quote.ask:.2f}",
        )

    mid = (quote.bid + quote.ask) / 2.0
    spread_bps = ((quote.ask - quote.bid) / mid) * 10000.0
    if spread_bps > config.max_spread_bps:
        gates.append(
            GateResult(
                name="SPREAD",
                status=GateStatus.FAIL,
                reason=f"Spread {spread_bps:.1f} bps > max {config.max_spread_bps} bps",
                data={"spread_bps": spread_bps, "max_bps": config.max_spread_bps},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["EXCESSIVE_SPREAD"],
            explanation=f"Spread {spread_bps:.1f} bps exceeds limit {config.max_spread_bps} bps",
        )

    gates.append(GateResult(name="SPREAD", status=GateStatus.PASS, reason=f"Spread {spread_bps:.1f} bps acceptable"))
    gates.append(GateResult(name="QUOTE_FRESHNESS", status=GateStatus.PASS, reason=f"Quote data fresh ({age_sec:.1f}s age)"))

    # 3. Market Regime Alignment Gate
    if market_regime == MarketRegime.NEUTRAL:
        gates.append(
            GateResult(
                name="REGIME_ALIGNMENT",
                status=GateStatus.FAIL,
                reason="Market regime is NEUTRAL; entries prohibited",
                data={"code": "MARKET_REGIME_NEUTRAL"},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["MARKET_REGIME_NEUTRAL"],
            explanation="Broad-market regime is neutral or mixed",
        )

    target_side = Side.BUY if market_regime == MarketRegime.BULLISH else Side.SELL
    gates.append(
        GateResult(
            name="REGIME_ALIGNMENT",
            status=GateStatus.PASS,
            reason=f"Regime aligns with target side: {target_side.value}",
        )
    )

    # 4. Opening Range Breakout Gate (with Breakout Cushion & CLV Conviction)
    cushion_mult = getattr(config, "min_breakout_cushion_atr", 0.15)
    min_clv = getattr(config, "min_clv_ratio", 0.70) if is_core_symbol else float(getattr(config, "probation_min_clv_ratio", 0.75))

    closes = [c.close for c in completed_candles]
    highs = [c.high for c in completed_candles]
    lows = [c.low for c in completed_candles]
    atr_series = compute_atr(highs, lows, closes, period=14)
    atr14 = atr_series[-1] if atr_series else max(latest_bar.high - latest_bar.low, 0.50)
    cushion = cushion_mult * atr14
    bar_range = max(latest_bar.high - latest_bar.low, 0.05)

    # 4A. Opening Range Width Cap (max 1.05%)
    if opening_range.low > 0:
        or_width_pct = (opening_range.high - opening_range.low) / opening_range.low
        if or_width_pct > 0.0105:
            gates.append(
                GateResult(
                    name="OR_WIDTH",
                    status=GateStatus.FAIL,
                    reason=f"Opening Range width ({or_width_pct*100:.2f}%) exceeds max 1.05%",
                    data={"or_width_pct": or_width_pct},
                )
            )
            return SignalEvaluation(
                symbol=symbol,
                decision=Decision.WAIT,
                gates=gates,
                reason_codes=["OR_WIDTH_EXCEEDED"],
                explanation=f"Opening Range width {or_width_pct*100:.2f}% > 1.05%",
            )
    gates.append(GateResult(name="OR_WIDTH", status=GateStatus.PASS, reason="Opening Range width acceptable (<= 1.05%)"))

    # 4B. Minimum Volatility Floor (ATR14 / Price >= min_atr_pct)
    min_atr = float(getattr(config, "min_atr_pct", 0.0))
    if min_atr > 0 and signal_price > 0 and (atr14 / signal_price) < min_atr:
        gates.append(
            GateResult(
                name="MIN_VOLATILITY",
                status=GateStatus.FAIL,
                reason=f"ATR/Price ratio ({atr14/signal_price*100:.3f}%) < min {min_atr*100:.3f}%",
                data={"atr_pct": atr14 / signal_price},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["INSUFFICIENT_VOLATILITY"],
            explanation=f"ATR/Price {atr14/signal_price*100:.3f}% is below minimum required {min_atr*100:.3f}%",
        )
    gates.append(GateResult(name="MIN_VOLATILITY", status=GateStatus.PASS, reason="Volatility confirmed"))

    # 4C. ADR Exhaustion Gate (Today's range consumed <= 60% of ADR14)
    current_date = latest_bar.timestamp.date()
    today_candles = [c for c in completed_candles if c.timestamp.date() == current_date]
    session_high = max(c.high for c in today_candles) if today_candles else latest_bar.high
    session_low = min(c.low for c in today_candles) if today_candles else latest_bar.low

    adr14_pct = getattr(config, "adr14_pct", 0.02)
    prior_dates = sorted(list(set(c.timestamp.date() for c in completed_candles if c.timestamp.date() < current_date)))
    if len(prior_dates) >= 2:
        daily_ranges = []
        for d in prior_dates[-14:]:
            day_c = [c for c in completed_candles if c.timestamp.date() == d]
            if day_c:
                d_high = max(c.high for c in day_c)
                d_low = min(c.low for c in day_c)
                if d_low > 0:
                    daily_ranges.append((d_high - d_low) / d_low)
        if daily_ranges:
            adr14_pct = sum(daily_ranges) / len(daily_ranges)

    max_adr_mult = float(getattr(config, "max_adr_consumed_pct", 0.60))
    min_adr_base = float(getattr(config, "min_adr_baseline", 0.025))
    effective_adr = max(min_adr_base, adr14_pct)
    if session_low > 0 and effective_adr > 0:
        today_range_pct = (session_high - session_low) / session_low
        if today_range_pct > max_adr_mult * effective_adr:
            gates.append(
                GateResult(
                    name="ADR_EXHAUSTION",
                    status=GateStatus.FAIL,
                    reason=f"Today's range ({today_range_pct*100:.2f}%) has consumed > 60% of ADR ({effective_adr*100:.2f}%)",
                    data={"today_range_pct": today_range_pct, "adr14_pct": effective_adr},
                )
            )
            return SignalEvaluation(
                symbol=symbol,
                decision=Decision.WAIT,
                gates=gates,
                reason_codes=["ADR_EXHAUSTED"],
                explanation=f"Today's range {today_range_pct*100:.2f}% exceeds 60% of ADR14",
            )
    gates.append(GateResult(name="ADR_EXHAUSTION", status=GateStatus.PASS, reason="ADR exhaustion check passed (<= 60% of ADR)"))

    if target_side == Side.BUY:
        required_long = opening_range.high + cushion
        if signal_price < required_long:
            gates.append(
                GateResult(
                    name="OR_BREAKOUT",
                    status=GateStatus.FAIL,
                    reason=f"Close ({signal_price:.2f}) < OR High + Cushion ({required_long:.2f}, cushion={cushion:.2f})",
                    data={"close": signal_price, "or_high": opening_range.high, "cushion": cushion},
                )
            )
            return SignalEvaluation(
                symbol=symbol,
                decision=Decision.WAIT,
                gates=gates,
                reason_codes=["NO_OR_HIGH_BREAKOUT"],
                explanation=f"Price {signal_price:.2f} has not cleared OR high + cushion {required_long:.2f}",
            )
        clv = (latest_bar.close - latest_bar.low) / bar_range
        if clv < min_clv:
            gates.append(
                GateResult(
                    name="OR_BREAKOUT",
                    status=GateStatus.FAIL,
                    reason=f"CLV ({clv:.2f}) < required {min_clv:.2f} (weak close in bar range)",
                    data={"clv": clv, "min_clv": min_clv},
                )
            )
            return SignalEvaluation(
                symbol=symbol,
                decision=Decision.WAIT,
                gates=gates,
                reason_codes=["WEAK_CANDLE_CLOSE_LOCATION"],
                explanation=f"Candle CLV {clv:.2f} below minimum conviction {min_clv:.2f}",
            )
    else:  # Side.SELL
        required_short = opening_range.low - cushion
        if signal_price > required_short:
            gates.append(
                GateResult(
                    name="OR_BREAKOUT",
                    status=GateStatus.FAIL,
                    reason=f"Close ({signal_price:.2f}) > OR Low - Cushion ({required_short:.2f}, cushion={cushion:.2f})",
                    data={"close": signal_price, "or_low": opening_range.low, "cushion": cushion},
                )
            )
            return SignalEvaluation(
                symbol=symbol,
                decision=Decision.WAIT,
                gates=gates,
                reason_codes=["NO_OR_LOW_BREAKOUT"],
                explanation=f"Price {signal_price:.2f} has not broken below OR low - cushion {required_short:.2f}",
            )
        clv = (latest_bar.high - latest_bar.close) / bar_range
        if clv < min_clv:
            gates.append(
                GateResult(
                    name="OR_BREAKOUT",
                    status=GateStatus.FAIL,
                    reason=f"CLV ({clv:.2f}) < required {min_clv:.2f} (weak close in bar range)",
                    data={"clv": clv, "min_clv": min_clv},
                )
            )
            return SignalEvaluation(
                symbol=symbol,
                decision=Decision.WAIT,
                gates=gates,
                reason_codes=["WEAK_CANDLE_CLOSE_LOCATION"],
                explanation=f"Candle CLV {clv:.2f} below minimum conviction {min_clv:.2f}",
            )
    gates.append(GateResult(name="OR_BREAKOUT", status=GateStatus.PASS, reason="Confirmed OR breakout with cushion & CLV conviction on completed bar"))

    # 5. Session VWAP Alignment Gate
    session_candles = today_candles if today_candles else completed_candles
    session_highs = [c.high for c in session_candles]
    session_lows = [c.low for c in session_candles]
    session_closes = [c.close for c in session_candles]
    session_volumes = [c.volume for c in session_candles]

    stock_vwap = compute_session_vwap(session_highs, session_lows, session_closes, session_volumes)

    if target_side == Side.BUY:
        if signal_price <= stock_vwap:
            gates.append(
                GateResult(
                    name="VWAP_ALIGNMENT",
                    status=GateStatus.FAIL,
                    reason=f"Close ({signal_price:.2f}) <= VWAP ({stock_vwap:.2f})",
                    data={"close": signal_price, "vwap": stock_vwap},
                )
            )
            return SignalEvaluation(
                symbol=symbol,
                decision=Decision.WAIT,
                gates=gates,
                reason_codes=["BELOW_VWAP"],
                explanation=f"Long requires price ({signal_price:.2f}) > session VWAP ({stock_vwap:.2f})",
            )
    else:  # Side.SELL
        if signal_price >= stock_vwap:
            gates.append(
                GateResult(
                    name="VWAP_ALIGNMENT",
                    status=GateStatus.FAIL,
                    reason=f"Close ({signal_price:.2f}) >= VWAP ({stock_vwap:.2f})",
                    data={"close": signal_price, "vwap": stock_vwap},
                )
            )
            return SignalEvaluation(
                symbol=symbol,
                decision=Decision.WAIT,
                gates=gates,
                reason_codes=["ABOVE_VWAP"],
                explanation=f"Short requires price ({signal_price:.2f}) < session VWAP ({stock_vwap:.2f})",
            )
    gates.append(GateResult(name="VWAP_ALIGNMENT", status=GateStatus.PASS, reason=f"VWAP aligned ({stock_vwap:.2f})"))

    # 6. Relative Volume Confirmation Gate
    if len(completed_candles) >= config.volume_avg_period + 1:
        prior_volumes = [c.volume for c in completed_candles[-config.volume_avg_period - 1 : -1]]
        avg_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 1.0
    else:
        all_prior = [c.volume for c in completed_candles[:-1]]
        avg_volume = sum(all_prior) / len(all_prior) if all_prior else 1.0

    current_volume = latest_bar.volume
    relative_vol = current_volume / avg_volume if avg_volume > 0 else 1.0

    min_vol = getattr(config, "min_relative_volume", 2.0) if is_core_symbol else float(getattr(config, "probation_volume_surge_ratio", 2.5))
    if relative_vol < min_vol:
        gates.append(
            GateResult(
                name="VOLUME_CONFIRMATION",
                status=GateStatus.FAIL,
                reason=f"Volume multiple {relative_vol:.2f}x < required {min_vol:.2f}x",
                data={"relative_volume": relative_vol, "min_required": min_vol},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["INSUFFICIENT_VOLUME"],
            explanation=f"Volume {relative_vol:.2f}x below required threshold {min_vol:.2f}x",
        )
    gates.append(
        GateResult(
            name="VOLUME_CONFIRMATION",
            status=GateStatus.PASS,
            reason=f"Volume confirmed ({relative_vol:.2f}x >= {min_vol:.2f}x)",
        )
    )

    # 7. Relative Strength / Weakness Gate
    bench_closes = [c.close for c in benchmark_candles_5m if c.is_complete]
    stock_closes = [c.close for c in completed_candles]
    rel_perf = compute_relative_strength(stock_closes, bench_closes, lookback=config.relative_strength_lookback)

    if target_side == Side.BUY and rel_perf <= 0.0:
        gates.append(
            GateResult(
                name="RELATIVE_STRENGTH",
                status=GateStatus.FAIL,
                reason=f"Stock relative performance ({rel_perf:+.2f}%) <= 0 vs benchmark",
                data={"rel_perf": rel_perf},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["WEAKER_THAN_BENCHMARK"],
            explanation=f"Stock is underperforming NIFTY 50 ({rel_perf:+.2f}%) over lookback",
        )
    elif target_side == Side.SELL and rel_perf >= 0.0:
        gates.append(
            GateResult(
                name="RELATIVE_STRENGTH",
                status=GateStatus.FAIL,
                reason=f"Stock relative performance ({rel_perf:+.2f}%) >= 0 vs benchmark",
                data={"rel_perf": rel_perf},
            )
        )
        return SignalEvaluation(
            symbol=symbol,
            decision=Decision.WAIT,
            gates=gates,
            reason_codes=["STRONGER_THAN_BENCHMARK"],
            explanation=f"Stock is outperforming NIFTY 50 ({rel_perf:+.2f}%) over lookback (short requires weakness)",
        )
    gates.append(
        GateResult(
            name="RELATIVE_STRENGTH",
            status=GateStatus.PASS,
            reason=f"Relative performance aligned: {rel_perf:+.2f}%",
        )
    )

    # 8. Circuit Limit Distance Gate
    if target_side == Side.BUY:
        if instrument.upper_circuit > 0.0:
            circuit_dist = ((instrument.upper_circuit - signal_price) / signal_price) * 100.0
            if circuit_dist < config.min_circuit_distance_pct:
                gates.append(
                    GateResult(
                        name="CIRCUIT_DISTANCE",
                        status=GateStatus.FAIL,
                        reason=f"Upper circuit distance {circuit_dist:.2f}% < min {config.min_circuit_distance_pct}%",
                        data={"dist_pct": circuit_dist, "upper_circuit": instrument.upper_circuit},
                    )
                )
                return SignalEvaluation(
                    symbol=symbol,
                    decision=Decision.WAIT,
                    gates=gates,
                    reason_codes=["TOO_CLOSE_TO_CIRCUIT"],
                    explanation=f"Upper circuit is only {circuit_dist:.2f}% away",
                )
        else:
            circuit_dist = 100.0
    else:
        if instrument.lower_circuit > 0.0:
            circuit_dist = ((signal_price - instrument.lower_circuit) / signal_price) * 100.0
            if circuit_dist < config.min_circuit_distance_pct:
                gates.append(
                    GateResult(
                        name="CIRCUIT_DISTANCE",
                        status=GateStatus.FAIL,
                        reason=f"Lower circuit distance {circuit_dist:.2f}% < min {config.min_circuit_distance_pct}%",
                        data={"dist_pct": circuit_dist, "lower_circuit": instrument.lower_circuit},
                    )
                )
                return SignalEvaluation(
                    symbol=symbol,
                    decision=Decision.WAIT,
                    gates=gates,
                    reason_codes=["TOO_CLOSE_TO_CIRCUIT"],
                    explanation=f"Lower circuit is only {circuit_dist:.2f}% away",
                )
        else:
            circuit_dist = 100.0
    gates.append(GateResult(name="CIRCUIT_DISTANCE", status=GateStatus.PASS, reason=f"Circuit distance safe ({circuit_dist:.2f}%)"))

    # All 8 gates passed!
    decision = Decision.PROPOSED_LONG if target_side == Side.BUY else Decision.PROPOSED_SHORT
    return SignalEvaluation(
        symbol=symbol,
        decision=decision,
        side=target_side,
        gates=gates,
        reason_codes=["ALL_GATES_PASSED"],
        explanation=f"All criteria met for {decision.value} on {symbol}",
        signal_price=signal_price,
        or_high=opening_range.high,
        or_low=opening_range.low,
        vwap=stock_vwap,
        relative_volume=relative_vol,
        relative_strength=rel_perf,
        circuit_distance_pct=circuit_dist,
    )


# ---------------------------------------------------------------------------
# Cross-Sectional Radar & Gate 9 Extensions
# ---------------------------------------------------------------------------

@dataclass
class StockCandidateBar:
    symbol: str
    timestamp_hhmm: str
    close: float
    or_high: float                  # 09:15 - 09:30 Opening Range High
    or_low: float                   # 09:15 - 09:30 Opening Range Low
    session_vwap: float
    stock_return_20: float          # 20-bar % return of stock
    nifty_return_20: float          # 20-bar % return of NIFTY 50
    bar_volume: float
    avg_volume_20: float
    bid_ask_spread_bps: float
    upper_circuit: float
    lower_circuit: float
    swing_low_5: float
    swing_high_5: float
    atr14: float
    session_high_so_far: float
    session_low_so_far: float
    adr14_pct: float                # 14-day Average Daily Range as decimal (e.g., 0.022 = 2.2%)
    bar_high: float = 0.0
    bar_low: float = 0.0
    tick_size: float = 0.05


@dataclass
class TradeSignal:
    symbol: str
    side: str                       # "LONG" or "SHORT"
    entry_price: float
    levels: "StopTargetLevels"
    volume_surge_ratio: float
    relative_strength: float
    conviction_score: float


@dataclass(frozen=True)
class LateStartValidityResult:
    """Explicit validity evaluation for a setup that triggered before or at late-start time."""
    is_valid: bool
    reason_code: str
    explanation: str
    drift_pct: float = 0.0


def _ensure_ist_candle_ts(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)



def validate_late_start_signal(
    symbol: str,
    side: Side,
    signal_price: float,
    initial_stop: float,
    target_price: float,
    or_high: float,
    or_low: float,
    trigger_bar_timestamp: datetime,
    subsequent_candles: list[Candle],
    quote: Optional[Quote],
    now_ist: datetime,
    entry_start_hhmm: str = "09:35",
    entry_cutoff_hhmm: str = "12:30",
    max_price_drift_pct: float = 0.0015,
) -> LateStartValidityResult:
    """Validate whether a setup that triggered at or before `now_ist` is still valid to enter.

    Rules enforced:
    1. Current time `now_ist` must be inside the allowed `[entry_start_hhmm, entry_cutoff_hhmm]` window.
    2. Live quote must be fresh and > 0.
    3. Stop-loss must NOT have been hit on any subsequent bar (`ts > trigger_bar_timestamp`) or live quote.
    4. Target must NOT have already been hit on any subsequent bar or live quote.
    5. Live price must still be outside the Opening Range breakout boundary and within `max_price_drift_pct`
       (default 0.15%) of `signal_price`.
    """
    now_norm = _ensure_ist_candle_ts(now_ist)
    trig_norm = _ensure_ist_candle_ts(trigger_bar_timestamp)
    now_hhmm = now_norm.strftime("%H:%M")

    if not (entry_start_hhmm <= now_hhmm <= entry_cutoff_hhmm):
        return LateStartValidityResult(
            is_valid=False,
            reason_code="OUTSIDE_ENTRY_WINDOW",
            explanation=f"Current time {now_hhmm} IST is outside entry window [{entry_start_hhmm}, {entry_cutoff_hhmm}]",
        )

    if quote is None or not getattr(quote, "is_fresh", False) or getattr(quote, "last_price", 0.0) <= 0:
        return LateStartValidityResult(
            is_valid=False,
            reason_code="STALE_QUOTE",
            explanation=f"Missing or stale live quote for {symbol} during late-start validity check",
        )

    ltp = float(quote.last_price)
    post_bars = [c for c in subsequent_candles if _ensure_ist_candle_ts(c.timestamp) > trig_norm]

    # Check stop-loss and target breach on all subsequent bars and live quote
    if side == Side.BUY:
        if initial_stop > 0:
            if any(c.low <= initial_stop for c in post_bars) or ltp <= initial_stop:
                return LateStartValidityResult(
                    is_valid=False,
                    reason_code="LATE_START_STOP_ALREADY_HIT",
                    explanation=f"Long setup for {symbol} invalidated: price already breached stop {initial_stop:.2f}",
                )
        if target_price > 0:
            if any(c.high >= target_price for c in post_bars) or ltp >= target_price:
                return LateStartValidityResult(
                    is_valid=False,
                    reason_code="LATE_START_TARGET_ALREADY_HIT",
                    explanation=f"Long setup for {symbol} invalidated: price already reached target {target_price:.2f}",
                )
        if or_high > 0 and ltp < or_high:
            return LateStartValidityResult(
                is_valid=False,
                reason_code="LATE_START_BREAKOUT_REVERSED",
                explanation=f"Long setup for {symbol} invalidated: live price {ltp:.2f} fell back below OR high {or_high:.2f}",
            )
    else:
        if initial_stop > 0:
            if any(c.high >= initial_stop for c in post_bars) or ltp >= initial_stop:
                return LateStartValidityResult(
                    is_valid=False,
                    reason_code="LATE_START_STOP_ALREADY_HIT",
                    explanation=f"Short setup for {symbol} invalidated: price already breached stop {initial_stop:.2f}",
                )
        if target_price > 0:
            if any(c.low <= target_price for c in post_bars) or ltp <= target_price:
                return LateStartValidityResult(
                    is_valid=False,
                    reason_code="LATE_START_TARGET_ALREADY_HIT",
                    explanation=f"Short setup for {symbol} invalidated: price already reached target {target_price:.2f}",
                )
        if or_low > 0 and ltp > or_low:
            return LateStartValidityResult(
                is_valid=False,
                reason_code="LATE_START_BREAKOUT_REVERSED",
                explanation=f"Short setup for {symbol} invalidated: live price {ltp:.2f} rose back above OR low {or_low:.2f}",
            )

    drift_pct = abs(ltp - signal_price) / signal_price if signal_price > 0 else 0.0
    if drift_pct > max_price_drift_pct + 1e-9:
        return LateStartValidityResult(
            is_valid=False,
            reason_code="LATE_START_PRICE_DRIFT_EXCEEDED",
            explanation=(
                f"Setup for {symbol} skipped: live price {ltp:.2f} drifted {drift_pct*100:.2f}% from trigger "
                f"{signal_price:.2f} (max allowed {max_price_drift_pct*100:.2f}%)"
            ),
            drift_pct=drift_pct,
        )

    return LateStartValidityResult(
        is_valid=True,
        reason_code="LATE_START_SIGNAL_VALID",
        explanation=f"Setup for {symbol} validated (drift {drift_pct*100:.2f}% <= {max_price_drift_pct*100:.2f}%, stop & target intact)",
        drift_pct=drift_pct,
    )


def evaluate_candidate_signal(
    bar: StockCandidateBar,
    regime: MarketRegime,
    active_positions_count: int,
    cfg: dict,
) -> Optional[TradeSignal]:
    """
    Evaluates all 9 fail-closed entry gates on a candidate symbol from the radar.
    Returns a ranked TradeSignal if all gates pass, else None.
    """
    from apex3.strategy.trail import compute_initial_stop_and_target

    sched = cfg.get("session_schedule", cfg.get("schedule", {}))
    gates = cfg.get("signal_gates", cfg.get("signal", {}))
    risk_cfg = cfg.get("risk_management", cfg.get("risk", {}))

    start_t = sched.get("entry_start_time", sched.get("entry_start", "09:35"))
    cutoff_t = sched.get("entry_cutoff_time", sched.get("entry_cutoff", "12:30"))
    # Gate 0: Session Window [09:35, 12:30] IST
    if not (start_t <= bar.timestamp_hhmm <= cutoff_t):
        return None

    # Active Position Capacity Check
    max_slots = int(risk_cfg.get("max_concurrent_positions", 2))
    if active_positions_count >= max_slots:
        return None

    # Gate 3: Index Regime Alignment
    regime_val = regime.value if hasattr(regime, "value") else str(regime)
    if regime_val == "BULLISH":
        side = "LONG"
    elif regime_val == "BEARISH":
        side = "SHORT"
    else:
        return None

    # Gate 9A: Opening Range Width Cap (1.05%)
    if bar.or_low > 0:
        or_width_pct = (bar.or_high - bar.or_low) / bar.or_low
        if or_width_pct > 0.0105:
            return None

    # Gate 9B: Minimum Volatility Floor (ATR14 / Price >= min_atr_pct)
    min_atr_pct = float(gates.get("min_atr_pct", 0.0))
    if min_atr_pct > 0 and bar.close > 0 and (bar.atr14 / bar.close) < min_atr_pct:
        return None

    # Gate 9C: ADR Exhaustion Filter (Today's range consumed <= 60% of ADR14, with baseline)
    max_adr_mult = float(gates.get("max_adr_consumed_pct", 0.60))
    min_adr_base = float(gates.get("min_adr_baseline", 0.025))
    effective_adr = max(min_adr_base, bar.adr14_pct)
    if effective_adr > 0 and bar.session_low_so_far > 0:
        today_range_pct = (bar.session_high_so_far - bar.session_low_so_far) / bar.session_low_so_far
        if today_range_pct > max_adr_mult * effective_adr:
            return None

    # Tiered Radar & Non-Core Probation Guard
    radar_cfg = cfg.get("radar", {})
    core_wl = radar_cfg.get("core_watchlist", [])
    is_core = (bar.symbol in core_wl)
    cushion_mult = float(gates.get("min_breakout_cushion_atr", 0.15))
    min_clv = float(gates.get("min_clv_ratio", 0.70)) if is_core else float(gates.get("probation_min_clv_ratio", 0.75))
    cushion = cushion_mult * bar.atr14

    rel_strength = bar.stock_return_20 - bar.nifty_return_20
    if side == "LONG":
        bar_range = max(bar.bar_high - bar.bar_low, 0.05) if (bar.bar_high > 0 and bar.bar_low > 0) else 1.0
        clv = (bar.close - bar.bar_low) / bar_range if (bar.bar_high > 0 and bar.bar_low > 0) else 1.0
        if not (bar.close >= (bar.or_high + cushion) and clv >= min_clv and bar.close > bar.session_vwap and rel_strength > 0):
            return None
    else:
        bar_range = max(bar.bar_high - bar.bar_low, 0.05) if (bar.bar_high > 0 and bar.bar_low > 0) else 1.0
        clv = (bar.bar_high - bar.close) / bar_range if (bar.bar_high > 0 and bar.bar_low > 0) else 1.0
        if not (bar.close <= (bar.or_low - cushion) and clv >= min_clv and bar.close < bar.session_vwap and rel_strength < 0):
            return None

    # Gate 5: Institutional Volume Surge (Core >= 2.0x, Non-Core Probation >= 2.5x)
    if bar.avg_volume_20 <= 0:
        return None
    vol_ratio = bar.bar_volume / bar.avg_volume_20
    min_vol = float(gates.get("min_volume_surge_ratio", 2.0)) if is_core else float(gates.get("probation_volume_surge_ratio", 2.5))
    if vol_ratio < min_vol:
        return None

    # Gate 6: Liquidity & Bid-Ask Spread (<= 4 bps)
    max_sprd = float(gates.get("max_spread_bps", 4.0))
    if bar.bid_ask_spread_bps > max_sprd:
        return None

    # Gate 7: Circuit Safety (>= 2.0% away from circuit limit)
    min_circuit_dist = float(gates.get("min_circuit_distance_pct", 0.020))
    if min_circuit_dist > 0.5:
        min_circuit_dist /= 100.0
    if side == "LONG":
        if (bar.upper_circuit - bar.close) / bar.close < min_circuit_dist:
            return None
    else:
        if (bar.close - bar.lower_circuit) / bar.close < min_circuit_dist:
            return None

    # Gate 9: Stop-Width Sweet Spot (0.55% <= R% <= 0.90%)
    levels = compute_initial_stop_and_target(
        side=side,
        entry_price=bar.close,
        swing_low_5=bar.swing_low_5,
        swing_high_5=bar.swing_high_5,
        atr14=bar.atr14,
        cfg=cfg,
        tick_size=getattr(bar, "tick_size", 0.05) or 0.05,
    )
    min_r_pct = float(gates.get("min_stop_distance_pct", 0.0055))
    max_r_pct = float(gates.get("max_stop_distance_pct", 0.0090))
    if not (min_r_pct <= levels.risk_pct <= max_r_pct):
        return None

    # Conviction Score with Core Priority Multiplier (1.35x)
    spread_floor = max(bar.bid_ask_spread_bps, 1.0)
    core_mult = float(radar_cfg.get("core_conviction_multiplier", 1.35)) if is_core else 1.0
    conviction_score = core_mult * ((abs(rel_strength) * 100.0 * vol_ratio) / spread_floor)

    return TradeSignal(
        symbol=bar.symbol,
        side=side,
        entry_price=bar.close,
        levels=levels,
        volume_surge_ratio=round(vol_ratio, 2),
        relative_strength=round(rel_strength, 5),
        conviction_score=round(conviction_score, 4),
    )
