"""APEX-3 Position Sizing and Equal Allocation Module.

Calculates share quantity based on:
1. Usable equity margin after safety reserve (M)
2. Live leverage ceiling (L = min(MAX_LEVERAGE, order_leverage))
3. Equal notional allocation across ONLY qualified candidates (N)
   B = M * L / N
4. Per-trade rupee risk budget (RISK = equity * risk_per_trade_pct)
   Q_risk = floor(RISK / (abs(E - S) + C))
5. Broker-reported exact order margin limits:
   Q = min(Q_notional, Q_risk, broker_margin_quantity_limit)

CRITICAL RULES:
- 4x is a ceiling, never assumed.
- If live leverage < min_accepted_leverage, symbol is skipped.
- Never force investment of all capital.
- Do not double-count utilised margin or treat collateral as unrestricted cash.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from apex3.config import LimitsConfig, RiskConfig
from apex3.models import (
    GateResult,
    GateStatus,
    InstrumentInfo,
    MarginInfo,
    OrderMarginEstimate,
    Side,
    get_mcx_lot_multiplier,
)


@dataclass(frozen=True)
class SizingResult:
    """Result of position sizing calculation for a symbol."""
    symbol: str
    quantity: int
    is_valid: bool
    side: Side
    entry_price: float
    stop_price: float
    per_share_risk: float
    notional_value: float
    trade_risk_rupees: float
    usable_margin: float
    per_symbol_budget: float
    effective_leverage: float
    q_notional: int
    q_risk: int
    q_margin: int
    gate: GateResult
    rejection_code: str = ""
    rejection_reason: str = ""


def compute_usable_margin(margin_info: MarginInfo, safety_reserve_pct: float = 5.0) -> float:
    """Compute usable equity margin after subtracting the safety reserve.

    Uses broker-reported equity_available directly without double-counting.
    Collateral is NOT treated as unrestricted cash.

    Args:
        margin_info: Broker-reported MarginInfo.
        safety_reserve_pct: Percentage buffer to hold back (e.g. 5%).

    Returns:
        Usable margin in Rupees (>= 0.0).
    """
    available = max(0.0, margin_info.equity_available)
    reserve = available * (safety_reserve_pct / 100.0)
    return max(0.0, available - reserve)


def calculate_mcx_intraday_charges(
    buy_price: float,
    sell_price: float,
    lots: int,
    lot_multiplier: int = 1,
) -> float:
    """Exact MCX + Zerodha commodity futures statutory fee calculator.

    - Brokerage: 0.03% or Rs. 20 per order (whichever is lower)
    - CTT: 0.01% on sell turnover only
    - MCX Exchange Turnover: 0.0021% on total turnover
    - SEBI Fee: 0.0001% (Rs. 10/Crore) on total turnover
    - Stamp Duty: 0.002% on buy turnover
    - GST: 18% on (Brokerage + Exchange Turnover + SEBI Fee)
    """
    if lots <= 0 or buy_price <= 0 or sell_price <= 0:
        return 0.0

    buy_turnover = buy_price * lots * lot_multiplier
    sell_turnover = sell_price * lots * lot_multiplier
    total_turnover = buy_turnover + sell_turnover

    brok_buy = min(20.0, buy_turnover * 0.0003)
    brok_sell = min(20.0, sell_turnover * 0.0003)
    brokerage = brok_buy + brok_sell

    ctt = 0.0001 * sell_turnover
    exchange_fee = 0.000021 * total_turnover
    sebi_fee = 0.000001 * total_turnover
    stamp_duty = 0.00002 * buy_turnover
    gst = 0.18 * (brokerage + exchange_fee + sebi_fee)

    return round(brokerage + ctt + exchange_fee + sebi_fee + stamp_duty + gst, 2)


def calculate_zerodha_intraday_charges(
    buy_price: float,
    sell_price: float,
    quantity: int,
    lot_multiplier: int = 1,
    exchange: str = "NSE",
) -> float:
    """Exact NSE / MCX Zerodha intraday statutory fee calculator.

    If exchange is MCX or lot_multiplier > 1, computes MCX commodity charges.
    Otherwise, computes standard NSE intraday equity MIS charges.
    """
    if quantity <= 0 or buy_price <= 0 or sell_price <= 0:
        return 0.0

    if exchange.upper() == "MCX" or lot_multiplier > 1:
        return calculate_mcx_intraday_charges(buy_price, sell_price, quantity, lot_multiplier)

    buy_turnover = buy_price * quantity
    sell_turnover = sell_price * quantity
    total_turnover = buy_turnover + sell_turnover

    brok_buy = min(20.0, buy_turnover * 0.0003)
    brok_sell = min(20.0, sell_turnover * 0.0003)
    brokerage = brok_buy + brok_sell

    stt = sell_turnover * 0.00025
    nse_charges = total_turnover * 0.0000318
    sebi_charges = total_turnover * 0.000001
    stamp_duty = buy_turnover * 0.00003
    gst = 0.18 * (brokerage + nse_charges + sebi_charges)

    return round(brokerage + stt + nse_charges + sebi_charges + stamp_duty + gst, 2)


def size_position(
    symbol: str,
    side: Side,
    entry_price: float,
    stop_price: float,
    num_qualified_symbols: int,
    margin_info: MarginInfo,
    instrument: InstrumentInfo,
    limits_config: LimitsConfig,
    risk_config: RiskConfig,
    order_margin_estimate: Optional[OrderMarginEstimate] = None,
) -> SizingResult:
    """Compute exact risk-sized and notional-budgeted quantity for a qualified symbol.

    Args:
        symbol: Candidate symbol
        side: BUY or SELL
        entry_price: Expected entry fill price (E)
        stop_price: Initial structural stop price (S)
        num_qualified_symbols: Total count of symbols passing all signal gates (N <= 3)
        margin_info: Fresh broker equity margin
        instrument: Instrument details (tick size, lot size)
        limits_config: Limits configuration (leverage, safety reserve)
        risk_config: Risk configuration (risk per trade, slippage allowance)
        order_margin_estimate: Optional live margin estimate from Kite for proposed order

    Returns:
        SizingResult with final quantity and all intermediate calculations.
    """
    # 0. Basic Validation
    if entry_price <= 0 or stop_price <= 0:
        return SizingResult(
            symbol=symbol,
            quantity=0,
            is_valid=False,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            per_share_risk=0.0,
            notional_value=0.0,
            trade_risk_rupees=0.0,
            usable_margin=0.0,
            per_symbol_budget=0.0,
            effective_leverage=0.0,
            q_notional=0,
            q_risk=0,
            q_margin=0,
            gate=GateResult(
                name="SIZING_VALIDATION",
                status=GateStatus.FAIL,
                reason="Invalid entry or stop price (must be > 0)",
            ),
            rejection_code="INVALID_PRICES",
            rejection_reason="Entry or stop price <= 0",
        )

    # Stop direction validation
    if side == Side.BUY and stop_price >= entry_price:
        return SizingResult(
            symbol=symbol,
            quantity=0,
            is_valid=False,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            per_share_risk=0.0,
            notional_value=0.0,
            trade_risk_rupees=0.0,
            usable_margin=0.0,
            per_symbol_budget=0.0,
            effective_leverage=0.0,
            q_notional=0,
            q_risk=0,
            q_margin=0,
            gate=GateResult(
                name="SIZING_VALIDATION",
                status=GateStatus.FAIL,
                reason=f"Long stop ({stop_price:.2f}) must be strictly below entry ({entry_price:.2f})",
            ),
            rejection_code="INVALID_STOP_DIRECTION",
            rejection_reason="Long stop >= entry",
        )
    elif side == Side.SELL and stop_price <= entry_price:
        return SizingResult(
            symbol=symbol,
            quantity=0,
            is_valid=False,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            per_share_risk=0.0,
            notional_value=0.0,
            trade_risk_rupees=0.0,
            usable_margin=0.0,
            per_symbol_budget=0.0,
            effective_leverage=0.0,
            q_notional=0,
            q_risk=0,
            q_margin=0,
            gate=GateResult(
                name="SIZING_VALIDATION",
                status=GateStatus.FAIL,
                reason=f"Short stop ({stop_price:.2f}) must be strictly above entry ({entry_price:.2f})",
            ),
            rejection_code="INVALID_STOP_DIRECTION",
            rejection_reason="Short stop <= entry",
        )

    # 1. Usable Margin M
    usable_margin = compute_usable_margin(margin_info, limits_config.safety_reserve_pct)
    if usable_margin <= 0:
        return SizingResult(
            symbol=symbol,
            quantity=0,
            is_valid=False,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            per_share_risk=0.0,
            notional_value=0.0,
            trade_risk_rupees=0.0,
            usable_margin=usable_margin,
            per_symbol_budget=0.0,
            effective_leverage=0.0,
            q_notional=0,
            q_risk=0,
            q_margin=0,
            gate=GateResult(
                name="MARGIN_AVAILABILITY",
                status=GateStatus.FAIL,
                reason=f"Usable margin is zero or negative ({usable_margin:.2f})",
                data={"available": margin_info.equity_available},
            ),
            rejection_code="INSUFFICIENT_MARGIN",
            rejection_reason="Usable equity margin is zero or negative",
        )

    # 2. Leverage Determination L
    # Broker leverage from live order margin estimate if available
    broker_leverage = limits_config.max_leverage
    if order_margin_estimate is not None and order_margin_estimate.leverage_available > 0:
        broker_leverage = order_margin_estimate.leverage_available

    # Check minimum accepted leverage
    if (broker_leverage + 1e-3) < limits_config.min_accepted_leverage:
        return SizingResult(
            symbol=symbol,
            quantity=0,
            is_valid=False,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            per_share_risk=0.0,
            notional_value=0.0,
            trade_risk_rupees=0.0,
            usable_margin=usable_margin,
            per_symbol_budget=0.0,
            effective_leverage=broker_leverage,
            q_notional=0,
            q_risk=0,
            q_margin=0,
            gate=GateResult(
                name="LEVERAGE_CHECK",
                status=GateStatus.FAIL,
                reason=f"Live leverage {broker_leverage:.2f}x < configured minimum {limits_config.min_accepted_leverage:.2f}x",
                data={"broker_leverage": broker_leverage, "min_accepted": limits_config.min_accepted_leverage},
            ),
            rejection_code="LEVERAGE_BELOW_MINIMUM",
            rejection_reason="Broker leverage below minimum accepted leverage",
        )

    effective_leverage = min(limits_config.max_leverage, broker_leverage)

    # 3. Equal Notional Allocation across max_symbols slots
    lot_multiplier = getattr(instrument, "lot_multiplier", 1) or get_mcx_lot_multiplier(symbol)
    order_lot_size = getattr(instrument, "order_lot_size", 1)
    lot_size = max(1, order_lot_size)

    n = max(1, limits_config.max_symbols)
    per_symbol_budget = (usable_margin * effective_leverage) / n
    notional_per_lot = entry_price * lot_multiplier

    # Q_notional = floor(B / (E * multiplier))
    q_notional = math.floor(per_symbol_budget / notional_per_lot) if notional_per_lot > 0 else 0

    # 4. Risk per lot and Q_risk
    # C = per-lot allowance for charges and slippage
    slippage_cost = entry_price * (risk_config.slippage_allowance_pct / 100.0) * lot_multiplier
    per_point_loss = abs(entry_price - stop_price)
    gross_risk_per_lot = per_point_loss * lot_multiplier
    per_lot_risk = gross_risk_per_lot + slippage_cost

    # RISK = Rupee risk budget for this trade
    equity_base = margin_info.net_value if margin_info.net_value > 0 else margin_info.equity_available
    trade_risk_rupees = equity_base * (risk_config.risk_per_trade_pct / 100.0)

    q_risk = math.floor(trade_risk_rupees / per_lot_risk) if per_lot_risk > 0 else 0

    # 5. Margin Quantity Limit
    q_margin = q_notional
    if order_margin_estimate is not None and order_margin_estimate.required_margin > 0:
        margin_per_lot = notional_per_lot / effective_leverage
        if margin_per_lot > 0:
            q_margin = math.floor(usable_margin / margin_per_lot)

    # Final Quantity: Smallest of Q_notional, Q_risk, Q_margin
    q_raw = min(q_notional, q_risk, q_margin)
    q_final = (q_raw // lot_size) * lot_size

    # Hard Risk Bound: Ensure total realized loss at stop-loss (including statutory friction) <= trade_risk_rupees
    is_mcx = symbol.upper().startswith(("SILVER", "GOLD")) or lot_multiplier > 1
    fee_calc = calculate_mcx_intraday_charges if is_mcx else (lambda bp, sp, q, lm: calculate_zerodha_intraday_charges(bp, sp, q, lm, exchange="NSE"))

    if q_final > 0 and getattr(risk_config, "bound_loss_with_charges", True):
        while q_final > 0:
            stop_exit_charges = fee_calc(entry_price, stop_price, q_final, lot_multiplier)
            total_realized_loss = (q_final * per_lot_risk) + stop_exit_charges
            if total_realized_loss <= trade_risk_rupees:
                break
            q_final -= lot_size

    if q_final <= 0:
        return SizingResult(
            symbol=symbol,
            quantity=0,
            is_valid=False,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            per_share_risk=per_lot_risk,
            notional_value=0.0,
            trade_risk_rupees=trade_risk_rupees,
            usable_margin=usable_margin,
            per_symbol_budget=per_symbol_budget,
            effective_leverage=effective_leverage,
            q_notional=q_notional,
            q_risk=q_risk,
            q_margin=q_margin,
            gate=GateResult(
                name="SIZING_QUANTITY",
                status=GateStatus.FAIL,
                reason=f"Risk constraints (stop-loss + statutory friction) exceed risk cap Rs. {trade_risk_rupees:.2f} even at minimum lot {lot_size}",
                data={"q_notional": q_notional, "q_risk": q_risk, "q_margin": q_margin},
            ),
            rejection_code="EXCEEDED_NET_RISK_CAP",
            rejection_reason=f"Total stop-loss risk with statutory charges exceeds Rs. {trade_risk_rupees:.2f}",
        )

    # Gate: Reject trade if charges consume > 18% of expected 2.0R gross profit
    target_price = round(entry_price + (2.0 * per_point_loss) if side == Side.BUY else entry_price - (2.0 * per_point_loss), 2)
    est_charges = fee_calc(entry_price, target_price, q_final, lot_multiplier)
    expected_2r_gross_profit = 2.0 * gross_risk_per_lot * q_final

    max_fee_ratio = getattr(risk_config, "max_fee_to_profit_ratio", 0.18)
    if expected_2r_gross_profit <= 0 or (est_charges / expected_2r_gross_profit) > max_fee_ratio:
        fee_pct = (est_charges / expected_2r_gross_profit * 100.0) if expected_2r_gross_profit > 0 else 100.0
        return SizingResult(
            symbol=symbol,
            quantity=0,
            is_valid=False,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            per_share_risk=per_lot_risk,
            notional_value=0.0,
            trade_risk_rupees=trade_risk_rupees,
            usable_margin=usable_margin,
            per_symbol_budget=per_symbol_budget,
            effective_leverage=effective_leverage,
            q_notional=q_notional,
            q_risk=q_risk,
            q_margin=q_margin,
            gate=GateResult(
                name="FEE_TO_PROFIT_RATIO",
                status=GateStatus.FAIL,
                reason=f"Estimated charges (Rs. {est_charges:.2f}) exceed {max_fee_ratio*100:.0f}% of expected 2.0R gross profit (Rs. {expected_2r_gross_profit:.2f}, {fee_pct:.1f}%)",
                data={"charges": est_charges, "expected_profit": expected_2r_gross_profit, "ratio": est_charges / expected_2r_gross_profit if expected_2r_gross_profit > 0 else 1.0},
            ),
            rejection_code="EXCEEDED_FEE_TO_PROFIT_LIMIT",
            rejection_reason=f"Charges consume > {max_fee_ratio*100:.0f}% of expected 2.0R gross profit ({fee_pct:.1f}%)",
        )

    notional_value = q_final * entry_price * lot_multiplier
    stop_exit_charges = calculate_mcx_intraday_charges(entry_price, stop_price, q_final, lot_multiplier)
    actual_trade_risk = (q_final * per_lot_risk) + stop_exit_charges

    return SizingResult(
        symbol=symbol,
        quantity=q_final,
        is_valid=True,
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        per_share_risk=per_lot_risk,
        notional_value=notional_value,
        trade_risk_rupees=actual_trade_risk,
        usable_margin=usable_margin,
        per_symbol_budget=per_symbol_budget,
        effective_leverage=effective_leverage,
        q_notional=q_notional,
        q_risk=q_risk,
        q_margin=q_margin,
        gate=GateResult(
            name="SIZING_QUANTITY",
            status=GateStatus.PASS,
            reason=f"Sized {q_final} lots (Notional: Rs. {notional_value:,.2f}, Realized Stop Risk: Rs. {actual_trade_risk:,.2f})",
            data={"quantity": q_final, "notional": notional_value, "risk": actual_trade_risk},
        ),
    )


# ---------------------------------------------------------------------------
# High-Profit 2-Slot Sizing & Exact Zerodha Charge Calculation
# ---------------------------------------------------------------------------

@dataclass
class PositionSizeResult:
    quantity: int
    notional_exposure: float
    margin_required: float
    planned_risk_inr: float
    estimated_roundtrip_charges_inr: float


def fetch_zerodha_live_capital(kite_client, fallback_capital: float) -> float:
    """
    Automatically fetches available live cash margin from Zerodha Kite Connect.
    Falls back to config starting_capital during backtests.
    """
    if kite_client is None:
        return fallback_capital
    try:
        margins = kite_client.margins(segment="equity")
        live_balance = float(margins["available"]["live_balance"])
        return live_balance if live_balance > 0 else fallback_capital
    except Exception:
        return fallback_capital



def calculate_position_size(
    entry_price: float,
    risk_per_share: float,
    target_price: float,
    cfg: dict,
    kite_client=None,
) -> Optional[PositionSizeResult]:
    """
    Computes controlled quantity Q = min(floor(Risk_Capital / R), floor(Slot_Budget / Entry_Price))
    and verifies that expected 2.0R gross profit is at least 5x total Zerodha charges.
    """
    if entry_price <= 0 or risk_per_share <= 0:
        return None

    acc_cfg = cfg.get("account", {})
    risk_cfg = cfg.get("risk_management", cfg.get("risk", {}))

    base_capital = fetch_zerodha_live_capital(
        kite_client if acc_cfg.get("fetch_live_zerodha_margin", False) else None,
        float(acc_cfg.get("starting_capital", 15000.0)),
    )

    safety_buffer = float(acc_cfg.get("safety_reserve_buffer_pct", 0.05))
    if safety_buffer > 0.5:
        safety_buffer /= 100.0
    leverage = float(acc_cfg.get("max_intraday_leverage", 4.0))
    max_slots = int(risk_cfg.get("max_concurrent_positions", 2))

    usable_capital = base_capital * (1.0 - safety_buffer)
    total_buying_power = usable_capital * leverage
    slot_budget = total_buying_power / max(1, max_slots)

    risk_pct = float(risk_cfg.get("risk_per_trade_pct", 0.0080))
    if risk_pct > 0.1:
        risk_pct /= 100.0
    risk_capital = base_capital * risk_pct

    qty_by_risk = math.floor(risk_capital / risk_per_share)
    qty_by_budget = math.floor(slot_budget / entry_price)
    quantity = min(qty_by_risk, qty_by_budget)

    if quantity < 1:
        return None

    notional = round(quantity * entry_price, 2)
    margin_req = round(notional / leverage, 2)
    planned_risk = round(quantity * risk_per_share, 2)

    est_charges = calculate_zerodha_intraday_charges(entry_price, target_price, quantity, exchange="NSE")
    expected_gross_win = abs(target_price - entry_price) * quantity

    # Reject trade if Zerodha charges consume > 18% of expected 2.0R gross profit
    if expected_gross_win <= 0 or (est_charges / expected_gross_win) > 0.18:
        return None

    return PositionSizeResult(
        quantity=quantity,
        notional_exposure=notional,
        margin_required=margin_req,
        planned_risk_inr=planned_risk,
        estimated_roundtrip_charges_inr=est_charges,
    )
