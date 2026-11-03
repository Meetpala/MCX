"""APEX-3 Risk Manager.

Enforces portfolio-level and trade-level risk limits:
1. MAX_SYMBOLS = 3: Never exceed 3 simultaneous strategy positions.
2. RISK_PER_TRADE = 0.25%: Rupee risk of proposed trade <= 0.25% of equity.
3. MAX_COMBINED_OPEN_RISK = 0.75%: Combined stop-distance risk across all open
   positions + proposed trade <= 0.75% of equity.
4. DAILY_LOSS_LIMIT = 1.0%: If realized + unrealized PnL (net of estimated charges)
   is <= -1.0% of equity, block all new entries and initiate risk halt.

WARNING: A risk percentage is not a loss guarantee. Gaps, circuits,
liquidity, API outages, and execution slippage can exceed planned risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from apex3.config import LimitsConfig, RiskConfig
from apex3.models import GateResult, GateStatus, MarginInfo, PositionState


@dataclass(frozen=True)
class PortfolioRiskStatus:
    """Current snapshot of portfolio risk exposure."""
    equity_base: float
    open_positions_count: int
    open_risk_rupees: float
    open_risk_pct: float
    realized_pnl: float
    unrealized_pnl: float
    net_day_pnl: float
    net_day_pnl_pct: float
    is_daily_loss_breached: bool
    is_max_positions_reached: bool
    can_open_new_trade: bool
    gate: GateResult
    reasons: list[str] = field(default_factory=list)


def calculate_position_risk(position: PositionState) -> float:
    """Calculate open stop-distance rupee risk for an existing position.

    Risk = Quantity * abs(Current_Stop - Average_Entry)
    If stop has trailed past entry into profit, risk is 0.0.
    """
    if position.quantity <= 0 or position.protective_stop_price <= 0:
        return 0.0

    if position.side.value == "BUY":
        if position.protective_stop_price >= position.average_entry:
            return 0.0  # Stop has locked profit / break-even
        return position.quantity * (position.average_entry - position.protective_stop_price)
    else:  # SELL
        if position.protective_stop_price <= position.average_entry:
            return 0.0  # Short stop locked profit / break-even
        return position.quantity * (position.protective_stop_price - position.average_entry)


def evaluate_portfolio_risk(
    open_positions: list[PositionState],
    margin_info: MarginInfo,
    proposed_trade_risk_rupees: float,
    limits_config: LimitsConfig,
    risk_config: RiskConfig,
    additional_realized_pnl: float = 0.0,
    estimated_charges: float = 0.0,
) -> PortfolioRiskStatus:
    """Evaluate whether a proposed trade can be taken without breaching any risk limits.

    Args:
        open_positions: List of current strategy-owned positions
        margin_info: Broker margin snapshot
        proposed_trade_risk_rupees: Rupee risk of the candidate trade (0 if only checking status)
        limits_config: LimitsConfig (max_symbols, etc.)
        risk_config: RiskConfig (per-trade, combined, daily loss)
        additional_realized_pnl: Realized P&L from closed trades today
        estimated_charges: Estimated brokerage + taxes incurred today

    Returns:
        PortfolioRiskStatus containing pass/fail gate and current risk metrics.
    """
    equity_base = margin_info.net_value if margin_info.net_value > 0 else margin_info.equity_available
    if equity_base <= 0:
        return PortfolioRiskStatus(
            equity_base=equity_base,
            open_positions_count=len(open_positions),
            open_risk_rupees=0.0,
            open_risk_pct=0.0,
            realized_pnl=additional_realized_pnl,
            unrealized_pnl=0.0,
            net_day_pnl=additional_realized_pnl - estimated_charges,
            net_day_pnl_pct=0.0,
            is_daily_loss_breached=True,
            is_max_positions_reached=True,
            can_open_new_trade=False,
            gate=GateResult(
                name="PORTFOLIO_RISK",
                status=GateStatus.FAIL,
                reason="Equity base is zero or negative; trading blocked",
                data={"equity_base": equity_base},
            ),
            reasons=["ZERO_OR_NEGATIVE_EQUITY"],
        )

    # 1. Open Positions Count
    open_count = len([p for p in open_positions if p.quantity > 0])
    is_max_positions = open_count >= limits_config.max_symbols

    # 2. Open Risk across all active positions
    current_open_risk = sum(calculate_position_risk(p) for p in open_positions if p.quantity > 0)
    total_potential_open_risk = current_open_risk + proposed_trade_risk_rupees
    total_open_risk_pct = (total_potential_open_risk / equity_base) * 100.0

    # 3. Daily PnL (Realized + Unrealized - Estimated Charges)
    total_realized = sum(p.realized_pnl for p in open_positions) + additional_realized_pnl
    total_unrealized = sum(p.unrealized_pnl for p in open_positions if p.quantity > 0)
    net_day_pnl = total_realized + total_unrealized - estimated_charges
    net_day_pnl_pct = (net_day_pnl / equity_base) * 100.0

    # Daily loss limit check (negative value)
    daily_loss_limit_rupees = -(equity_base * (risk_config.daily_loss_limit_pct / 100.0))
    is_daily_loss_breached = net_day_pnl <= daily_loss_limit_rupees

    # 4. Evaluate Gate
    reasons: list[str] = []
    can_open = True

    if is_daily_loss_breached:
        can_open = False
        reasons.append(
            f"Daily loss limit breached: Net P&L ₹{net_day_pnl:,.2f} ({net_day_pnl_pct:.2f}%) <= limit ₹{daily_loss_limit_rupees:,.2f} (-{risk_config.daily_loss_limit_pct}%)"
        )

    if proposed_trade_risk_rupees > 0:
        if is_max_positions:
            can_open = False
            reasons.append(
                f"Max simultaneous positions reached ({open_count}/{limits_config.max_symbols})"
            )

        if total_open_risk_pct > risk_config.max_combined_open_risk_pct:
            can_open = False
            reasons.append(
                f"Combined open risk ({total_open_risk_pct:.2f}%) exceeds max ({risk_config.max_combined_open_risk_pct}%)"
            )

        per_trade_risk_pct = (proposed_trade_risk_rupees / equity_base) * 100.0
        # Allow tiny float leeway (0.01%) for rounding
        if per_trade_risk_pct > (risk_config.risk_per_trade_pct + 0.01):
            can_open = False
            reasons.append(
                f"Per-trade risk ({per_trade_risk_pct:.2f}%) exceeds max ({risk_config.risk_per_trade_pct}%)"
            )

    gate_status = GateStatus.PASS if can_open else GateStatus.FAIL
    gate_reason = "Risk checks passed" if can_open else "; ".join(reasons)

    return PortfolioRiskStatus(
        equity_base=equity_base,
        open_positions_count=open_count,
        open_risk_rupees=current_open_risk,
        open_risk_pct=(current_open_risk / equity_base) * 100.0,
        realized_pnl=total_realized,
        unrealized_pnl=total_unrealized,
        net_day_pnl=net_day_pnl,
        net_day_pnl_pct=net_day_pnl_pct,
        is_daily_loss_breached=is_daily_loss_breached,
        is_max_positions_reached=is_max_positions,
        can_open_new_trade=can_open,
        gate=GateResult(
            name="PORTFOLIO_RISK",
            status=gate_status,
            reason=gate_reason,
            data={
                "equity_base": equity_base,
                "net_pnl": net_day_pnl,
                "open_risk_pct": total_open_risk_pct,
                "positions_count": open_count,
            },
        ),
        reasons=reasons,
    )
