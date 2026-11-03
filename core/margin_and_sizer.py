"""
core/margin_and_sizer.py
------------------------
Zerodha Kite Connect live margin manager, dynamic contract tier router,
risk-budgeted position sizer, and all-in breakeven cost calculator.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.settings import (
    MCX_INSTRUMENTS,
    TIER_DOWNGRADE_ORDER,
    CAPITAL_RISK_PER_TRADE,
    CommodityFamily,
    ContractTier,
    InstrumentSpec,
    MAX_MARGIN_UTILIZATION,
)

logger = logging.getLogger("MCXTrader.MarginSizer")


@dataclass
class SizingResult:
    symbol: str
    instrument_spec: InstrumentSpec
    lots: int
    quantity: int
    risk_per_trade_inr: float
    usable_margin_inr: float
    required_margin_inr: float
    margin_utilization_pct: float
    max_lots_by_risk: int
    max_lots_by_margin: int
    downgraded_from: Optional[str] = None


def compute_all_in_breakeven_offset(
    entry_price: float,
    lot_multiplier: float,
    tick_size: float = 1.0,
) -> float:
    """
    Computes dynamic all-in cost buffer (Delta_cost) per unit:
    Delta_cost = (P_entry * 0.00065) + (Rs. 48 / Lot Multiplier) + 2 ticks

    Accounts for:
    - Brokerage: Rs. 20 * 2 = Rs. 40 round trip
    - GST: 18% on brokerage = Rs. 7.20 (Total brokerage + GST ~ Rs. 47.20 -> Rs. 48)
    - CTT: 0.05% on non-agri futures sell turnover
    - MCX Exchange Turnover Charge: ~0.0026%
    - Stamp Duty (Buy): ~0.002%
    - SEBI Turnover Charges: ~0.0001%
    - 2 Ticks Execution Buffer for adverse slippage

    Returns Delta_cost rounded UP to the nearest tick.
    """
    turnover_tax_pct = 0.00065  # Combined statutory turnover rate
    brokerage_cost = 48.0 / max(lot_multiplier, 1.0)
    turnover_cost = entry_price * turnover_tax_pct
    slippage_buffer = 2.0 * tick_size

    raw_offset = turnover_cost + brokerage_cost + slippage_buffer
    # Round up to nearest tick
    ticks = math.ceil(raw_offset / tick_size)
    return max(ticks * tick_size, 3.0 * tick_size)


def calculate_statutory_charges(
    buy_price: float,
    sell_price: float,
    quantity: int,
    multiplier: float,
) -> Dict[str, float]:
    """
    Calculates exact statutory taxes and brokerage for an MCX futures round-trip.
    """
    buy_turnover = buy_price * multiplier * quantity
    sell_turnover = sell_price * multiplier * quantity
    total_turnover = buy_turnover + sell_turnover

    # Brokerage: Rs. 20 per executed order (buy + sell)
    brokerage = 40.0
    # CTT: 0.05% on sell turnover for commodity futures
    ctt = sell_turnover * 0.0005
    # MCX exchange transaction charge: 0.0026%
    exchange_charges = total_turnover * 0.000026
    # GST: 18% on (brokerage + exchange charges)
    gst = (brokerage + exchange_charges) * 0.18
    # Stamp duty: 0.002% on buy turnover
    stamp_duty = buy_turnover * 0.00002
    # SEBI turnover fee: Rs. 10 per crore (0.0001%)
    sebi_charges = total_turnover * 0.000001

    total_charges = brokerage + ctt + exchange_charges + gst + stamp_duty + sebi_charges

    return {
        "brokerage": brokerage,
        "ctt": ctt,
        "exchange_charges": exchange_charges,
        "gst": gst,
        "stamp_duty": stamp_duty,
        "sebi_charges": sebi_charges,
        "total_charges": total_charges,
    }


def calculate_net_pnl(
    entry_price: float,
    exit_price: float,
    quantity: int,
    multiplier: float,
    is_long: bool = True,
) -> Tuple[float, float, float]:
    """
    Calculates (gross_pnl, total_charges, net_pnl).
    """
    if is_long:
        buy_price = entry_price
        sell_price = exit_price
        points = exit_price - entry_price
    else:
        buy_price = exit_price
        sell_price = entry_price
        points = entry_price - exit_price

    gross_pnl = points * multiplier * quantity
    charges = calculate_statutory_charges(buy_price, sell_price, quantity, multiplier)
    net_pnl = gross_pnl - charges["total_charges"]
    return gross_pnl, charges["total_charges"], net_pnl


class LiveMarginManager:
    """
    Queries Zerodha Kite Connect margins and sizes orders based on capital & margin gates.
    """

    def __init__(self, kite_client: Optional[Any] = None, default_equity: float = 250000.0):
        self.kite = kite_client
        self.default_equity = default_equity

    def get_account_margins(self) -> Dict[str, float]:
        """
        Retrieves live margins from Zerodha.
        Supports both dedicated Commodity Segment and Single Ledger.
        Returns dict with keys: 'total_equity', 'usable_margin', 'raw_commodity', 'raw_equity'.
        """
        if self.kite is None:
            # Paper trading / simulation fallback
            usable = self.default_equity * MAX_MARGIN_UTILIZATION
            return {
                "total_equity": self.default_equity,
                "usable_margin": usable,
                "commodity_live": usable,
                "equity_net": self.default_equity,
            }

        try:
            margins = self.kite.margins()
            commodity = margins.get("commodity", {})
            equity = margins.get("equity", {})

            # Usable margin from commodity segment
            comm_live = float(commodity.get("available", {}).get("live_balance", 0.0))
            comm_net = float(commodity.get("net", 0.0))
            eq_live = float(equity.get("available", {}).get("live_balance", 0.0))
            eq_net = float(equity.get("net", 0.0))

            # Commodity segment balance preferred; fallback to equity net if single-ledger
            best_comm_balance = max(comm_live, comm_net)
            best_eq_balance = max(eq_live, eq_net)

            if best_comm_balance > 0:
                base_free_margin = best_comm_balance
                total_equity = comm_net if comm_net > 0 else comm_live
            else:
                base_free_margin = best_eq_balance
                total_equity = eq_net if eq_net > 0 else eq_live

            if total_equity <= 0:
                total_equity = self.default_equity
                base_free_margin = self.default_equity

            usable_margin = MAX_MARGIN_UTILIZATION * base_free_margin

            return {
                "total_equity": total_equity,
                "usable_margin": usable_margin,
                "commodity_live": comm_live,
                "equity_net": eq_net,
            }
        except Exception as e:
            logger.error(f"Error fetching Kite margins: {e}. Falling back to default equity {self.default_equity}")
            usable = self.default_equity * MAX_MARGIN_UTILIZATION
            return {
                "total_equity": self.default_equity,
                "usable_margin": usable,
                "commodity_live": usable,
                "equity_net": self.default_equity,
            }

    def get_single_lot_required_margin(
        self,
        symbol: str,
        current_price: float,
        is_buy: bool = True,
    ) -> float:
        """
        Queries kite.order_margins() to get exact SPAN + Exposure required margin for 1 lot.
        Falls back to instrument matrix percentage if Kite API is unavailable.
        """
        spec = MCX_INSTRUMENTS.get(symbol)
        if spec is None:
            raise ValueError(f"Unknown MCX instrument symbol: {symbol}")

        if self.kite is not None:
            try:
                order_params = [
                    {
                        "exchange": spec.exchange,
                        "tradingsymbol": symbol,
                        "transaction_type": "BUY" if is_buy else "SELL",
                        "variety": "regular",
                        "product": "NRML",
                        "order_type": "LIMIT",
                        "quantity": spec.lot_size,
                        "price": current_price,
                    }
                ]
                resp = self.kite.order_margins(order_params)
                if isinstance(resp, list) and len(resp) > 0:
                    tot = resp[0].get("total", 0.0)
                    if tot > 0:
                        return float(tot)
            except Exception as e:
                logger.warning(f"order_margins call failed for {symbol}: {e}. Using spec fallback.")

        # Spec-based approximation fallback
        contract_value = current_price * spec.multiplier * spec.lot_size
        return contract_value * spec.approx_nrml_margin

    def size_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        is_buy: bool = True,
        override_equity: Optional[float] = None,
        allow_downgrade: bool = True,
    ) -> SizingResult:
        """
        Calculates position size using:
        - Max lots by 1% Capital Risk
        - Max lots by 65% Usable Margin
        - Auto-downgrades from Standard -> Mini -> Micro if lots == 0
        """
        margin_info = self.get_account_margins()
        account_equity = override_equity if override_equity is not None else margin_info["total_equity"]
        usable_margin = margin_info["usable_margin"]

        risk_distance = abs(entry_price - stop_loss_price)
        if risk_distance <= 0:
            raise ValueError(f"Stop loss distance must be > 0. Entry: {entry_price}, SL: {stop_loss_price}")

        initial_spec = MCX_INSTRUMENTS.get(symbol)
        if initial_spec is None:
            raise ValueError(f"Unsupported instrument: {symbol}")

        family = initial_spec.family
        downgrade_chain = TIER_DOWNGRADE_ORDER.get(family, [symbol])
        # Find start index in downgrade chain
        start_idx = downgrade_chain.index(symbol) if symbol in downgrade_chain else 0
        search_symbols = downgrade_chain[start_idx:] if allow_downgrade else [symbol]

        chosen_result: Optional[SizingResult] = None
        downgraded_from: Optional[str] = None

        for sym in search_symbols:
            spec = MCX_INSTRUMENTS[sym]
            risk_per_unit = risk_distance * spec.multiplier
            max_risk_amount = account_equity * CAPITAL_RISK_PER_TRADE

            max_lots_by_risk = math.floor(max_risk_amount / risk_per_unit) if risk_per_unit > 0 else 0
            single_lot_margin = self.get_single_lot_required_margin(sym, entry_price, is_buy)
            max_lots_by_margin = math.floor(usable_margin / single_lot_margin) if single_lot_margin > 0 else 0

            final_lots = min(max_lots_by_risk, max_lots_by_margin)

            if final_lots > 0:
                total_qty = final_lots * spec.lot_size
                req_margin = final_lots * single_lot_margin
                total_risk_inr = final_lots * risk_per_unit
                util_pct = (req_margin / usable_margin * 100.0) if usable_margin > 0 else 0.0

                chosen_result = SizingResult(
                    symbol=sym,
                    instrument_spec=spec,
                    lots=final_lots,
                    quantity=total_qty,
                    risk_per_trade_inr=total_risk_inr,
                    usable_margin_inr=usable_margin,
                    required_margin_inr=req_margin,
                    margin_utilization_pct=util_pct,
                    max_lots_by_risk=max_lots_by_risk,
                    max_lots_by_margin=max_lots_by_margin,
                    downgraded_from=downgraded_from,
                )
                break
            else:
                downgraded_from = symbol
                logger.info(
                    f"Contract {sym} yielded 0 lots (Risk lots={max_lots_by_risk}, "
                    f"Margin lots={max_lots_by_margin}, Req margin={single_lot_margin:.1f}, "
                    f"Usable={usable_margin:.1f}). Attempting micro downgrade..."
                )

        if chosen_result is None:
            # Sizing failed even at micro level
            last_spec = MCX_INSTRUMENTS[search_symbols[-1]]
            single_lot_margin = self.get_single_lot_required_margin(last_spec.symbol, entry_price, is_buy)
            risk_per_unit = risk_distance * last_spec.multiplier
            max_risk_amount = account_equity * CAPITAL_RISK_PER_TRADE
            max_lots_by_risk = math.floor(max_risk_amount / risk_per_unit) if risk_per_unit > 0 else 0
            max_lots_by_margin = math.floor(usable_margin / single_lot_margin) if single_lot_margin > 0 else 0

            return SizingResult(
                symbol=last_spec.symbol,
                instrument_spec=last_spec,
                lots=0,
                quantity=0,
                risk_per_trade_inr=0.0,
                usable_margin_inr=usable_margin,
                required_margin_inr=single_lot_margin,
                margin_utilization_pct=0.0,
                max_lots_by_risk=max_lots_by_risk,
                max_lots_by_margin=max_lots_by_margin,
                downgraded_from=downgraded_from,
            )

        return chosen_result

