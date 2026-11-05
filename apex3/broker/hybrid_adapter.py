"""Hybrid Broker Adapter for APEX-3.

SPLIT-FEED ARCHITECTURE:
- Market Data Layer: Angel One SmartAPI (100% Free Forever live quotes, candles, instruments).
- Execution Layer: Zerodha Kite Connect Personal Free (100% Free Forever orders, margins, positions).

Strict Security & Separation:
All order placements, modifications, cancellations, and margin checks are routed EXCLUSIVELY
to Zerodha. Angel One is used solely as a read-only market data stream.
"""
from __future__ import annotations

import inspect
import logging
from datetime import date, datetime
from typing import Optional

from apex3.broker.angel_data_provider import AngelDataProvider
from apex3.broker.interface import BrokerInterface
from apex3.models import (
    Candle,
    InstrumentInfo,
    MarginInfo,
    OrderMarginEstimate,
    OrderRequest,
    OrderState,
    PositionState,
    Quote,
)

logger = logging.getLogger(__name__)


class HybridBrokerAdapter(BrokerInterface):
    """Hybrid broker adapter combining Angel One data feed with Zerodha order execution."""

    def __init__(
        self,
        data_provider: AngelDataProvider,
        execution_broker: BrokerInterface,
    ) -> None:
        self.data_provider = data_provider
        self.execution_broker = execution_broker
        logger.info(
            "Initialized APEX-3 HYBRID BROKER ADAPTER: [Data: Angel One SmartAPI (Free)] -> [Execution: Zerodha Personal (Free)]"
        )

    # ── Market Data (Delegated to Angel One SmartAPI) ────────────────

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Quote:
        """Fetch live quote from Angel One free data feed."""
        q = self.data_provider.get_quote(symbol=symbol, exchange=exchange)
        from apex3.broker.paper import PaperBroker
        if isinstance(self.execution_broker, PaperBroker) and q is not None:
            if q.last_price > 0:
                self.execution_broker.simulate_tick(symbol, q.last_price)
            self.execution_broker.set_quote(q)
        return q

    def get_quotes_batch(self, symbols: list[str], exchange: str = "NSE") -> dict[str, Quote]:
        """Fetch batch quotes from Angel One free data feed in a single snapshot."""
        if hasattr(self.data_provider, "get_quotes_batch"):
            batch = self.data_provider.get_quotes_batch(symbols, exchange=exchange)
        else:
            batch = {sym: self.data_provider.get_quote(sym, exchange=exchange) for sym in symbols}
        from apex3.broker.paper import PaperBroker
        if isinstance(self.execution_broker, PaperBroker) and isinstance(batch, dict):
            for sym, q in batch.items():
                if q is not None:
                    if q.last_price > 0:
                        self.execution_broker.simulate_tick(sym, q.last_price)
                    self.execution_broker.set_quote(q)
        return batch

    def get_historical(
        self,
        symbol: str,
        interval: str,
        from_dt: datetime,
        to_dt: datetime,
        exchange: str = "NSE",
    ) -> list[Candle]:
        """Fetch historical candles from Angel One free data feed."""
        return self.data_provider.get_historical(
            symbol=symbol,
            interval=interval,
            from_dt=from_dt,
            to_dt=to_dt,
            exchange=exchange,
        )

    def get_instrument_info(self, symbol: str, exchange: Optional[str] = None) -> InstrumentInfo:
        """Fetch instrument info, using execution broker for execution metadata and data provider for streaming/circuits."""
        if exchange is None:
            exchange = "MCX" if any(symbol.upper().startswith(r) for r in ("SILVER", "GOLD", "CRUDE", "COPPER", "ZINC", "NATURALGAS")) else "NSE"

        # If symbol is a bare commodity root (e.g. SILVERMIC), attempt resolving to active dated contract
        if exchange == "MCX" and symbol.upper() in ("SILVERMIC", "GOLDGUINEA", "GOLDTEN", "SILVERM", "GOLDM", "SILVER", "GOLD"):
            try:
                resolved_map = self.resolve_active_mcx_futures([symbol.upper()])
                if symbol.upper() in resolved_map:
                    symbol = resolved_map[symbol.upper()]
            except Exception as re:
                logger.debug("Could not resolve root %s to dated contract: %s", symbol, re)

        # 1. Query execution broker for authoritative Zerodha execution parameters
        exec_info: Optional[InstrumentInfo] = None
        if hasattr(self.execution_broker, "get_instrument_info"):
            try:
                exec_info = self.execution_broker.get_instrument_info(symbol=symbol, exchange=exchange)
            except Exception as e:
                logger.warning("Could not fetch execution broker instrument info for %s: %s", symbol, e)

        # 2. Query data provider for Angel streaming token and cached circuits
        data_info = self.data_provider.get_instrument_info(symbol=symbol, exchange=exchange)
        angel_token = getattr(data_info, "exchange_token", "") or getattr(data_info, "instrument_token", "")
        if not angel_token and hasattr(self.data_provider, "resolve_token"):
            try:
                angel_token = self.data_provider.resolve_token(symbol, exchange) or ""
            except Exception:
                pass

        if exec_info is not None:
            up_c = data_info.upper_circuit if data_info.upper_circuit > 0 else exec_info.upper_circuit
            low_c = data_info.lower_circuit if data_info.lower_circuit > 0 else exec_info.lower_circuit
            t_size = exec_info.tick_size if exec_info.tick_size > 0 else data_info.tick_size
            if (up_c <= 0 or low_c <= 0) and t_size > 0:
                cached_q = getattr(self.data_provider, "_quote_cache", {}).get(symbol)
                ref_px = 0.0
                if cached_q:
                    ref_px = cached_q.close if cached_q.close > 0 else cached_q.last_price
                if ref_px > 0:
                    up_c = round(round(ref_px * 1.10 / t_size) * t_size, 2) if up_c <= 0 else up_c
                    low_c = round(round(ref_px * 0.90 / t_size) * t_size, 2) if low_c <= 0 else low_c

            order_lot = exec_info.order_lot_size if exec_info.order_lot_size > 0 else getattr(data_info, "order_lot_size", 1)
            lot_mult = exec_info.lot_multiplier if exec_info.lot_multiplier > 0 else getattr(data_info, "lot_multiplier", 1)
            exp_d = exec_info.expiry_date or getattr(data_info, "expiry_date", None)
            dte = exec_info.days_to_expiry if exec_info.days_to_expiry < 999 else getattr(data_info, "days_to_expiry", 999)

            return InstrumentInfo(
                symbol=exec_info.symbol,
                exchange=exec_info.exchange,
                tick_size=t_size,
                lot_size=exec_info.lot_size if exec_info.lot_size > 0 else data_info.lot_size,
                order_lot_size=order_lot,
                lot_multiplier=lot_mult,
                expiry_date=exp_d,
                days_to_expiry=dte,
                exchange_token=angel_token,
                instrument_token=exec_info.instrument_token or angel_token,
                upper_circuit=up_c,
                lower_circuit=low_c,
                mis_available=exec_info.mis_available,
            )

        # If execution broker info is missing, fail safe by marking mis_available as False
        logger.warning("Execution broker instrument info unavailable for %s; refusing to assume MIS eligibility", symbol)
        return InstrumentInfo(
            symbol=data_info.symbol,
            exchange=data_info.exchange,
            tick_size=data_info.tick_size,
            lot_size=data_info.lot_size,
            order_lot_size=getattr(data_info, "order_lot_size", 1),
            lot_multiplier=getattr(data_info, "lot_multiplier", 1),
            expiry_date=getattr(data_info, "expiry_date", None),
            days_to_expiry=getattr(data_info, "days_to_expiry", 999),
            exchange_token=getattr(data_info, "instrument_token", ""),
            instrument_token=getattr(data_info, "instrument_token", ""),
            upper_circuit=data_info.upper_circuit,
            lower_circuit=data_info.lower_circuit,
            mis_available=False,
        )

    def resolve_active_mcx_futures(
        self,
        symbol_roots: list[str],
        min_days_to_expiry: int = 5,
        as_of: Optional[date] = None,
    ) -> dict[str, InstrumentInfo]:
        """Resolve active front-month MCX futures and synchronize tokens across providers."""
        exec_resolved: dict[str, InstrumentInfo] = {}
        if hasattr(self.execution_broker, "resolve_active_mcx_futures"):
            exec_resolved = self.execution_broker.resolve_active_mcx_futures(
                symbol_roots, min_days_to_expiry=min_days_to_expiry, as_of=as_of
            )
        data_resolved: dict[str, InstrumentInfo] = {}
        if hasattr(self.data_provider, "resolve_active_mcx_futures"):
            data_resolved = self.data_provider.resolve_active_mcx_futures(
                symbol_roots, min_days_to_expiry=min_days_to_expiry, as_of=as_of
            )
        return exec_resolved or data_resolved

    # ── Order Execution & Portfolio (Delegated to Zerodha Personal) ──

    def get_margins(self, segment: str = "commodity") -> MarginInfo:
        """Fetch margin from Zerodha."""
        if hasattr(self.execution_broker, "get_margins"):
            sig = inspect.signature(self.execution_broker.get_margins)
            if "segment" in sig.parameters:
                return self.execution_broker.get_margins(segment=segment)
        return self.execution_broker.get_margins()

    def estimate_order_margin(self, request: OrderRequest) -> OrderMarginEstimate:
        """Estimate order margin from Zerodha."""
        return self.execution_broker.estimate_order_margin(request)

    def place_order(self, request: OrderRequest) -> str:
        """Submit live order EXCLUSIVELY to Zerodha."""
        logger.info("HYBRID ROUTE: Routing order for %s to ZERODHA", request.symbol)
        return self.execution_broker.place_order(request)

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> bool:
        """Modify order EXCLUSIVELY on Zerodha."""
        return self.execution_broker.modify_order(
            order_id=order_id,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            order_type=order_type,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order EXCLUSIVELY on Zerodha."""
        return self.execution_broker.cancel_order(order_id)

    def get_order_status(self, order_id: str) -> OrderState:
        """Fetch order status from Zerodha."""
        return self.execution_broker.get_order_status(order_id)

    def get_orders(self) -> list[OrderState]:
        """Fetch orders from Zerodha."""
        return self.execution_broker.get_orders()

    def get_trades(self) -> list[dict]:
        """Fetch trades from Zerodha."""
        return self.execution_broker.get_trades()

    def get_positions(self) -> list[PositionState]:
        """Fetch net positions from Zerodha."""
        return self.execution_broker.get_positions()

    def is_session_valid(self) -> bool:
        """Check session validity on BOTH data provider and execution broker, self-healing Angel token if dropped."""
        exec_valid = bool(self.execution_broker.is_session_valid())
        if not exec_valid:
            logger.critical("Hybrid session check FAILED: Zerodha execution session is invalid or expired.")
            return False

        data_valid = True
        if hasattr(self.data_provider, "is_session_valid"):
            data_valid = bool(self.data_provider.is_session_valid())
            if not data_valid and hasattr(self.data_provider, "authenticate"):
                logger.warning("Angel One data session invalid; attempting automatic self-heal authentication...")
                try:
                    self.data_provider.authenticate()
                    data_valid = bool(self.data_provider.is_session_valid())
                except Exception as auth_err:
                    logger.error("Angel One auto self-heal authentication failed: %s", auth_err)
                    data_valid = False

        if not data_valid:
            logger.critical("Hybrid session check FAILED: Angel One market data session is invalid or unauthenticated.")
            return False

        return True

