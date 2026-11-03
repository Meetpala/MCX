"""Abstract broker interface for APEX-3.

Defines the contract that all broker adapters (paper, Kite, etc.)
must implement. Every method that could trigger exchange interaction
documents that order acceptance ≠ execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional

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


class BrokerInterface(ABC):
    """Abstract base class for all APEX-3 broker adapters.

    Implementors must handle:
    - Thread safety for concurrent calls (e.g. quote + order status).
    - Rate limiting per exchange/broker constraints.
    - Graceful handling of network/API errors returning appropriate
      OrderStatus.ERROR or raising BrokerError.
    """

    # ── Margin & Capital ─────────────────────────────────────────────

    @abstractmethod
    def get_margins(self, segment: str = "commodity") -> MarginInfo:
        """Fetch current margin/funds information from the broker.

        Returns broker-reported values directly. Do not derive or
        double-count available vs. utilized margin.

        Returns:
            MarginInfo with equity_available, equity_used, collateral,
            net_value, and timestamp.
        """
        ...

    @abstractmethod
    def estimate_order_margin(self, request: OrderRequest) -> OrderMarginEstimate:
        """Estimate the margin required for a proposed order.

        Uses the broker's margin calculator for the exact symbol,
        product, side, order type, price, and quantity. The returned
        leverage_available is the *actual* leverage the broker will
        provide — 4× is a ceiling, never assumed.

        Args:
            request: The order to estimate margin for. Not placed.

        Returns:
            OrderMarginEstimate with required_margin, charges, total,
            and leverage_available.
        """
        ...

    # ── Order Management ─────────────────────────────────────────────

    @abstractmethod
    def place_order(self, request: OrderRequest) -> str:
        """Submit an order to the broker/exchange.

        WARNING: A returned order_id means the broker *accepted* the
        request — it does NOT guarantee exchange execution or fill.
        Stop-limit orders can fail to fill entirely. Always verify
        via get_order_status() and get_trades().

        Args:
            request: Fully populated OrderRequest.

        Returns:
            Broker-assigned order_id string.

        Raises:
            BrokerError: On API/network failure.
        """
        ...

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> bool:
        """Modify an open/pending order.

        WARNING: Kite Connect caps modifications at 25 per order per
        session. Exceeding this will result in a rejected modification.
        Modification acceptance does NOT guarantee the exchange applied
        the change — always verify via get_order_status().

        Args:
            order_id: Broker-assigned order ID.
            quantity: New quantity (optional).
            price: New limit price (optional).
            trigger_price: New trigger/stop price (optional).
            order_type: New order type string (optional).

        Returns:
            True if the modification request was accepted by the broker.
            Acceptance is not confirmation of exchange-level change.
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Request cancellation of an open/pending order.

        WARNING: Cancellation acceptance does NOT guarantee the order
        was not already filled or partially filled on the exchange.
        Always reconcile via get_order_status() and get_trades().

        Args:
            order_id: Broker-assigned order ID.

        Returns:
            True if the cancellation request was accepted by the broker.
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderState:
        """Fetch the current status of a specific order.

        Use this to verify whether an order was actually filled,
        partially filled, rejected, or still pending. Do not rely
        solely on place_order() success.

        Args:
            order_id: Broker-assigned order ID.

        Returns:
            OrderState with current status, filled_quantity,
            average_price, etc.
        """
        ...

    @abstractmethod
    def get_orders(self) -> list[OrderState]:
        """Fetch all orders for the current trading session.

        Returns the full order book including completed, cancelled,
        and rejected orders. Use for reconciliation.

        Returns:
            List of OrderState for every order placed this session.
        """
        ...

    @abstractmethod
    def get_trades(self) -> list[dict]:
        """Fetch all executed trades for the current session.

        Trades represent actual exchange executions, not order
        placements. One order can produce multiple trades
        (partial fills).

        Returns:
            List of trade dicts with at minimum: order_id, trade_id,
            symbol, side, quantity, price, timestamp.
        """
        ...

    # ── Position Tracking ────────────────────────────────────────────

    @abstractmethod
    def get_positions(self) -> list[PositionState]:
        """Fetch current net positions from the broker.

        Returns broker-confirmed positions, not locally tracked state.
        Use for reconciliation against local PositionState.

        Returns:
            List of PositionState for all open positions.
        """
        ...

    # ── Market Data ──────────────────────────────────────────────────

    @abstractmethod
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Quote:
        """Fetch a live quote snapshot for a symbol.

        Args:
            symbol: Trading symbol (e.g. "RELIANCE").
            exchange: Exchange segment (default "NSE").

        Returns:
            Quote with last_price, bid, ask, volume, timestamp.
        """
        ...

    def get_quotes_batch(self, symbols: list[str], exchange: str = "NSE") -> dict[str, Quote]:
        """Fetch live quote snapshots for multiple symbols in a batch.

        Subclasses may override with native multi-symbol batch APIs.
        """
        return {sym: self.get_quote(sym, exchange=exchange) for sym in symbols}

    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        interval: str,
        from_dt: datetime,
        to_dt: datetime,
        exchange: str = "NSE",
    ) -> list[Candle]:
        """Fetch historical OHLCV candle data.

        Args:
            symbol: Trading symbol.
            interval: Candle interval (e.g. "minute", "5minute",
                      "15minute", "day").
            from_dt: Start datetime (inclusive).
            to_dt: End datetime (inclusive).
            exchange: Exchange segment (default "NSE").

        Returns:
            List of Candle objects sorted by timestamp ascending.
        """
        ...

    def resolve_active_mcx_futures(
        self,
        symbol_roots: list[str],
        min_days_to_expiry: int = 5,
        as_of: Optional[date] = None,
    ) -> dict[str, InstrumentInfo]:
        """Resolve active front-month MCX futures contracts for given symbol roots.

        Filters out contracts within the delivery tender period (< min_days_to_expiry).
        """
        return {}

    @abstractmethod
    def get_instrument_info(self, symbol: str, exchange: Optional[str] = None) -> InstrumentInfo:
        """Fetch exchange instrument metadata.

        Args:
            symbol: Trading symbol.
            exchange: Exchange segment (default "NSE").

        Returns:
            InstrumentInfo with tick_size, lot_size, circuit limits, etc.
        """
        ...

    # ── Session ──────────────────────────────────────────────────────

    @abstractmethod
    def is_session_valid(self) -> bool:
        """Check whether the broker session/token is still valid.

        Kite sessions expire daily after market close and require
        re-authentication via the login flow.

        Returns:
            True if the session is active and API calls will succeed.
        """
        ...
