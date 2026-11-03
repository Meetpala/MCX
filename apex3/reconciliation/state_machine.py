"""APEX-3 Order and Position State Machine.

Maintains durable state across order lifecycle events and system restarts.
Distinguishes strategy-owned orders from external/manual orders using
strategy tags.

Order Lifecycle States:
PENDING -> OPEN -> TRIGGER_PENDING / PARTIALLY_FILLED -> COMPLETE / CANCELLED / REJECTED

Position States:
FLAT -> ENTERING -> OPEN_UNPROTECTED -> OPEN_PROTECTED -> EXITING -> FLAT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional

from apex3.models import (
    OrderRequest,
    OrderState,
    OrderStatus,
    PositionState,
    Side,
)


class PositionPhase(Enum):
    """Macro lifecycle phase for a position."""
    FLAT = auto()
    CLOSED = auto()
    ENTERING = auto()          # Entry order sent, waiting for fill
    OPEN_UNPROTECTED = auto()  # Filled, but protective stop not yet confirmed
    OPEN_PROTECTED = auto()    # Filled and verified protective stop active
    EXITING = auto()           # Exit or square-off order in transit
    ERROR = auto()             # Reconciliation mismatch or critical alert


@dataclass
class ManagedPosition:
    """Internal tracking state for a strategy-managed symbol position."""
    symbol: str
    phase: PositionPhase = PositionPhase.FLAT
    side: Side = Side.BUY
    quantity: int = 0
    average_entry: float = 0.0
    current_price: float = 0.0
    entry_order_id: str = ""
    protective_order_id: str = ""
    exit_order_id: str = ""
    initial_stop_price: float = 0.0
    current_stop_price: float = 0.0
    target_price: float = 0.0
    target_order_id: str = ""
    best_favorable_price: float = 0.0
    r_value: float = 0.0
    trail_activated: bool = False
    fee_lock_applied: bool = False
    stop_modification_count: int = 0
    strategy_tag: str = ""
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_timestamp: Optional[datetime] = None
    requested_quantity: int = 0
    last_reconciled: Optional[datetime] = None

    def to_position_state(self) -> PositionState:
        """Convert to public PositionState model."""
        return PositionState(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            average_entry=self.average_entry,
            current_price=self.current_price,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            protective_order_id=self.protective_order_id,
            protective_stop_price=self.current_stop_price,
            initial_stop_price=self.initial_stop_price,
            is_protected=(self.phase == PositionPhase.OPEN_PROTECTED),
            target_price=self.target_price,
            target_order_id=self.target_order_id,
            best_favorable_price=self.best_favorable_price,
            r_value=self.r_value,
            current_r_multiple=(
                (self.best_favorable_price - self.average_entry) / self.r_value
                if self.side == Side.BUY and self.r_value > 0
                else (self.average_entry - self.best_favorable_price) / self.r_value
                if self.side == Side.SELL and self.r_value > 0
                else 0.0
            ),
            trail_activated=self.trail_activated,
            fee_lock_applied=self.fee_lock_applied,
            strategy_tag=self.strategy_tag,
            entry_time=self.entry_timestamp,
        )


class StateMachine:
    """In-memory state machine for strategy orders and positions."""

    def __init__(self, session_tag_prefix: str = "APEX3"):
        self.session_tag_prefix = session_tag_prefix
        self.orders: dict[str, OrderState] = {}
        self.positions: dict[str, ManagedPosition] = {}

    def is_strategy_order(self, tag: Optional[str]) -> bool:
        """Check if an order tag belongs to this APEX-3 session."""
        if not tag:
            return False
        return tag.startswith(self.session_tag_prefix)

    def get_or_create_position(self, symbol: str) -> ManagedPosition:
        """Get or initialize a ManagedPosition for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = ManagedPosition(symbol=symbol)
        return self.positions[symbol]

    def record_order_request(self, order_id: str, request: OrderRequest) -> OrderState:
        """Record an order that was submitted to the broker."""
        state = OrderState(
            order_id=order_id,
            request=request,
            status=OrderStatus.OPEN,
            pending_quantity=request.quantity,
            modification_count=0,
            is_strategy_owned=self.is_strategy_order(request.tag),
        )
        self.orders[order_id] = state
        return state

    def update_order_fill(
        self,
        order_id: str,
        filled_qty: int,
        avg_price: float,
        is_complete: bool,
    ) -> Optional[OrderState]:
        """Update order fill information from trade confirmations."""
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]
        order.filled_quantity = filled_qty
        order.pending_quantity = max(0, order.request.quantity - filled_qty)
        order.average_price = avg_price
        order.status = OrderStatus.COMPLETE if is_complete else OrderStatus.PARTIALLY_FILLED
        return order
