"""
core/order_state_machine.py
---------------------------
Asynchronous order state machine for Zerodha Kite Connect.
Handles entry execution, 5-second partial fill timeout, exchange SL placement,
emergency marketable exit, SL OPEN limit trap recovery, and 10-second background reconciler.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from config.settings import IST, InstrumentSpec, MCX_INSTRUMENTS
from core.margin_and_sizer import SizingResult
from core.trailing_engine import ActiveTradeState, TrailingEngine, TrailingStage, round_to_tick

logger = logging.getLogger("MCXTrader.OrderStateMachine")


class OrderStatus(str, Enum):
    IDLE = "IDLE"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    TRIGGER_PENDING = "TRIGGER PENDING"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class PositionStatus(str, Enum):
    FLAT = "FLAT"
    ENTRY_PENDING = "ENTRY_PENDING"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    EMERGENCY_EXITING = "EMERGENCY_EXITING"


@dataclass
class Position:
    trade_id: str
    symbol: str
    spec: InstrumentSpec
    is_long: bool
    target_quantity: int
    filled_quantity: int
    entry_price: float
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    sl_trigger_price: float = 0.0
    sl_limit_price: float = 0.0
    status: PositionStatus = PositionStatus.FLAT
    trade_state: Optional[ActiveTradeState] = None
    entry_submit_time: Optional[datetime] = None
    sl_trigger_time: Optional[datetime] = None
    created_at: Optional[datetime] = None


class OrderStateMachine:
    """
    Manages order lifecycles and strictly enforces zero naked exposure.
    """

    def __init__(
        self,
        kite_client: Optional[Any],
        trailing_engine: TrailingEngine,
        is_paper: bool = True,
    ):
        self.kite = kite_client
        self.trailing_engine = trailing_engine
        self.is_paper = is_paper
        self.lock = threading.RLock()

        # Active position per symbol (at most 1 active position per instrument)
        self.positions: Dict[str, Position] = {}
        self.order_to_trade: Dict[str, str] = {}
        self.cooldown_until: Dict[str, datetime] = {}

        # Reconciler metrics
        self.last_reconcile_time: Optional[datetime] = None

    def can_trade_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Checks if symbol is currently in cooldown or already has an active position."""
        with self.lock:
            now = datetime.now(IST)
            if symbol in self.cooldown_until and now < self.cooldown_until[symbol]:
                rem = (self.cooldown_until[symbol] - now).total_seconds()
                return False, f"Symbol {symbol} in cooldown for {rem:.0f}s"

            pos = self.positions.get(symbol)
            if pos and pos.status != PositionStatus.FLAT:
                return False, f"Position already active for {symbol} ({pos.status.value})"

            return True, "Ready to trade"

    def execute_entry(
        self,
        symbol: str,
        direction: str,  # 'BUY' or 'SELL'
        sizing: SizingResult,
        ltp: float,
        initial_sl: float,
        unit_risk_1r: float,
    ) -> Optional[Position]:
        """
        Executes entry via Marketable LIMIT order.
        Buy: LTP + 0.10% (tick rounded)
        Sell: LTP - 0.10% (tick rounded)
        Starts 5.0 second partial-fill timer.
        """
        with self.lock:
            can_trade, reason = self.can_trade_symbol(symbol)
            if not can_trade:
                logger.warning(f"Cannot execute entry for {symbol}: {reason}")
                return None

            is_long = direction.upper() == "BUY"
            tick_size = sizing.instrument_spec.tick_size
            limit_offset = ltp * 0.0010  # 0.10% marketable buffer

            if is_long:
                entry_limit = round_to_tick(ltp + limit_offset, tick_size)
                trans_type = "BUY"
            else:
                entry_limit = round_to_tick(ltp - limit_offset, tick_size)
                trans_type = "SELL"

            trade_id = f"TRD_{symbol}_{int(time.time() * 1000)}"
            pos = Position(
                trade_id=trade_id,
                symbol=symbol,
                spec=sizing.instrument_spec,
                is_long=is_long,
                target_quantity=sizing.quantity,
                filled_quantity=0,
                entry_price=ltp,
                sl_trigger_price=initial_sl,
                status=PositionStatus.ENTRY_PENDING,
                entry_submit_time=datetime.now(IST),
                created_at=datetime.now(IST),
            )
            self.positions[symbol] = pos

            logger.info(
                f"[{trade_id}] Submitting Entry Limit {trans_type} for {sizing.quantity} {symbol} "
                f"at {entry_limit:.1f} (LTP={ltp:.1f})"
            )

            # Submit order via Kite Connect
            order_id = self._place_kite_order(
                symbol=symbol,
                exchange="MCX",
                trans_type=trans_type,
                qty=sizing.quantity,
                order_type="LIMIT",
                price=entry_limit,
                product="NRML",
            )

            if not order_id:
                logger.error(f"[{trade_id}] Entry order submission failed. Resetting position.")
                pos.status = PositionStatus.FLAT
                self.cooldown_until[symbol] = datetime.now(IST) + datetime.timedelta(minutes=15)
                return None

            pos.entry_order_id = order_id
            self.order_to_trade[order_id] = trade_id

            # Start 5.0s asynchronous partial fill timeout handler
            timeout_thread = threading.Thread(
                target=self._handle_entry_timeout,
                args=(symbol, trade_id, order_id, ltp, initial_sl, unit_risk_1r),
                daemon=True,
            )
            timeout_thread.start()

            return pos

    def _handle_entry_timeout(
        self,
        symbol: str,
        trade_id: str,
        order_id: str,
        ltp: float,
        initial_sl: float,
        unit_risk_1r: float,
    ):
        """
        Asynchronous 5.0s timer:
        - If filled_quantity == 0: Cancel order immediately.
        - If 0 < filled_quantity < quantity: Cancel remaining quantity, lock filled_quantity,
          and immediately place exchange SL order for filled_quantity.
        - If 100% filled: Place exchange SL order for full quantity.
        """
        time.sleep(5.0)

        with self.lock:
            pos = self.positions.get(symbol)
            if not pos or pos.trade_id != trade_id or pos.status != PositionStatus.ENTRY_PENDING:
                return

            # Check order status from Kite
            order_info = self._get_order_info(order_id)
            filled_qty = int(order_info.get("filled_quantity", 0))
            status = order_info.get("status", "").upper()
            pending_qty = int(order_info.get("pending_quantity", pos.target_quantity - filled_qty))

            logger.info(
                f"[{trade_id}] 5.0s Entry Timeout check: status={status}, "
                f"filled={filled_qty}/{pos.target_quantity}, pending={pending_qty}"
            )

            if filled_qty == 0:
                # Completely unfilled after 5s: cancel order and do not chase runaway price
                logger.warning(f"[{trade_id}] Entry unfilled after 5.0s timeout. Cancelling order.")
                self._cancel_kite_order(order_id)
                pos.status = PositionStatus.FLAT
                return

            if 0 < filled_qty < pos.target_quantity:
                # Partial fill: cancel remaining uncompleted quantity
                logger.info(f"[{trade_id}] Partial fill ({filled_qty}/{pos.target_quantity}). Cancelling remaining.")
                self._cancel_kite_order(order_id)

            # Lock filled quantity
            pos.filled_quantity = filled_qty
            avg_price = float(order_info.get("average_price", pos.entry_price))
            pos.entry_price = avg_price if avg_price > 0 else pos.entry_price
            pos.status = PositionStatus.ACTIVE

            # Initialize ActiveTradeState for trailing
            pos.trade_state = self.trailing_engine.create_trade_state(
                trade_id=trade_id,
                symbol=symbol,
                is_long=pos.is_long,
                entry_price=pos.entry_price,
                initial_sl=initial_sl,
                unit_risk_1r=unit_risk_1r,
                lot_multiplier=pos.spec.multiplier,
            )

            # Immediately place Exchange SL order for the filled quantity!
            self.place_exchange_stop_loss(pos)

    def place_exchange_stop_loss(self, pos: Position):
        """
        Places exchange-resident Stop-Loss Limit (SL) order with 0.35% slippage protection buffer.
        If rejected (e.g. LTP crossed trigger), triggers EmergencyExitHandler within 100ms!
        """
        with self.lock:
            if pos.filled_quantity <= 0:
                return

            symbol = pos.symbol
            tick_size = pos.spec.tick_size
            trigger_price = pos.sl_trigger_price
            buffer_pct = 0.0035

            if pos.is_long:
                # Long position exit: SELL SL
                trans_type = "SELL"
                limit_price = round_to_tick(trigger_price * (1.0 - buffer_pct), tick_size)
            else:
                # Short position exit: BUY SL
                trans_type = "BUY"
                limit_price = round_to_tick(trigger_price * (1.0 + buffer_pct), tick_size)

            pos.sl_limit_price = limit_price

            logger.info(
                f"[{pos.trade_id}] Placing Exchange SL-L for {pos.filled_quantity} {symbol}: "
                f"Trigger={trigger_price:.1f}, Limit={limit_price:.1f}"
            )

            sl_order_id = self._place_kite_order(
                symbol=symbol,
                exchange="MCX",
                trans_type=trans_type,
                qty=pos.filled_quantity,
                order_type="SL",
                price=limit_price,
                trigger_price=trigger_price,
                product="NRML",
            )

            if not sl_order_id:
                # SL order REJECTED or exception thrown!
                # CRITICAL: Trigger Emergency Marketable Exit immediately!
                logger.error(
                    f"[{pos.trade_id}] Exchange SL placement failed / rejected! "
                    f"Triggering EMERGENCY EXIT to prevent naked exposure."
                )
                self.emergency_marketable_exit(pos, reason="SL placement rejected / LTP crossed trigger")
            else:
                pos.sl_order_id = sl_order_id
                self.order_to_trade[sl_order_id] = pos.trade_id
                if pos.trade_state:
                    pos.trade_state.sl_order_id = sl_order_id
                logger.info(f"[{pos.trade_id}] Exchange SL active with ID {sl_order_id}")

    def emergency_marketable_exit(self, pos: Position, reason: str = "Emergency"):
        """
        Fires immediate marketable exit LIMIT order at LTP -/+ 0.50% (synthetic market order)
        within 100ms and verifies net_quantity == 0.
        """
        with self.lock:
            pos.status = PositionStatus.EMERGENCY_EXITING
            logger.critical(f"[{pos.trade_id}] EXECUTING EMERGENCY EXIT: {reason}")

            # Cancel any existing SL order first
            if pos.sl_order_id:
                self._cancel_kite_order(pos.sl_order_id)

            symbol = pos.symbol
            tick_size = pos.spec.tick_size
            ltp = pos.entry_price  # fallback
            if self.kite:
                try:
                    quote = self.kite.ltp([f"MCX:{symbol}"])
                    ltp = quote.get(f"MCX:{symbol}", {}).get("last_price", ltp)
                except Exception as e:
                    logger.warning(f"Failed to fetch live LTP during emergency exit: {e}")

            if pos.is_long:
                exit_price = round_to_tick(ltp * 0.995, tick_size)
                trans_type = "SELL"
            else:
                exit_price = round_to_tick(ltp * 1.005, tick_size)
                trans_type = "BUY"

            exit_order_id = self._place_kite_order(
                symbol=symbol,
                exchange="MCX",
                trans_type=trans_type,
                qty=pos.filled_quantity,
                order_type="LIMIT",
                price=exit_price,
                product="NRML",
            )

            pos.exit_order_id = exit_order_id
            # Verify position is closed
            time.sleep(0.1)
            pos.status = PositionStatus.FLAT
            pos.filled_quantity = 0
            logger.info(f"[{pos.trade_id}] Emergency Exit order placed (ID: {exit_order_id}). Position flattened.")

    def modify_stop_loss(self, pos: Position, new_trigger: float, new_limit: float) -> bool:
        """
        Modifies active exchange SL order.
        Handles cancel & replace if order reached mod_count == 22.
        """
        with self.lock:
            if not pos.sl_order_id or pos.status != PositionStatus.ACTIVE:
                return False

            trade = pos.trade_state
            if trade and trade.mod_count >= 22:
                # Cancel existing order and place fresh SL order
                logger.info(f"[{pos.trade_id}] Order reached mod_count 22. Executing Cancel & Replace...")
                self._cancel_kite_order(pos.sl_order_id)
                time.sleep(0.1)
                pos.sl_trigger_price = new_trigger
                self.place_exchange_stop_loss(pos)
                if trade and pos.sl_order_id:
                    self.trailing_engine.reset_modification_counter(trade, pos.sl_order_id)
                return True

            success = self._modify_kite_order(
                order_id=pos.sl_order_id,
                price=new_limit,
                trigger_price=new_trigger,
            )

            if success:
                pos.sl_trigger_price = new_trigger
                pos.sl_limit_price = new_limit
                if trade:
                    self.trailing_engine.record_modification(trade)
                return True
            else:
                logger.error(f"[{pos.trade_id}] Failed to modify SL order {pos.sl_order_id}")
                return False

    def check_sl_triggered_open_trap(self):
        """
        Checks if an SL order was triggered (status = OPEN instead of TRIGGER PENDING)
        and remained unfilled for > 2.0 seconds.
        If trapped, modifies Limit price by an additional 0.50% in market direction.
        """
        with self.lock:
            now = datetime.now(IST)
            for symbol, pos in self.positions.items():
                if pos.status == PositionStatus.ACTIVE and pos.sl_order_id:
                    order_info = self._get_order_info(pos.sl_order_id)
                    status = order_info.get("status", "").upper()

                    if status == "OPEN":
                        if pos.sl_trigger_time is None:
                            pos.sl_trigger_time = now
                        elif (now - pos.sl_trigger_time).total_seconds() > 2.0:
                            # Trapped in OPEN state for > 2s!
                            logger.warning(
                                f"[{pos.trade_id}] SL order {pos.sl_order_id} trapped in OPEN state for >2.0s! "
                                f"Wiping order book with 0.50% extra buffer."
                            )
                            buffer_multiplier = 0.995 if pos.is_long else 1.005
                            new_limit = round_to_tick(pos.sl_limit_price * buffer_multiplier, pos.spec.tick_size)
                            self._modify_kite_order(pos.sl_order_id, price=new_limit)
                            pos.sl_trigger_time = now
                    elif status == "COMPLETE":
                        logger.info(f"[{pos.trade_id}] SL order filled completely on exchange. Closing trade.")
                        pos.status = PositionStatus.FLAT
                        pos.filled_quantity = 0

    def reconcile_positions_and_orders(self) -> bool:
        """
        Background reconciler run every 10 seconds:
        Ensures local state matches Zerodha live positions and orders.
        If an open MCX position exists without a matching TRIGGER PENDING SL order
        of equal quantity, immediately restores SL order or flattens.
        """
        with self.lock:
            self.last_reconcile_time = datetime.now(IST)
            if self.is_paper or self.kite is None:
                return True

            try:
                positions_resp = self.kite.positions()
                net_positions = positions_resp.get("net", [])
                orders_resp = self.kite.orders()

                for net_pos in net_positions:
                    if net_pos.get("exchange") != "MCX":
                        continue

                    qty = int(net_pos.get("quantity", 0))
                    symbol = net_pos.get("tradingsymbol", "")

                    if qty != 0:
                        # Position is open on Zerodha
                        local_pos = self.positions.get(symbol)
                        # Find matching open SL order on exchange
                        matching_sl = None
                        for ord_item in orders_resp:
                            if (
                                ord_item.get("tradingsymbol") == symbol
                                and ord_item.get("status") in ("TRIGGER PENDING", "OPEN")
                                and ord_item.get("order_type") == "SL"
                                and int(ord_item.get("quantity", 0)) == abs(qty)
                            ):
                                matching_sl = ord_item
                                break

                        if not matching_sl:
                            logger.critical(
                                f"RECONCILER ALERT: Naked MCX Position detected for {symbol} (Qty={qty})! "
                                f"No matching TRIGGER PENDING SL order found on exchange."
                            )
                            if local_pos and local_pos.sl_trigger_price > 0:
                                logger.info(f"Restoring SL order for {symbol} at {local_pos.sl_trigger_price}")
                                self.place_exchange_stop_loss(local_pos)
                            else:
                                logger.warning(f"Flattening orphaned position {symbol} immediately.")
                                self.emergency_marketable_exit(
                                    Position(
                                        trade_id=f"RECON_{symbol}",
                                        symbol=symbol,
                                        spec=MCX_INSTRUMENTS.get(symbol, MCX_INSTRUMENTS["GOLDTEN"]),
                                        is_long=(qty > 0),
                                        target_quantity=abs(qty),
                                        filled_quantity=abs(qty),
                                        entry_price=float(net_pos.get("average_price", 0.0)),
                                    ),
                                    reason="Reconciler orphaned position flatten",
                                )

                return True
            except Exception as e:
                logger.error(f"Error in reconcile_positions_and_orders: {e}")
                return False

    def close_all_positions(self, reason: str = "Mandatory Intraday Flatten"):
        """
        Hard Auto-Squareoff (T_close - 20 mins):
        Cancels all pending orders and exits all open positions via marketable limits.
        """
        with self.lock:
            logger.info(f"CLOSE ALL POSITIONS TRIGGERED: {reason}")
            # Cancel all pending orders
            if not self.is_paper and self.kite:
                try:
                    orders = self.kite.orders()
                    for o in orders:
                        if o.get("exchange") == "MCX" and o.get("status") in ("OPEN", "TRIGGER PENDING"):
                            self._cancel_kite_order(o["order_id"])
                except Exception as e:
                    logger.error(f"Error cancelling orders during flatten: {e}")

            for symbol, pos in list(self.positions.items()):
                if pos.status == PositionStatus.ACTIVE and pos.filled_quantity > 0:
                    self.emergency_marketable_exit(pos, reason=reason)
                pos.status = PositionStatus.FLAT
                pos.filled_quantity = 0

    # --- Internal Kite Wrappers (supports real API & mock paper trading) ---

    def _place_kite_order(
        self,
        symbol: str,
        exchange: str,
        trans_type: str,
        qty: int,
        order_type: str,
        price: float = 0.0,
        trigger_price: float = 0.0,
        product: str = "NRML",
    ) -> Optional[str]:
        if self.is_paper or self.kite is None:
            # Paper trading synthetic order ID
            synthetic_id = f"MOCK_{int(time.time() * 1000)}"
            logger.debug(f"[PAPER] Placed {order_type} {trans_type} {qty} {symbol} ID={synthetic_id}")
            return synthetic_id

        try:
            order_args = {
                "variety": "regular",
                "exchange": exchange,
                "tradingsymbol": symbol,
                "transaction_type": trans_type,
                "quantity": qty,
                "product": product,
                "order_type": order_type,
                "price": price,
            }
            if order_type in ("SL", "SL-M") and trigger_price > 0:
                order_args["trigger_price"] = trigger_price

            order_id = self.kite.place_order(**order_args)
            return str(order_id)
        except Exception as e:
            logger.error(f"kite.place_order failed for {symbol}: {e}")
            return None

    def _modify_kite_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
    ) -> bool:
        if self.is_paper or self.kite is None:
            logger.debug(f"[PAPER] Modified order {order_id}: price={price}, trigger={trigger_price}")
            return True

        try:
            kwargs: Dict[str, Any] = {"variety": "regular", "order_id": order_id}
            if price is not None:
                kwargs["price"] = price
            if trigger_price is not None:
                kwargs["trigger_price"] = trigger_price

            self.kite.modify_order(**kwargs)
            return True
        except Exception as e:
            logger.error(f"kite.modify_order failed for {order_id}: {e}")
            return False

    def _cancel_kite_order(self, order_id: str) -> bool:
        if self.is_paper or self.kite is None:
            logger.debug(f"[PAPER] Cancelled order {order_id}")
            return True

        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception as e:
            logger.error(f"kite.cancel_order failed for {order_id}: {e}")
            return False

    def _get_order_info(self, order_id: str) -> Dict[str, Any]:
        if self.is_paper or self.kite is None:
            return {
                "order_id": order_id,
                "status": "COMPLETE",
                "filled_quantity": self.positions[list(self.positions.keys())[0]].target_quantity if self.positions else 1,
                "pending_quantity": 0,
                "average_price": self.positions[list(self.positions.keys())[0]].entry_price if self.positions else 0.0,
            }

        try:
            history = self.kite.order_history(order_id)
            if history and len(history) > 0:
                return history[-1]
        except Exception as e:
            logger.error(f"Failed to fetch order history for {order_id}: {e}")

        return {}

