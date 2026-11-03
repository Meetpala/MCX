"""Paper trading broker adapter for APEX-3.

Simulates order placement, fills, and position tracking entirely
in-memory. NEVER connects to any real broker or exchange. Used for
strategy development, backtesting, and integration testing.

All fills are simulated — slippage is configurable but the adapter
makes no claims about real-market execution quality.
"""
from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from apex3.broker.interface import BrokerInterface
from apex3.models import (
    Candle,
    InstrumentInfo,
    MarginInfo,
    OrderMarginEstimate,
    OrderRequest,
    OrderState,
    OrderStatus,
    OrderType,
    PositionState,
    Product,
    Quote,
    Side,
)

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Kite caps order modifications at 25 per order per session
_MAX_MODIFICATIONS_PER_ORDER = 25

# Default MIS leverage ceiling — 4× is a ceiling, never assumed.
# Paper broker uses a conservative 5× for simulation to allow
# testing leverage scenarios, but real leverage varies by symbol.
_DEFAULT_MIS_LEVERAGE = 5.0


@dataclass
class _PaperFillConfig:
    """Configuration for simulated order fills."""
    # Slippage in price ticks applied against the trader
    # (buy fills higher, sell fills lower)
    slippage_ticks: int = 1
    tick_size: float = 0.05
    # Probability that a limit order fills (1.0 = always fills)
    limit_fill_probability: float = 1.0
    # Probability that a SL/SL-M order triggers AND fills
    # In reality SL orders can fail to fill — this models that risk
    sl_fill_probability: float = 0.95


class PaperBroker(BrokerInterface):
    """In-memory paper trading broker for APEX-3.

    Maintains simulated orders, positions, and margin state.
    Generates deterministic market data from loaded fixtures.
    Logs every action for journal replay.

    Args:
        starting_margin: Initial available margin in INR.
            Defaults to 100,000.
        fill_config: Optional fill simulation configuration.
    """

    def __init__(
        self,
        starting_margin: float = 100_000.0,
        fill_config: Optional[_PaperFillConfig] = None,
        initial_margin: Optional[float] = None,
    ) -> None:
        if initial_margin is not None:
            starting_margin = initial_margin
        self._starting_margin = starting_margin
        self._fill_config = fill_config or _PaperFillConfig()

        # ── Internal state ───────────────────────────────────────
        self._margin_available: float = starting_margin
        self._margin_used: float = 0.0

        # order_id -> OrderState
        self._orders: dict[str, OrderState] = {}
        # trade records (list of dicts mirroring broker trade format)
        self._trades: list[dict[str, Any]] = []
        # symbol -> PositionState
        self._positions: dict[str, PositionState] = {}
        # symbol -> InstrumentInfo
        self._instruments: dict[str, InstrumentInfo] = {}
        # symbol -> list[Candle]  (fixture data for historical)
        self._fixture_candles: dict[str, list[Candle]] = {}
        # symbol -> Quote (latest simulated quote)
        self._fixture_quotes: dict[str, Quote] = {}

        # ── Audit log for journal replay ─────────────────────────
        self._action_log: list[dict[str, Any]] = []

        # Pre-populate a default instrument
        self._instruments["RELIANCE"] = InstrumentInfo(
            symbol="RELIANCE",
            exchange="NSE",
            tick_size=0.05,
            lot_size=1,
            exchange_token="738561",
            instrument_token="738561",
            upper_circuit=3000.0,
            lower_circuit=2000.0,
            mis_available=True,
        )
        logger.info(
            "PaperBroker initialized: margin=%.2f, slippage_ticks=%d",
            starting_margin,
            self._fill_config.slippage_ticks,
        )

    # ── Fixture Loading ──────────────────────────────────────────

    def load_fixture(self, fixture_name: str) -> None:
        """Load a deterministic test fixture for candle/quote data.

        Fixtures provide repeatable market data for testing strategy
        logic without any network dependency.

        Args:
            fixture_name: Name of the fixture scenario to load.
                Supported: "breakout_up", "breakout_down",
                "range_bound", "gap_open".
        """
        now = datetime.now(IST).replace(second=0, microsecond=0)
        base_price = 2500.0

        generators: dict[str, list[tuple[float, float, float, float, int]]] = {
            # (open, high, low, close, volume) relative to base
            "breakout_up": [
                (0, 10, -5, 8, 100000),
                (8, 20, 5, 18, 150000),
                (18, 35, 15, 32, 200000),
                (32, 50, 28, 48, 250000),
                (48, 55, 40, 52, 180000),
            ],
            "breakout_down": [
                (0, 5, -10, -8, 100000),
                (-8, -5, -25, -20, 150000),
                (-20, -15, -40, -35, 200000),
                (-35, -28, -55, -50, 250000),
                (-50, -42, -60, -55, 180000),
            ],
            "range_bound": [
                (0, 8, -5, 3, 80000),
                (3, 6, -3, -1, 70000),
                (-1, 5, -6, 2, 75000),
                (2, 7, -4, 0, 65000),
                (0, 4, -3, 1, 60000),
            ],
            "gap_open": [
                (30, 38, 25, 35, 200000),
                (35, 42, 30, 40, 180000),
                (40, 45, 35, 38, 150000),
                (38, 40, 32, 34, 120000),
                (34, 36, 30, 31, 100000),
            ],
        }

        if fixture_name not in generators:
            logger.warning(
                "Unknown fixture '%s', defaulting to 'range_bound'",
                fixture_name,
            )
            fixture_name = "range_bound"

        bars = generators[fixture_name]
        candles: list[Candle] = []
        for i, (o, h, l, c, v) in enumerate(bars):
            ts = now - timedelta(minutes=(len(bars) - i) * 5)
            candles.append(
                Candle(
                    timestamp=ts,
                    open=base_price + o,
                    high=base_price + h,
                    low=base_price + l,
                    close=base_price + c,
                    volume=v,
                    is_complete=True,
                )
            )

        symbol = "RELIANCE"
        self._fixture_candles[symbol] = candles

        # Set the quote to the last candle's close
        last = candles[-1]
        spread = self._fill_config.tick_size * 2
        self._fixture_quotes[symbol] = Quote(
            symbol=symbol,
            exchange="NSE",
            last_price=last.close,
            bid=last.close - spread,
            ask=last.close + spread,
            volume=last.volume,
            timestamp=last.timestamp,
            is_fresh=True,
        )

        self._log_action("load_fixture", {"fixture": fixture_name, "symbol": symbol})
        logger.info("Loaded fixture '%s' with %d candles", fixture_name, len(candles))

    # ── BrokerInterface Implementation ───────────────────────────

    def get_margins(self, segment: str = "commodity") -> MarginInfo:
        """Return current simulated margin state."""
        now = datetime.now(IST)
        return MarginInfo(
            equity_available=self._margin_available,
            equity_used=self._margin_used,
            collateral=0.0,
            net_value=self._margin_available + self._margin_used,
            timestamp=now,
        )

    def resolve_active_mcx_futures(
        self,
        symbol_roots: list[str],
        min_days_to_expiry: int = 5,
        as_of: Optional[date] = None,
    ) -> dict[str, InstrumentInfo]:
        """Simulate active front-month MCX futures contracts for paper trading."""
        today = as_of or datetime.now(IST).date()
        target_expiry = today + timedelta(days=25)
        results = {}
        for root in symbol_roots:
            clean = root.upper().strip()
            exp_str = target_expiry.strftime("%d%b%y").upper()
            tsym = f"{clean}{exp_str}FUT"
            lot_mult = 1
            if "SILVERM" in clean:
                lot_mult = 5
            elif "GOLDM" in clean:
                lot_mult = 10
            elif "SILVER" in clean:
                lot_mult = 30
            elif "GOLD" in clean and clean not in ("GOLDGUINEA", "GOLDPETAL", "GOLDTEN"):
                lot_mult = 100

            info = InstrumentInfo(
                symbol=tsym,
                exchange="MCX",
                tick_size=1.0,
                lot_size=1,
                order_lot_size=1,
                lot_multiplier=lot_mult,
                expiry_date=target_expiry,
                days_to_expiry=(target_expiry - today).days,
                instrument_token=f"PAPER-{clean}",
                mis_available=True,
            )
            self._instruments[tsym] = info
            self._instruments[clean] = info
            results[clean] = info
        return results

    def estimate_order_margin(self, request: OrderRequest) -> OrderMarginEstimate:
        """Estimate margin for a proposed order using simulated leverage.

        Uses a conservative fixed leverage factor. Real leverage varies
        by symbol and broker policy — 4× is a ceiling, never assumed.
        """
        inst = self._instruments.get(request.symbol)
        lot_mult = inst.lot_multiplier if inst else (5 if "SILVERM" in request.symbol else (10 if "GOLDM" in request.symbol else 1))
        notional = request.price * lot_mult * request.quantity if request.price > 0 else 0.0
        # Simulate MIS leverage: margin = notional / leverage
        leverage = min(_DEFAULT_MIS_LEVERAGE, 4.0)  # Cap at 4×
        required = notional / leverage if leverage > 0 else notional
        # Simulated charges: ~0.05% of notional
        charges = notional * 0.0005
        return OrderMarginEstimate(
            required_margin=round(required, 2),
            charges=round(charges, 2),
            total=round(required + charges, 2),
            leverage_available=leverage,
        )

    def place_order(self, request: OrderRequest) -> str:
        """Simulate order placement.

        WARNING: A returned order_id means the paper broker *accepted*
        the order internally. In real markets, acceptance does NOT
        guarantee fill. Stop-limit orders can fail to fill entirely.

        Returns:
            Simulated order_id string.
        """
        order_id = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
        now = datetime.now(IST)

        order_state = OrderState(
            order_id=order_id,
            request=copy.deepcopy(request),
            status=OrderStatus.PENDING,
            filled_quantity=0,
            pending_quantity=request.quantity,
            average_price=0.0,
            modification_count=0,
            last_updated=now,
            status_message="Order placed (paper)",
            is_strategy_owned=True,
        )
        self._orders[order_id] = order_state

        self._log_action("place_order", {
            "order_id": order_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": request.quantity,
            "order_type": request.order_type.value,
            "price": request.price,
            "trigger_price": request.trigger_price,
        })
        logger.info(
            "Paper order placed: %s %s %s qty=%d @ %.2f (trigger=%.2f) -> %s",
            request.side.value,
            request.symbol,
            request.order_type.value,
            request.quantity,
            request.price,
            request.trigger_price,
            order_id,
        )

        # Attempt immediate simulated fill for MARKET and LIMIT orders
        self._attempt_fill(order_id)
        return order_id

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> bool:
        """Modify an existing paper order.

        Enforces the 25-modification cap per order. Modification
        acceptance in real markets does NOT guarantee the exchange
        applied the change.
        """
        if order_id not in self._orders:
            logger.warning("Modify failed: order %s not found", order_id)
            return False

        order = self._orders[order_id]

        # Only open/pending/trigger_pending orders can be modified
        if order.status not in (
            OrderStatus.OPEN,
            OrderStatus.PENDING,
            OrderStatus.TRIGGER_PENDING,
        ):
            logger.warning(
                "Modify failed: order %s status=%s not modifiable",
                order_id,
                order.status.value,
            )
            return False

        # Enforce 25-modification cap
        if order.modification_count >= _MAX_MODIFICATIONS_PER_ORDER:
            order.status = OrderStatus.REJECTED
            order.status_message = (
                f"Modification cap ({_MAX_MODIFICATIONS_PER_ORDER}) exceeded"
            )
            logger.warning(
                "Modify rejected: order %s hit %d-mod cap",
                order_id,
                _MAX_MODIFICATIONS_PER_ORDER,
            )
            self._log_action("modify_rejected_cap", {"order_id": order_id})
            raise ValueError(f"Maximum allowed order modifications ({_MAX_MODIFICATIONS_PER_ORDER}) exceeded")

        # Apply modifications to the underlying request
        if quantity is not None:
            order.request.quantity = quantity
            order.pending_quantity = quantity - order.filled_quantity
        if price is not None:
            order.request.price = price
        if trigger_price is not None:
            order.request.trigger_price = trigger_price
        if order_type is not None:
            order.request.order_type = OrderType(order_type)

        order.modification_count += 1
        order.last_updated = datetime.now(IST)
        order.status_message = f"Modified (count={order.modification_count})"

        self._log_action("modify_order", {
            "order_id": order_id,
            "mod_count": order.modification_count,
            "new_price": price,
            "new_trigger": trigger_price,
            "new_qty": quantity,
        })
        logger.info(
            "Paper order modified: %s (mod #%d)",
            order_id,
            order.modification_count,
        )
        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order.

        WARNING: In real markets, cancellation acceptance does NOT
        guarantee the order was not already filled on the exchange.
        """
        if order_id not in self._orders:
            logger.warning("Cancel failed: order %s not found", order_id)
            return False

        order = self._orders[order_id]
        if order.status in (
            OrderStatus.COMPLETE,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        ):
            logger.warning(
                "Cancel failed: order %s already %s",
                order_id,
                order.status.value,
            )
            return False

        order.status = OrderStatus.CANCELLED
        order.last_updated = datetime.now(IST)
        order.status_message = "Cancelled by strategy"

        self._log_action("cancel_order", {"order_id": order_id})
        logger.info("Paper order cancelled: %s", order_id)
        return True

    def get_order_status(self, order_id: str) -> OrderState:
        """Fetch current state of a paper order."""
        if order_id not in self._orders:
            return OrderState(
                order_id=order_id,
                status=OrderStatus.ERROR,
                status_message="Order not found in paper broker",
            )
        return copy.deepcopy(self._orders[order_id])

    def get_orders(self) -> list[OrderState]:
        """Return all paper orders for this session."""
        return [copy.deepcopy(o) for o in self._orders.values()]

    def get_trades(self) -> list[dict]:
        """Return all simulated trades for this session."""
        return [copy.deepcopy(t) for t in self._trades]

    def get_positions(self) -> list[PositionState]:
        """Return all current paper positions."""
        return [
            copy.deepcopy(p)
            for p in self._positions.values()
            if p.quantity > 0
        ]

    def get_quotes_batch(self, symbols: list[str], exchange: str = "NSE") -> dict[str, Quote]:
        """Return latest simulated quotes for multiple symbols."""
        return {sym: self.get_quote(sym, exchange=exchange) for sym in symbols}

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Quote:
        """Return the latest simulated quote for a symbol.

        If no fixture quote exists, returns a default quote at 0.
        """
        if symbol in self._fixture_quotes:
            return self._fixture_quotes[symbol]

        now = datetime.now(IST)
        return Quote(
            symbol=symbol,
            exchange=exchange,
            last_price=0.0,
            bid=0.0,
            ask=0.0,
            volume=0,
            timestamp=now,
            is_fresh=False,
        )

    def get_historical(
        self,
        symbol: str,
        interval: str,
        from_dt: datetime,
        to_dt: datetime,
        exchange: str = "NSE",
    ) -> list[Candle]:
        """Return fixture candle data filtered by time range.

        Only returns data loaded via load_fixture(). No network calls.
        """
        if symbol not in self._fixture_candles:
            logger.warning("No fixture data for %s", symbol)
            return []

        def _is_between(c_ts: datetime, f_dt: datetime | None, t_dt: datetime | None) -> bool:
            if f_dt is not None:
                ts_cmp = c_ts
                f_cmp = f_dt
                if ts_cmp.tzinfo is not None and f_cmp.tzinfo is None:
                    f_cmp = f_cmp.replace(tzinfo=ts_cmp.tzinfo)
                elif ts_cmp.tzinfo is None and f_cmp.tzinfo is not None:
                    ts_cmp = ts_cmp.replace(tzinfo=f_cmp.tzinfo)
                if f_cmp > ts_cmp:
                    return False
            if t_dt is not None:
                ts_cmp = c_ts
                t_cmp = t_dt
                if ts_cmp.tzinfo is not None and t_cmp.tzinfo is None:
                    t_cmp = t_cmp.replace(tzinfo=ts_cmp.tzinfo)
                elif ts_cmp.tzinfo is None and t_cmp.tzinfo is not None:
                    ts_cmp = ts_cmp.replace(tzinfo=t_cmp.tzinfo)
                if ts_cmp > t_cmp:
                    return False
            return True

        return [
            c
            for c in self._fixture_candles[symbol]
            if _is_between(c.timestamp, from_dt, to_dt)
        ]

    def get_instrument_info(self, symbol: str, exchange: str = "MCX") -> InstrumentInfo:
        """Return instrument metadata from internal registry."""
        if symbol in self._instruments:
            return self._instruments[symbol]

        # Standard NSE price-band tick sizes for radar stocks
        default_ticks = {
            "MARUTI": 1.00,
            "JIOFIN": 0.01,
            "TATASTEEL": 0.01,
        }
        tick = 1.0 if exchange == "MCX" else default_ticks.get(symbol.upper(), 0.05)
        lot_mult = 1
        if exchange == "MCX":
            if "SILVERM" in symbol.upper():
                lot_mult = 5
            elif "GOLDM" in symbol.upper():
                lot_mult = 10
            elif "SILVER" in symbol.upper():
                lot_mult = 30
            elif "GOLD" in symbol.upper() and not any(p in symbol.upper() for p in ("GUINEA", "PETAL", "TEN")):
                lot_mult = 100

        # Return a generic default for unknown instruments
        return InstrumentInfo(
            symbol=symbol,
            exchange=exchange,
            tick_size=tick,
            lot_size=1,
            order_lot_size=1,
            lot_multiplier=lot_mult,
            mis_available=True,
        )

    def is_session_valid(self) -> bool:
        """Paper broker sessions are always valid."""
        return True

    # ── Simulated Fill Engine ────────────────────────────────────

    def _attempt_fill(self, order_id: str) -> None:
        """Attempt to simulate a fill for the given order.

        Fill logic by order type:
        - MARKET: Always fills at last_price ± slippage.
        - LIMIT: Fills at limit price (assumes price reached).
        - SL / SL-M: Transitions to TRIGGER_PENDING. Call
          simulate_tick() to trigger based on price movement.

        Slippage is applied *against* the trader (buy fills higher,
        sell fills lower) to model realistic execution.
        """
        if order_id not in self._orders:
            return

        order = self._orders[order_id]
        request = order.request
        cfg = self._fill_config

        if request.order_type == OrderType.MARKET:
            # Market orders fill immediately with slippage
            quote = self.get_quote(request.symbol, request.exchange)
            slippage = cfg.slippage_ticks * cfg.tick_size
            if request.side == Side.BUY:
                fill_price = quote.ask + slippage if quote.ask > 0 else quote.last_price + slippage
            else:
                fill_price = quote.bid - slippage if quote.bid > 0 else quote.last_price - slippage

            fill_price = max(fill_price, cfg.tick_size)  # Never fill at 0 or negative
            self._execute_fill(order_id, fill_price, request.quantity)

        elif request.order_type == OrderType.LIMIT:
            # Limit orders fill at the limit price (optimistic simulation)
            fill_price = request.price
            if fill_price > 0:
                self._execute_fill(order_id, fill_price, request.quantity)
            else:
                order.status = OrderStatus.REJECTED
                order.status_message = "Limit price must be > 0"

        elif request.order_type in (OrderType.SL, OrderType.SL_M):
            # SL orders go to TRIGGER_PENDING — they don't fill immediately
            order.status = OrderStatus.TRIGGER_PENDING
            order.status_message = (
                f"SL trigger pending at {request.trigger_price:.2f}"
            )
            logger.info(
                "Paper SL order %s: trigger pending at %.2f",
                order_id,
                request.trigger_price,
            )

    def simulate_tick(self, symbol: str, price: float) -> list[str]:
        """Simulate a market tick to trigger pending SL/SL-M orders.

        Call this during backtesting/paper trading to simulate price
        movement that may trigger stop-loss orders.

        Args:
            symbol: The symbol that ticked.
            price: The tick price.

        Returns:
            List of order_ids that were triggered and filled.
        """
        triggered: list[str] = []
        now = datetime.now(IST)

        # Update the quote for this symbol
        self._fixture_quotes[symbol] = Quote(
            symbol=symbol,
            exchange="NSE",
            last_price=price,
            bid=price - self._fill_config.tick_size,
            ask=price + self._fill_config.tick_size,
            volume=0,
            timestamp=now,
            is_fresh=True,
        )

        for order_id, order in self._orders.items():
            if order.status != OrderStatus.TRIGGER_PENDING:
                continue
            if order.request.symbol != symbol:
                continue

            req = order.request
            should_trigger = False

            # For BUY SL: trigger when price >= trigger_price (price rising past stop)
            # For SELL SL: trigger when price <= trigger_price (price falling past stop)
            if req.side == Side.BUY and price >= req.trigger_price:
                should_trigger = True
            elif req.side == Side.SELL and price <= req.trigger_price:
                should_trigger = True

            if should_trigger:
                # SL-M fills at market (tick price ± slippage)
                # SL (limit) fills at the limit price if reachable,
                # but can fail — modeled by sl_fill_probability
                import random
                if random.random() > self._fill_config.sl_fill_probability:
                    # SL order failed to fill — this happens in real markets
                    order.status = OrderStatus.CANCELLED
                    order.status_message = (
                        "SL trigger hit but fill failed (simulated slippage gap)"
                    )
                    logger.warning(
                        "Paper SL %s triggered but FAILED to fill (simulated)",
                        order_id,
                    )
                    self._log_action("sl_fill_failed", {
                        "order_id": order_id,
                        "trigger_price": req.trigger_price,
                        "tick_price": price,
                    })
                    continue

                if req.order_type == OrderType.SL_M:
                    slippage = self._fill_config.slippage_ticks * self._fill_config.tick_size
                    fill_price = (
                        price + slippage if req.side == Side.BUY else price - slippage
                    )
                else:
                    # SL limit fills at the limit price
                    fill_price = req.price if req.price > 0 else price

                fill_price = max(fill_price, self._fill_config.tick_size)
                self._execute_fill(order_id, fill_price, req.quantity)
                triggered.append(order_id)

        return triggered

    def _execute_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: int,
    ) -> None:
        """Record a simulated fill and update positions/margin.

        Args:
            order_id: The order being filled.
            fill_price: Simulated fill price (including slippage).
            fill_quantity: Number of shares filled.
        """
        order = self._orders[order_id]
        request = order.request
        now = datetime.now(IST)

        # Round fill price to tick size
        tick = self._fill_config.tick_size
        fill_price = round(round(fill_price / tick) * tick, 2)

        # Update order state
        order.filled_quantity = fill_quantity
        order.pending_quantity = 0
        order.average_price = fill_price
        order.status = OrderStatus.COMPLETE
        order.last_updated = now
        order.status_message = f"Filled at {fill_price:.2f}"

        # Record the trade
        trade_id = f"PTRADE-{uuid.uuid4().hex[:8].upper()}"
        self._trades.append({
            "trade_id": trade_id,
            "order_id": order_id,
            "symbol": request.symbol,
            "exchange": request.exchange,
            "side": request.side.value,
            "quantity": fill_quantity,
            "price": fill_price,
            "product": request.product.value,
            "timestamp": now.isoformat(),
        })

        # Update position
        self._update_position(request, fill_price, fill_quantity, now)

        # Update margin — deduct for entry, credit for exit
        notional = fill_price * fill_quantity
        margin_impact = notional / min(_DEFAULT_MIS_LEVERAGE, 4.0)

        if self._is_opening_trade(request):
            self._margin_available -= margin_impact
            self._margin_used += margin_impact
        else:
            self._margin_available += margin_impact
            self._margin_used = max(0.0, self._margin_used - margin_impact)

        self._log_action("fill", {
            "order_id": order_id,
            "trade_id": trade_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": fill_quantity,
            "price": fill_price,
            "margin_available": round(self._margin_available, 2),
            "margin_used": round(self._margin_used, 2),
        })
        logger.info(
            "Paper fill: %s %s %d @ %.2f (order=%s, trade=%s)",
            request.side.value,
            request.symbol,
            fill_quantity,
            fill_price,
            order_id,
            trade_id,
        )

    def _update_position(
        self,
        request: OrderRequest,
        fill_price: float,
        fill_quantity: int,
        now: datetime,
    ) -> None:
        """Update internal position state after a fill."""
        symbol = request.symbol

        if symbol not in self._positions:
            self._positions[symbol] = PositionState(
                symbol=symbol,
                side=request.side,
                quantity=0,
                average_entry=0.0,
                product=request.product,
                entry_time=now,
            )

        pos = self._positions[symbol]

        if request.side == pos.side:
            # Adding to position — compute weighted average entry
            total_cost = (pos.average_entry * pos.quantity) + (fill_price * fill_quantity)
            pos.quantity += fill_quantity
            pos.average_entry = total_cost / pos.quantity if pos.quantity > 0 else 0.0
        else:
            # Reducing/closing position
            if fill_quantity >= pos.quantity:
                # Position closed or flipped
                pnl_per_share = (
                    (fill_price - pos.average_entry)
                    if pos.side == Side.BUY
                    else (pos.average_entry - fill_price)
                )
                pos.realized_pnl += pnl_per_share * pos.quantity
                remaining = fill_quantity - pos.quantity

                if remaining > 0:
                    # Position flipped to opposite side
                    pos.side = request.side
                    pos.quantity = remaining
                    pos.average_entry = fill_price
                    pos.entry_time = now
                else:
                    # Position fully closed
                    pos.quantity = 0
                    pos.average_entry = 0.0
            else:
                # Partial reduction
                pnl_per_share = (
                    (fill_price - pos.average_entry)
                    if pos.side == Side.BUY
                    else (pos.average_entry - fill_price)
                )
                pos.realized_pnl += pnl_per_share * fill_quantity
                pos.quantity -= fill_quantity

    def _is_opening_trade(self, request: OrderRequest) -> bool:
        """Determine whether this trade opens or closes a position."""
        symbol = request.symbol
        if symbol not in self._positions:
            return True
        pos = self._positions[symbol]
        if pos.quantity == 0:
            return True
        # Opening if same side as current position, or no position
        return request.side == pos.side

    # ── Audit Log ────────────────────────────────────────────────

    def _log_action(self, action: str, details: dict[str, Any]) -> None:
        """Append an action to the audit log for journal replay."""
        self._action_log.append({
            "timestamp": datetime.now(IST).isoformat(),
            "action": action,
            **details,
        })

    def get_action_log(self) -> list[dict[str, Any]]:
        """Return the full audit log of paper broker actions.

        Used by the journal module for trade replay and analysis.

        Returns:
            List of action dicts in chronological order.
        """
        return list(self._action_log)

    # ── Utility Methods ──────────────────────────────────────────

    def register_instrument(self, info: InstrumentInfo) -> None:
        """Register an instrument for paper trading.

        Args:
            info: InstrumentInfo to register. Overwrites any existing
                entry for the same symbol.
        """
        self._instruments[info.symbol] = info
        logger.info("Registered instrument: %s", info.symbol)

    def set_quote(self, quote: Quote) -> None:
        """Set a simulated quote for a symbol.

        Args:
            quote: Quote snapshot to register.
        """
        self._fixture_quotes[quote.symbol] = quote

    def reset(self) -> None:
        """Reset all paper broker state to initial values.

        Clears orders, trades, positions, and action log.
        Restores starting margin.
        """
        self._margin_available = self._starting_margin
        self._margin_used = 0.0
        self._orders.clear()
        self._trades.clear()
        self._positions.clear()
        self._fixture_candles.clear()
        self._fixture_quotes.clear()
        self._action_log.clear()
        logger.info("PaperBroker reset to initial state")
