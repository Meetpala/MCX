"""Kite Connect broker adapter for APEX-3.

Integrates Zerodha Kite Connect SDK (kiteconnect>=5.0) for live market data,
quote fetching, order execution, position tracking, and margin queries.

All order-placement, modification, and cancellation methods require
the `_live_enabled` flag to be True — it defaults to False and should
ONLY be enabled after thorough testing in paper mode.

CRITICAL REQUIREMENTS FOR LIVE MODE:
- Static IP whitelisting configured in Kite Developer Console.
- Rate limits: ~10 orders/sec, ~1 quote/sec, ~3 historical/sec.
- Max 25 order modifications per order per session.
- Sessions expire daily; re-auth via login flow required each morning.
- Never log API credentials (api_key, access_token).
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from apex3.broker.interface import BrokerInterface
from apex3.broker.rate_limiter import RateLimiter, RateLimitTimeoutError
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


class LiveModeDisabledError(Exception):
    """Raised when an order-mutating method is called with live mode off.

    APEX-3 defaults to paper-mode only. Live trading must be
    explicitly enabled after thorough validation.
    """


# ── Conditional import of kiteconnect ────────────────────────────
_kiteconnect_available = False
_KiteConnect = None

try:
    from kiteconnect import KiteConnect as _KiteConnect  # type: ignore[no-redef]
    _kiteconnect_available = True
except ImportError:
    logger.info(
        "kiteconnect package not installed. Install via "
        "'pip install kiteconnect' to enable live Kite integration."
    )


def _map_kite_order_status(status_str: str) -> OrderStatus:
    """Map Zerodha Kite order status string to APEX-3 OrderStatus enum."""
    s = (status_str or "").upper().strip()
    mapping = {
        "COMPLETE": OrderStatus.COMPLETE,
        "REJECTED": OrderStatus.REJECTED,
        "CANCELLED": OrderStatus.CANCELLED,
        "OPEN": OrderStatus.OPEN,
        "TRIGGER PENDING": OrderStatus.TRIGGER_PENDING,
        "UPDATE": OrderStatus.OPEN,
        "PUT ORDER REQ RECEIVED": OrderStatus.PENDING,
        "VALIDATION PENDING": OrderStatus.PENDING,
        "OPEN PENDING": OrderStatus.PENDING,
        "MODIFY PENDING": OrderStatus.PENDING,
        "MODIFY VALIDATION PENDING": OrderStatus.PENDING,
        "CANCEL PENDING": OrderStatus.PENDING,
        "AMO REQ RECEIVED": OrderStatus.PENDING,
    }
    return mapping.get(s, OrderStatus.UNKNOWN)


def _map_kite_order_type(type_str: str) -> OrderType:
    """Map Kite order type string to APEX-3 OrderType enum."""
    t = (type_str or "").upper().strip()
    mapping = {
        "LIMIT": OrderType.LIMIT,
        "MARKET": OrderType.MARKET,
        "SL": OrderType.SL,
        "SL-M": OrderType.SL_M,
    }
    if t in mapping:
        return mapping[t]
    logger.warning("Unrecognized Kite order type '%s'; returning OrderType.UNKNOWN", type_str)
    return OrderType.UNKNOWN


MCX_LOT_MULTIPLIERS: dict[str, int] = {
    "SILVERMIC": 1,
    "GOLDGUINEA": 1,
    "GOLDPETAL": 1,
    "GOLDTEN": 1,
    "SILVERM": 5,
    "GOLDM": 10,
    "SILVER": 30,
    "GOLD": 100,
}


def get_mcx_lot_multiplier(symbol_or_root: str) -> int:
    """Return the P&L rupee multiplier per ₹1.00 move for MCX precious metals."""
    clean = symbol_or_root.upper().strip()
    for root in sorted(MCX_LOT_MULTIPLIERS.keys(), key=len, reverse=True):
        if clean.startswith(root):
            return MCX_LOT_MULTIPLIERS[root]
    return 1


def parse_expiry_date(expiry_val: Any) -> Optional[date]:
    """Parse expiry date from date, datetime, or standard exchange string formats."""
    if expiry_val is None:
        return None
    if isinstance(expiry_val, datetime):
        return expiry_val.date()
    if isinstance(expiry_val, date):
        return expiry_val
    if isinstance(expiry_val, str):
        s = expiry_val.strip()
        for fmt in ("%Y-%m-%d", "%d%b%Y", "%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%d%b%y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
        import re
        m = re.search(r"(\d{1,2}[A-Za-z]{3}\d{2,4})", s)
        if m:
            sub = m.group(1).upper()
            for fmt in ("%d%b%y", "%d%b%Y"):
                try:
                    return datetime.strptime(sub, fmt).date()
                except ValueError:
                    pass
    return None


def resolve_active_mcx_futures(
    instruments: list[dict[str, Any]],
    symbol_roots: list[str],
    min_days_to_expiry: int = 5,
    as_of: Optional[date] = None,
) -> dict[str, dict[str, Any]]:
    """Resolve the active front-month MCX futures contract for each symbol root.

    Filters contracts where (expiry_date - today).days >= min_days_to_expiry to avoid physical
    delivery tender periods, and selects the nearest active expiry.
    """
    today = as_of or datetime.now(IST).date()
    results: dict[str, dict[str, Any]] = {}

    for root in symbol_roots:
        clean_root = root.upper().strip()
        candidates: list[tuple[date, dict[str, Any]]] = []
        for inst in instruments:
            seg = str(inst.get("segment", "") or inst.get("exch_seg", "") or inst.get("exchange", "")).upper()
            if seg and "MCX" not in seg:
                continue

            itype = str(inst.get("instrument_type", "") or inst.get("instrumenttype", "")).upper()
            if itype and itype not in ("FUT", "FUTCOM"):
                continue

            inst_name = str(inst.get("name", "")).upper().strip()
            tradingsymbol = str(inst.get("tradingsymbol", "") or inst.get("symbol", "")).upper().strip()

            if tradingsymbol.endswith("CE") or tradingsymbol.endswith("PE"):
                continue
            if not tradingsymbol.endswith("FUT") and itype not in ("FUT", "FUTCOM"):
                continue

            matched = False
            if inst_name == clean_root:
                matched = True
            elif tradingsymbol.startswith(clean_root):
                longer_matches = [
                    k for k in MCX_LOT_MULTIPLIERS
                    if len(k) > len(clean_root) and tradingsymbol.startswith(k)
                ]
                if not longer_matches:
                    matched = True

            if matched:
                exp = parse_expiry_date(inst.get("expiry"))
                if exp is not None:
                    days_left = (exp - today).days
                    if days_left >= min_days_to_expiry:
                        candidates.append((exp, inst))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            results[clean_root] = candidates[0][1]
        else:
            logger.warning(
                "No active front-month contract found for root %s (min_days_to_expiry=%d)",
                clean_root,
                min_days_to_expiry,
            )

    return results


class KiteAdapter(BrokerInterface):
    """Zerodha Kite Connect broker adapter.

    Reads API credentials from environment variables and NEVER logs or persists them.
    All order-mutating operations require `_live_enabled=True`.

    Args:
        api_key: Kite API key. Defaults to KITE_API_KEY env var.
        access_token: Kite access token. Defaults to KITE_ACCESS_TOKEN env var.
        live_enabled: Whether live trading is enabled. Defaults to False.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        live_enabled: bool = False,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._api_key: str = api_key or os.environ.get("KITE_API_KEY", "")
        self._access_token: str = access_token or os.environ.get("KITE_ACCESS_TOKEN", "")
        self._explicit_token_passed: bool = access_token is not None
        self._live_enabled: bool = live_enabled
        self.rate_limiter: Optional[RateLimiter] = rate_limiter

        self._kite: Optional[Any] = None
        self._instruments_cache: dict[str, dict[str, dict]] = {}
        self._instrument_info_cache: dict[str, InstrumentInfo] = {}
        self._mis_margin_cache: dict[str, bool] = {}
        self._quote_permission_denied: bool = False
        self._cache_date: Optional[str] = None

        if _kiteconnect_available and _KiteConnect is not None:
            if self._api_key:
                self._kite = _KiteConnect(api_key=self._api_key)
                if self._access_token:
                    self._kite.set_access_token(self._access_token)
                logger.info("KiteAdapter initialized (live_enabled=%s)", self._live_enabled)
            else:
                logger.warning("KiteAdapter: No API key provided. Set KITE_API_KEY env var.")
        else:
            logger.info("KiteAdapter: kiteconnect library not installed.")

    def _acquire_rate_limit(self, category: str, timeout: float = 5.0) -> bool:
        """Acquire a token from the rate limiter if configured.

        Raises RateLimitTimeoutError on timeout to prevent unthrottled API bursts.
        """
        if self.rate_limiter:
            acquired = self.rate_limiter.acquire(category, timeout=timeout)
            if not acquired:
                err_msg = f"Rate limit acquire timed out for category '{category}' in KiteAdapter"
                logger.error(err_msg)
                raise RateLimitTimeoutError(err_msg)
            return True
        return True

    def _require_live(self) -> None:
        """Guard that raises if live mode is disabled."""
        if not self._live_enabled:
            raise LiveModeDisabledError(
                "Live trading is disabled. Set live_enabled=True only after "
                "thorough paper-mode validation. APEX-3 defaults to paper-mode."
            )

    def _require_kite(self) -> None:
        """Guard that raises if the kiteconnect client is not available."""
        if self._kite is None:
            raise RuntimeError(
                "Kite client not initialized. Ensure kiteconnect is installed "
                "and KITE_API_KEY / KITE_ACCESS_TOKEN env vars are set."
            )

    def _get_instruments(self, exchange: str = "NSE") -> dict[str, dict]:
        """Fetch and cache daily instruments dump from Kite."""
        # Backward compatibility for flat symbol dictionary assignments in unit tests
        if self._instruments_cache and not any(k in self._instruments_cache for k in ("NSE", "MCX", "NFO", "BSE")):
            return self._instruments_cache

        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        if exchange in self._instruments_cache and self._cache_date == today_str:
            return self._instruments_cache[exchange]

        self._require_kite()
        try:
            raw_list = self._kite.instruments(exchange)
            self._instruments_cache[exchange] = {i["tradingsymbol"]: i for i in raw_list}
            self._cache_date = today_str
            logger.info("Cached %d instruments from Kite (%s)", len(self._instruments_cache[exchange]), exchange)
        except Exception as e:
            logger.error("Failed to fetch instruments from Kite for %s: %s", exchange, e)
        return self._instruments_cache.get(exchange, {})

    def _resolve_instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Resolve instrument token for a given symbol."""
        cache = self._get_instruments(exchange)
        inst = cache.get(symbol)
        if inst:
            return int(inst["instrument_token"])
        raise ValueError(f"Instrument token not found for {exchange}:{symbol}")

    def resolve_active_mcx_futures(
        self,
        symbol_roots: list[str],
        min_days_to_expiry: int = 5,
        as_of: Optional[date] = None,
    ) -> dict[str, InstrumentInfo]:
        """Resolve active front-month MCX futures contracts for given symbol roots."""
        self._require_kite()
        try:
            instruments_list = self._kite.instruments("MCX")
        except Exception as e:
            logger.error("Failed to fetch MCX instruments: %s", e)
            instruments_list = list(self._get_instruments("MCX").values())

        active_dict = resolve_active_mcx_futures(instruments_list, symbol_roots, min_days_to_expiry, as_of)
        res: dict[str, InstrumentInfo] = {}
        for root, raw in active_dict.items():
            sym = raw.get("tradingsymbol", "")
            info = self.get_instrument_info(sym, exchange="MCX")
            if not info.symbol_root:
                info = InstrumentInfo(
                    symbol=info.symbol,
                    exchange=info.exchange,
                    symbol_root=root,
                    tick_size=info.tick_size,
                    lot_size=info.lot_size,
                    order_lot_size=info.order_lot_size,
                    lot_multiplier=info.lot_multiplier,
                    expiry_date=info.expiry_date,
                    days_to_expiry=info.days_to_expiry,
                    exchange_token=info.exchange_token,
                    instrument_token=info.instrument_token,
                    upper_circuit=info.upper_circuit,
                    lower_circuit=info.lower_circuit,
                    mis_available=info.mis_available,
                )
            res[root] = info
        return res

    # ── Margin & Capital ─────────────────────────────────────────

    def get_margins(self, segment: str = "commodity") -> MarginInfo:
        """Fetch margin information from Kite (defaults to commodity)."""
        self._require_kite()
        try:
            try:
                margins = self._kite.margins(segment=segment)
            except Exception as e:
                logger.warning("Kite margins failed for segment '%s': %s; falling back to 'equity'", segment, e)
                margins = self._kite.margins(segment="equity")

            avail_data = margins.get("available", {})
            util_data = margins.get("utilised", {})

            # live_balance or net
            live_balance = float(avail_data.get("live_balance", margins.get("net", 0.0)))
            equity_used = float(util_data.get("debits", 0.0))
            collateral = float(avail_data.get("collateral", 0.0))
            net_value = float(margins.get("net", live_balance + equity_used))

            return MarginInfo(
                equity_available=live_balance,
                equity_used=equity_used,
                collateral=collateral,
                net_value=net_value,
                timestamp=datetime.now(IST),
            )
        except Exception as e:
            logger.error("Failed to fetch Kite margins: %s", e)
            raise

    def estimate_order_margin(self, request: OrderRequest) -> OrderMarginEstimate:
        """Estimate margin via Kite's order margin calculator."""
        self._require_kite()
        self._acquire_rate_limit("orders")
        exchange = request.exchange or "MCX"
        if exchange == "NSE" and any(k in request.symbol for k in ("GOLD", "SILVER", "COPPER", "ZINC", "CRUDE", "NATURALGAS")):
            exchange = "MCX"
        order_type_str = request.order_type.value
        params = [{
            "exchange": exchange,
            "tradingsymbol": request.symbol,
            "transaction_type": request.side.value,
            "variety": "regular",
            "product": request.product.value,
            "order_type": order_type_str,
            "quantity": request.quantity,
            "price": request.price if order_type_str in ("LIMIT", "SL") else 0.0,
            "trigger_price": request.trigger_price if order_type_str in ("SL", "SL-M") else 0.0,
        }]

        lot_mult = get_mcx_lot_multiplier(request.symbol)
        notional = request.price * lot_mult * request.quantity

        try:
            result = self._kite.order_margins(params)
            res = result[0] if result else {}
            req_m = float(res.get("total", 0.0))
            charges = float(res.get("charges", {}).get("total", 0.0))
            leverage = (notional / req_m) if req_m > 0 else 6.0
            return OrderMarginEstimate(
                required_margin=req_m,
                charges=charges,
                total=req_m + charges,
                leverage_available=min(10.0, round(leverage, 2)),
            )
        except Exception as e:
            logger.error("Order margin estimation failed for %s: %s", request.symbol, e)
            if getattr(self, "_live_enabled", False):
                raise RuntimeError(
                    f"Fresh margin estimate failed for {request.symbol}: {e}. "
                    "Cannot safely assume fallback leverage in live trading."
                ) from e
            est_req = notional * 0.15
            return OrderMarginEstimate(
                required_margin=est_req,
                charges=0.0,
                total=est_req,
                leverage_available=round(notional / est_req, 2) if est_req > 0 else 6.67,
            )

    # ── Order Management ─────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> str:
        """Place an order via Kite Connect."""
        self._require_live()
        self._require_kite()
        self._acquire_rate_limit("orders")

        order_type_str = request.order_type.value
        price = request.price if order_type_str in ("LIMIT", "SL") else None
        trigger_price = request.trigger_price if order_type_str in ("SL", "SL-M") else None
        tag = request.tag[:20] if request.tag else None

        logger.info(
            "Placing LIVE Kite order: %s %s %d @ %s (Type: %s, Trig: %s)",
            request.side.value,
            request.symbol,
            request.quantity,
            price,
            order_type_str,
            trigger_price,
        )

        exchange = request.exchange or "MCX"
        if exchange == "NSE" and any(k in request.symbol for k in ("GOLD", "SILVER", "COPPER", "ZINC", "CRUDE", "NATURALGAS")):
            exchange = "MCX"

        order_id = self._kite.place_order(
            variety="regular",
            exchange=exchange,
            tradingsymbol=request.symbol,
            transaction_type=request.side.value,
            quantity=request.quantity,
            product=request.product.value,
            order_type=order_type_str,
            price=price,
            trigger_price=trigger_price,
            tag=tag,
        )
        return str(order_id)

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> bool:
        """Modify an order via Kite Connect."""
        self._require_live()
        self._require_kite()
        self._acquire_rate_limit("orders")

        params: dict[str, Any] = {}
        if quantity is not None:
            params["quantity"] = quantity
        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if order_type is not None:
            params["order_type"] = order_type

        logger.info("Modifying LIVE Kite order %s: %s", order_id, params)
        self._kite.modify_order(
            variety="regular",
            order_id=order_id,
            **params,
        )
        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order via Kite Connect."""
        self._require_live()
        self._require_kite()
        self._acquire_rate_limit("orders")

        logger.info("Cancelling LIVE Kite order %s", order_id)
        self._kite.cancel_order(variety="regular", order_id=order_id)
        return True

    # ── Order & Trade Queries (read-only) ────────────────────────

    def get_order_status(self, order_id: str) -> OrderState:
        """Fetch order status from Kite."""
        self._require_kite()
        self._acquire_rate_limit("orders")
        history = self._kite.order_history(order_id)
        if not history:
            raise ValueError(f"No history returned for order {order_id}")
        latest = history[-1]

        order_req = OrderRequest(
            symbol=latest.get("tradingsymbol", ""),
            exchange=latest.get("exchange", "NSE"),
            side=getattr(Side, str(latest.get("transaction_type", "BUY")).upper().strip(), Side.BUY),
            quantity=int(latest.get("quantity", 0)),
            product=getattr(Product, str(latest.get("product", "MIS")).upper().strip(), Product.UNKNOWN),
            order_type=_map_kite_order_type(latest.get("order_type", "LIMIT")),
            price=float(latest.get("price", 0.0)),
            trigger_price=float(latest.get("trigger_price", 0.0)),
            tag=latest.get("tag", ""),
        )

        return OrderState(
            order_id=order_id,
            request=order_req,
            status=_map_kite_order_status(latest.get("status", "")),
            filled_quantity=int(latest.get("filled_quantity", 0)),
            pending_quantity=int(latest.get("pending_quantity", 0)),
            average_price=float(latest.get("average_price", 0.0)),
            modification_count=max(0, len(history) - 1),
            last_updated=latest.get("order_timestamp", datetime.now(IST)),
            status_message=latest.get("status_message", ""),
            is_strategy_owned=True,
        )

    def get_orders(self) -> list[OrderState]:
        """Fetch all orders for the current session from Kite."""
        self._require_kite()
        self._acquire_rate_limit("orders")
        raw_orders = self._kite.orders()
        states = []
        for o in raw_orders:
            order_id = str(o.get("order_id", ""))
            order_req = OrderRequest(
                symbol=o.get("tradingsymbol", ""),
                exchange=o.get("exchange", "NSE"),
                side=getattr(Side, str(o.get("transaction_type", "BUY")).upper().strip(), Side.BUY),
                quantity=int(o.get("quantity", 0)),
                product=getattr(Product, str(o.get("product", "MIS")).upper().strip(), Product.UNKNOWN),
                order_type=_map_kite_order_type(o.get("order_type", "LIMIT")),
                price=float(o.get("price", 0.0)),
                trigger_price=float(o.get("trigger_price", 0.0)),
                tag=o.get("tag", ""),
            )
            states.append(
                OrderState(
                    order_id=order_id,
                    request=order_req,
                    status=_map_kite_order_status(o.get("status", "")),
                    filled_quantity=int(o.get("filled_quantity", 0)),
                    pending_quantity=int(o.get("pending_quantity", 0)),
                    average_price=float(o.get("average_price", 0.0)),
                    last_updated=o.get("order_timestamp", datetime.now(IST)),
                    status_message=o.get("status_message", ""),
                    is_strategy_owned=True,
                )
            )
        return states

    def get_trades(self) -> list[dict]:
        """Fetch all executed trades from Kite."""
        self._require_kite()
        self._acquire_rate_limit("orders")
        return self._kite.trades()

    # ── Positions ────────────────────────────────────────────────

    def get_positions(self) -> list[PositionState]:
        """Fetch current positions from Kite."""
        self._require_kite()
        self._acquire_rate_limit("orders")
        raw_pos = self._kite.positions()
        net_pos = raw_pos.get("net", [])

        positions = []
        for p in net_pos:
            qty = int(p.get("quantity", 0))
            if qty == 0:
                continue

            side = Side.BUY if qty > 0 else Side.SELL
            positions.append(
                PositionState(
                    symbol=p.get("tradingsymbol", ""),
                    side=side,
                    quantity=abs(qty),
                    average_entry=float(p.get("average_price", 0.0)),
                    current_price=float(p.get("last_price", 0.0)),
                    unrealized_pnl=float(p.get("unrealised", 0.0)),
                    realized_pnl=float(p.get("realised", 0.0)),
                    product=getattr(Product, str(p.get("product", "MIS")).upper().strip(), Product.UNKNOWN),
                )
            )
        return positions

    # ── Market Data ──────────────────────────────────────────────

    def _parse_kite_quote(self, symbol: str, q: dict, exchange: str = "NSE") -> Quote:
        """Parse raw Kite quote dictionary into Quote with strict freshness checks."""
        depth = q.get("depth", {})
        buys = depth.get("buy", [])
        sells = depth.get("sell", [])

        has_depth = bool(buys and sells and buys[0].get("price", 0) > 0 and sells[0].get("price", 0) > 0)
        last_price = float(q.get("last_price", 0.0))

        if has_depth:
            bid = float(buys[0]["price"])
            ask = float(sells[0]["price"])
        else:
            bid = 0.0
            ask = 0.0

        source_ts = q.get("timestamp") or q.get("last_trade_time")
        quote_ts = None
        has_valid_ts = False
        if isinstance(source_ts, datetime):
            quote_ts = source_ts.astimezone(IST) if source_ts.tzinfo else source_ts.replace(tzinfo=IST)
            has_valid_ts = True
        elif isinstance(source_ts, str):
            try:
                quote_ts = datetime.fromisoformat(source_ts).astimezone(IST)
                has_valid_ts = True
            except Exception:
                quote_ts = None

        now_ist = datetime.now(IST)
        is_index = symbol.upper() in ("NIFTY 50", "NIFTY", "NSEI", "^NSEI", "NIFTY BANK", "BANKNIFTY") or "NIFTY" in symbol.upper()

        if has_valid_ts and quote_ts is not None:
            age_sec = (now_ist - quote_ts).total_seconds()
            stale_limit = getattr(self.config.schedule, "stale_threshold_sec", 5) if hasattr(self, "config") and hasattr(self.config, "schedule") else 5
            is_time_fresh = (-15.0 <= age_sec <= max(stale_limit, 15.0))
            if is_index:
                # Indices do not have depth on the exchange
                is_fresh = bool(last_price > 0.0 and is_time_fresh)
            else:
                is_fresh = bool(has_depth and last_price > 0.0 and is_time_fresh)
        else:
            quote_ts = now_ist
            is_fresh = False

        return Quote(
            symbol=symbol,
            exchange=exchange,
            last_price=last_price,
            bid=bid,
            ask=ask,
            volume=int(q.get("volume", 0)),
            timestamp=quote_ts,
            is_fresh=is_fresh,
        )

    def get_quotes_batch(self, symbols: list[str], exchange: str = "NSE") -> dict[str, Quote]:
        """Fetch live quote snapshots for multiple symbols in a single Kite API call."""
        self._require_kite()
        self._acquire_rate_limit("quotes")
        keys = [f"{exchange}:{s}" for s in symbols]
        try:
            raw_data = self._kite.quote(keys)
        except Exception as e:
            logger.error("Kite batch quote error for %s: %s", symbols, e)
            now = datetime.now(IST)
            return {
                s: Quote(
                    symbol=s,
                    exchange=exchange,
                    last_price=0.0,
                    bid=0.0,
                    ask=0.0,
                    volume=0,
                    timestamp=now,
                    is_fresh=False,
                )
                for s in symbols
            }

        quotes: dict[str, Quote] = {}
        for s in symbols:
            key = f"{exchange}:{s}"
            q_dict = raw_data.get(key, {})
            if q_dict:
                quotes[s] = self._parse_kite_quote(s, q_dict, exchange=exchange)
            else:
                quotes[s] = Quote(
                    symbol=s,
                    exchange=exchange,
                    last_price=0.0,
                    bid=0.0,
                    ask=0.0,
                    volume=0,
                    timestamp=datetime.now(IST),
                    is_fresh=False,
                )
        return quotes

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Quote:
        """Fetch a live quote snapshot from Kite with verified source timestamp and depth."""
        res = self.get_quotes_batch([symbol], exchange=exchange)
        return res.get(symbol, Quote(
            symbol=symbol,
            exchange=exchange,
            last_price=0.0,
            bid=0.0,
            ask=0.0,
            volume=0,
            timestamp=datetime.now(IST),
            is_fresh=False,
        ))

    def get_historical(
        self,
        symbol: str,
        interval: str,
        from_dt: datetime,
        to_dt: datetime,
        exchange: str = "NSE",
    ) -> list[Candle]:
        """Fetch historical candle data from Kite."""
        self._require_kite()
        self._acquire_rate_limit("historical")
        token = self._resolve_instrument_token(symbol, exchange)

        # Normalize interval strings ("5min" -> "5minute", "1min" -> "minute")
        norm_interval = interval
        if interval in ("5min", "5m"):
            norm_interval = "5minute"
        elif interval in ("1min", "1m"):
            norm_interval = "minute"

        if from_dt is None:
            from_dt = datetime.now(IST) - timedelta(days=5)
        if to_dt is None:
            to_dt = datetime.now(IST)

        raw_bars = self._kite.historical_data(
            instrument_token=token,
            from_date=from_dt,
            to_date=to_dt,
            interval=norm_interval,
        )

        candles = []
        now_ist = datetime.now(IST)
        tf_minutes = 5 if "5" in interval else (1 if "minute" in interval else 5)
        for row in raw_bars:
            row_dt = row["date"] if isinstance(row["date"], datetime) else datetime.fromisoformat(str(row["date"]))
            row_dt = row_dt.astimezone(IST) if row_dt.tzinfo else row_dt.replace(tzinfo=IST)
            is_done = now_ist >= (row_dt + timedelta(minutes=tf_minutes))
            candles.append(
                Candle(
                    timestamp=row_dt,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    is_complete=is_done,
                )
            )
        return candles

    def get_instrument_info(self, symbol: str, exchange: Optional[str] = None) -> InstrumentInfo:
        """Fetch instrument metadata and circuit boundaries from Kite."""
        if exchange is None:
            exchange = "MCX" if symbol.upper().startswith(("SILVER", "GOLD")) else "NSE"
        if symbol in self._instrument_info_cache:
            return self._instrument_info_cache[symbol]

        self._require_kite()
        inst = self._get_instruments(exchange).get(symbol)
        if inst is None and exchange == "MCX":
            try:
                resolved_map = self.resolve_active_mcx_futures([symbol])
                if symbol in resolved_map:
                    dated_sym = resolved_map[symbol]
                    inst = self._get_instruments(exchange).get(dated_sym)
                    if inst:
                        symbol = dated_sym
            except Exception as res_err:
                logger.debug("Could not resolve bare MCX symbol %s to dated contract: %s", symbol, res_err)

        default_tick = 1.0 if exchange == "MCX" else 0.05
        tick = float(inst.get("tick_size", default_tick)) if inst else default_tick
        order_lot = int(inst.get("lot_size", 1)) if inst else 1
        lot_mult = get_mcx_lot_multiplier(symbol)
        exp_date = parse_expiry_date(inst.get("expiry")) if inst else None
        today = datetime.now(IST).date()
        dte = (exp_date - today).days if exp_date else 999

        # Get circuit limits from live quote with rate limiting
        q_data = {}
        upper_c = 0.0
        lower_c = 0.0
        if not self._quote_permission_denied:
            try:
                self._acquire_rate_limit("quotes")
                q_data = self._kite.quote(f"{exchange}:{symbol}").get(f"{exchange}:{symbol}", {})
                upper_c = float(q_data.get("upper_circuit_limit", 0.0))
                lower_c = float(q_data.get("lower_circuit_limit", 0.0))
            except Exception as qe:
                if "insufficient permission" in str(qe).lower():
                    self._quote_permission_denied = True
                    logger.info(
                        "Kite quote API permission not enabled on this plan (%s); "
                        "skipping Kite quote calls in get_instrument_info (Hybrid/Angel feed provides circuits).",
                        qe,
                    )
                else:
                    logger.warning("Could not fetch live quote/circuits for %s: %s", symbol, qe)
                upper_c = 0.0
                lower_c = 0.0

        # Do not fabricate synthetic +-10% circuit limits; report 0.0 if not verified from exchange
        if upper_c <= 0.0:
            upper_c = 0.0
        if lower_c <= 0.0:
            lower_c = 0.0

        token_str = str(inst.get("instrument_token", "")) if inst else ""

        # Validate broker-side margin eligibility
        mis_eligible = False
        if exchange == "MCX":
            mis_eligible = True
        elif inst and inst.get("segment") == "NSE" and inst.get("instrument_type") == "EQ":
            clean_sym = symbol.upper()
            # SEBI / NSE restricted surveillance or trade-to-trade series prohibit intraday MIS
            if not any(clean_sym.endswith(sfx) for sfx in ("-BE", "-BZ", "-SM", "-ST", "-E1", "-SZ", "-IL")):
                if self._kite and self._access_token:
                    try:
                        if symbol in self._mis_margin_cache:
                            mis_eligible = self._mis_margin_cache[symbol]
                        else:
                            self._acquire_rate_limit("orders")
                            ref_price = float(q_data.get("last_price", 100.0) or 100.0)
                            if ref_price <= 0:
                                ref_price = 100.0
                            # Authoritatively verify both BUY and SELL (short intraday) orders
                            m_res = self._kite.order_margins([
                                {
                                    "exchange": exchange,
                                    "tradingsymbol": symbol,
                                    "transaction_type": "BUY",
                                    "variety": "regular",
                                    "product": "MIS",
                                    "order_type": "LIMIT",
                                    "quantity": 1,
                                    "price": ref_price,
                                },
                                {
                                    "exchange": exchange,
                                    "tradingsymbol": symbol,
                                    "transaction_type": "SELL",
                                    "variety": "regular",
                                    "product": "MIS",
                                    "order_type": "LIMIT",
                                    "quantity": 1,
                                    "price": ref_price,
                                },
                            ])
                            if (
                                m_res
                                and isinstance(m_res, list)
                                and len(m_res) >= 2
                                and m_res[0].get("total", 0) > 0
                                and m_res[1].get("total", 0) > 0
                                and "error" not in m_res[0]
                                and "error" not in m_res[1]
                            ):
                                mis_eligible = True
                            else:
                                mis_eligible = False
                            self._mis_margin_cache[symbol] = mis_eligible
                    except Exception as me:
                        logger.warning("Broker MIS margin verification failed for %s, failing closed: %s", symbol, me)
                        mis_eligible = False
                        self._mis_margin_cache[symbol] = False
                else:
                    mis_eligible = True

        sym_root = ""
        for r in sorted(MCX_LOT_MULTIPLIERS.keys(), key=len, reverse=True):
            if symbol.upper().startswith(r):
                sym_root = r
                break

        info = InstrumentInfo(
            symbol=symbol,
            exchange=exchange,
            symbol_root=sym_root,
            upper_circuit=upper_c,
            lower_circuit=lower_c,
            tick_size=tick,
            lot_size=order_lot,
            order_lot_size=order_lot,
            lot_multiplier=lot_mult,
            expiry_date=exp_date,
            days_to_expiry=dte,
            instrument_token=token_str,
            mis_available=mis_eligible,
        )
        self._instrument_info_cache[symbol] = info
        return info

    # ── Session ──────────────────────────────────────────────────

    def is_session_valid(self) -> bool:
        """Check if the Kite session token is active and valid, with .env hot-reload support."""
        if self._kite is None:
            return False
        if self._access_token:
            try:
                profile = self._kite.profile()
                if bool(profile and profile.get("user_id")):
                    return True
            except Exception:
                pass

        # If running in cloud/24x7 mode and token was refreshed in .env, hot-reload it
        if self._api_key and self._api_key == os.environ.get("KITE_API_KEY", "").strip():
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                new_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
                if new_token and new_token != self._access_token:
                    self._access_token = new_token
                    self._kite.set_access_token(new_token)
                    self._instrument_info_cache.clear()
                    self._mis_margin_cache.clear()
                    profile = self._kite.profile()
                    if bool(profile and profile.get("user_id")):
                        logger.info("Hot-reloaded refreshed KITE_ACCESS_TOKEN from .env successfully.")
                        return True
            except Exception:
                pass
        return False
