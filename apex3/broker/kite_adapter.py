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
        self._instruments_cache: dict[str, dict] = {}
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
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        if self._instruments_cache and self._cache_date == today_str:
            return self._instruments_cache

        self._require_kite()
        try:
            raw_list = self._kite.instruments(exchange)
            self._instruments_cache = {i["tradingsymbol"]: i for i in raw_list}
            self._cache_date = today_str
            logger.info("Cached %d instruments from Kite (%s)", len(self._instruments_cache), exchange)
        except Exception as e:
            logger.error("Failed to fetch instruments from Kite: %s", e)
        return self._instruments_cache

    def _resolve_instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Resolve instrument token for a given symbol."""
        cache = self._get_instruments(exchange)
        inst = cache.get(symbol)
        if inst:
            return int(inst["instrument_token"])
        raise ValueError(f"Instrument token not found for {exchange}:{symbol}")

    # ── Margin & Capital ─────────────────────────────────────────

    def get_margins(self) -> MarginInfo:
        """Fetch margin information from Kite."""
        self._require_kite()
        try:
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
        order_type_str = request.order_type.value
        params = [{
            "exchange": request.exchange,
            "tradingsymbol": request.symbol,
            "transaction_type": request.side.value,
            "variety": "regular",
            "product": request.product.value,
            "order_type": order_type_str,
            "quantity": request.quantity,
            "price": request.price if order_type_str in ("LIMIT", "SL") else 0.0,
            "trigger_price": request.trigger_price if order_type_str in ("SL", "SL-M") else 0.0,
        }]
        try:
            result = self._kite.order_margins(params)
            res = result[0] if result else {}
            req_m = float(res.get("total", 0.0))
            charges = float(res.get("charges", {}).get("total", 0.0))
            notional = request.price * request.quantity
            leverage = (notional / req_m) if req_m > 0 else 4.0
            return OrderMarginEstimate(
                required_margin=req_m,
                charges=charges,
                total=req_m + charges,
                leverage_available=min(4.0, round(leverage, 2)),
            )
        except Exception as e:
            logger.error("Order margin estimation failed for %s: %s", request.symbol, e)
            if getattr(self, "_live_enabled", False):
                raise RuntimeError(
                    f"Fresh margin estimate failed for {request.symbol}: {e}. "
                    "Cannot safely assume 4x leverage in live trading."
                ) from e
            notional = request.price * request.quantity
            est_req = notional / 4.0
            return OrderMarginEstimate(
                required_margin=est_req,
                charges=0.0,
                total=est_req,
                leverage_available=4.0,
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

        order_id = self._kite.place_order(
            variety="regular",
            exchange=request.exchange,
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

    def get_instrument_info(self, symbol: str, exchange: str = "NSE") -> InstrumentInfo:
        """Fetch instrument metadata and circuit boundaries from Kite."""
        if symbol in self._instrument_info_cache:
            return self._instrument_info_cache[symbol]

        self._require_kite()
        inst = self._get_instruments(exchange).get(symbol)
        tick = float(inst.get("tick_size", 0.05)) if inst else 0.05

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

        # Validate actual broker-side MIS eligibility (segment, equity series, and margin rules)
        # Defaults to False (fail-closed)
        mis_eligible = False
        if inst and inst.get("segment") == "NSE" and inst.get("instrument_type") == "EQ":
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

        info = InstrumentInfo(
            symbol=symbol,
            exchange=exchange,
            upper_circuit=upper_c,
            lower_circuit=lower_c,
            tick_size=tick,
            lot_size=int(inst.get("lot_size", 1)) if inst else 1,
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
