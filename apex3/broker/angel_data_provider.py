"""Angel One SmartAPI Market Data Provider for APEX-3.

STRICTLY READ-ONLY:
This provider is exclusively designed for streaming real-time quotes, 
5-minute candles, and market depth from Angel One's 100% free SmartAPI.
It does NOT contain or expose any order execution, modification, or cancellation
capabilities. All trading orders are routed exclusively through the execution broker (Zerodha).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from apex3.broker.rate_limiter import RateLimiter
from apex3.models import Candle, InstrumentInfo, Quote

logger = logging.getLogger(__name__)
try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    from datetime import timezone
    IST = timezone(timedelta(hours=5, minutes=30))

# Static token mapping for top liquid NIFTY candidates (cached for sub-millisecond lookup)
DEFAULT_ANGEL_TOKENS: dict[str, str] = {
    # Audited Elite 10 Institutional Watchlist
    "HINDALCO": "1363",
    "JSWSTEEL": "11723",
    "RELIANCE": "2885",
    "MARUTI": "10999",
    "ITC": "1660",
    "TCS": "11536",
    "SBIN": "3045",
    "TRENT": "1964",
    "KOTAKBANK": "1922",
    "SUNPHARMA": "3351",
    # Tier 2 & Extended Radar Symbols
    "BHARTIARTL": "10604",
    "JIOFIN": "18143",
    "TATASTEEL": "3499",
    "AXISBANK": "5900",
    "TATAPOWER": "3426",
    # Benchmark & High Liquidity Proxies
    "NIFTY 50": "99926000",
    "NIFTY": "99926000",
    "NSEI": "99926000",
    "^NSEI": "99926000",
    "INFY": "1594",
    "HDFCBANK": "1333",
    "ICICIBANK": "4963",
    "LT": "11483",
    "BAJFINANCE": "317",
    "HINDUNILVR": "1394",
    # MCX Precious Metals Watchlist Defaults
    "SILVERMIC": "256265",
    "GOLDGUINEA": "256266",
    "SILVERM": "256267",
    "GOLDM": "256268",
    "SILVER": "256269",
    "GOLD": "256270",
}


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
    scrip_list: list[dict[str, Any]],
    symbol_roots: list[str],
    min_days_to_expiry: int = 5,
    as_of: Optional[date] = None,
) -> dict[str, dict[str, Any]]:
    """Resolve the active front-month MCX futures contract for each symbol root from Angel One scrip records.

    Filters contracts where exch_seg == 'MCX' and instrumenttype in ('FUTCOM', 'FUT'), where
    (expiry - as_of).days >= min_days_to_expiry, and selects the nearest active expiry.
    """
    today = as_of or datetime.now(IST).date()
    results: dict[str, dict[str, Any]] = {}

    for root in symbol_roots:
        clean_root = root.upper().strip()
        candidates: list[tuple[date, dict[str, Any]]] = []
        for item in scrip_list:
            seg = str(item.get("exch_seg", "") or item.get("segment", "") or item.get("exchange", "")).upper()
            if seg and "MCX" not in seg:
                continue

            itype = str(item.get("instrumenttype", "") or item.get("instrument_type", "")).upper()
            if itype and itype not in ("FUTCOM", "FUT"):
                continue

            name = str(item.get("name", "")).upper().strip()
            symbol = str(item.get("symbol", "") or item.get("tradingsymbol", "")).upper().strip()

            if symbol.endswith("CE") or symbol.endswith("PE"):
                continue
            if not symbol.endswith("FUT") and itype not in ("FUTCOM", "FUT"):
                continue

            matched = False
            if name == clean_root:
                matched = True
            elif symbol.startswith(clean_root):
                longer_matches = [
                    k for k in MCX_LOT_MULTIPLIERS
                    if len(k) > len(clean_root) and symbol.startswith(k)
                ]
                if not longer_matches:
                    matched = True

            if matched:
                exp = parse_expiry_date(item.get("expiry") or item.get("tradingsymbol") or item.get("symbol"))
                if exp is not None:
                    days_left = (exp - today).days
                    if days_left >= min_days_to_expiry:
                        candidates.append((exp, item))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            results[clean_root] = candidates[0][1]
        else:
            logger.warning(
                "No active Angel One front-month MCX contract found for %s (min_days_to_expiry=%d)",
                clean_root,
                min_days_to_expiry,
            )

    return results


class AngelDataProvider:
    """Read-only market data provider powered by Angel One SmartAPI."""

    def __init__(
        self,
        api_key: str = "",
        client_code: str = "",
        pin: str = "",
        totp_secret: str = "",
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self.api_key = api_key
        self.client_code = client_code
        self.pin = pin
        self.totp_secret = totp_secret
        self._rate_limiter = rate_limiter

        self._smart_api: Optional[Any] = None
        self._auth_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._feed_token: Optional[str] = None
        self._token_cache: dict[str, str] = dict(DEFAULT_ANGEL_TOKENS)
        self._circuit_cache: dict[str, tuple[float, float]] = {}
        self._candle_cache: dict[tuple[str, str], tuple[float, int, list[Candle]]] = {}
        self._candle_cache_ttl: float = 120.0
        self._error_backoff: dict[tuple[str, str], float] = {}
        self._last_reauth_mono: float = 0.0

        if api_key and client_code and pin and totp_secret:
            self.authenticate()

    def authenticate(self, force: bool = False) -> bool:
        """Authenticate with Angel One SmartAPI using TOTP (idempotent with rate-limit backoff)."""
        if self._auth_token and self._smart_api and not force:
            return True
        try:
            from SmartApi import SmartConnect
            import pyotp

            for attempt in range(3):
                try:
                    logger.info("Authenticating with Angel One SmartAPI (Client: %s)...", self.client_code)
                    orig_cwd = os.getcwd()
                    try:
                        test_dir = os.path.join(orig_cwd, "logs", time.strftime("%Y-%m-%d"))
                        os.makedirs(test_dir, exist_ok=True)
                    except Exception:
                        import tempfile
                        os.chdir(tempfile.gettempdir())
                    try:
                        self._smart_api = SmartConnect(api_key=self.api_key)
                    finally:
                        try:
                            os.chdir(orig_cwd)
                        except Exception:
                            pass

                    totp_val = pyotp.TOTP(self.totp_secret).now()
                    session_data = self._smart_api.generateSession(self.client_code, self.pin, totp_val)

                    if not session_data or not session_data.get("status"):
                        msg = session_data.get("message", "Unknown error") if session_data else "Empty response"
                        if "exceeding access rate" in str(msg).lower() and attempt < 2:
                            time.sleep(3.0 * (attempt + 1))
                            continue
                        logger.error("Angel One session generation failed: %s", msg)
                        return False

                    data = session_data.get("data", {})
                    self._auth_token = data.get("jwtToken")
                    self._refresh_token = data.get("refreshToken")
                    self._feed_token = self._smart_api.getfeedToken()
                    logger.info("Angel One SmartAPI authentication SUCCESSFUL! Free market data feed active.")
                    return True
                except Exception as inner_e:
                    if "exceeding access rate" in str(inner_e).lower() and attempt < 2:
                        logger.warning("Angel One login rate-limited (attempt %d/3); retrying in %.1fs...", attempt + 1, 3.0 * (attempt + 1))
                        time.sleep(3.0 * (attempt + 1))
                        continue
                    raise inner_e
            return False
        except ImportError as ie:
            logger.warning("smartapi-python or pyotp import failed (%s). Angel data feed disabled.", ie)
            return False
        except Exception as e:
            logger.error("Failed to connect to Angel One SmartAPI: %s", e)
            return False

    def is_session_valid(self) -> bool:
        """Verify that Angel One SmartAPI session is active, auto-refreshing via TOTP after daily reset."""
        if not self._smart_api or not self._auth_token:
            if self.api_key and self.client_code and self.pin and self.totp_secret:
                now_m = time.monotonic()
                if now_m - self._last_reauth_mono > 60.0:
                    self._last_reauth_mono = now_m
                    return self.authenticate(force=True)
            return False
        try:
            if hasattr(self._smart_api, "getProfile"):
                prof = self._smart_api.getProfile(self._refresh_token)
                if prof and prof.get("status") is False:
                    if self.api_key and self.client_code and self.pin and self.totp_secret:
                        now_m = time.monotonic()
                        if now_m - self._last_reauth_mono > 60.0:
                            logger.info("Angel One daily token expired; auto-refreshing session via TOTP...")
                            self._last_reauth_mono = now_m
                            return self.authenticate(force=True)
                    return False
            return True
        except Exception as e:
            logger.warning("Angel One session validity check failed: %s", e)
            return False

    def resolve_token(self, symbol: str, exchange: str = "NSE") -> str:
        """Resolve exchange symbol token for Angel One. Fails closed on unmapped symbols."""
        clean = symbol.replace("-EQ", "").upper()
        if exchange != "MCX" and clean.startswith(("SILVER", "GOLD")):
            exchange = "MCX"

        # For equity, cached tokens are static; for MCX futures, dynamically resolve front-month
        if exchange != "MCX" and clean in self._token_cache:
            return self._token_cache[clean]

        if self._smart_api and self._auth_token and hasattr(self._smart_api, "searchScrip"):
            try:
                res = self._smart_api.searchScrip(exchange, clean)
                if isinstance(res, dict) and res.get("status") and isinstance(res.get("data"), list):
                    if exchange == "MCX":
                        fut_matches = [
                            item for item in res["data"]
                            if isinstance(item, dict)
                            and (str(item.get("tradingsymbol", "")).upper() == clean
                                 or (str(item.get("tradingsymbol", "")).upper().startswith(clean)
                                     and str(item.get("tradingsymbol", "")).upper().endswith("FUT")))
                        ]
                        if fut_matches:
                            exact = [m for m in fut_matches if str(m.get("tradingsymbol", "")).upper() == clean]
                            target_match = exact[0] if exact else fut_matches[0]
                            tok = str(target_match.get("symboltoken", "")).strip()
                            if tok:
                                self._token_cache[clean] = tok
                                return tok
                    else:
                        target_ts = f"{clean}-EQ"
                        for item in res["data"]:
                            if isinstance(item, dict) and str(item.get("tradingsymbol", "")).upper() == target_ts:
                                tok = str(item.get("symboltoken", "")).strip()
                                if tok:
                                    self._token_cache[clean] = tok
                                    return tok
            except Exception as e:
                logger.debug("Angel One searchScrip lookup failed for %s: %s", clean, e)

        if clean in self._token_cache:
            return self._token_cache[clean]

        logger.error("Unknown Angel One token mapping for symbol %s in %s (no default fallback)", clean, exchange)
        return ""

    def _parse_market_item(self, item: dict, symbol: str, exchange: str) -> Quote:
        """Parse raw SmartAPI getMarketData item into Quote with strict freshness checks."""
        clean = symbol.replace("-EQ", "").upper()
        token = str(item.get("symbolToken", ""))
        is_index = clean in ("NIFTY 50", "NIFTY", "NSEI", "^NSEI", "NIFTY BANK", "BANKNIFTY") or token == "99926000"

        ltp = float(item.get("ltp", 0.0) or 0.0)
        depth = item.get("depth", {})
        buy_levels = depth.get("buy", []) if isinstance(depth, dict) else []
        sell_levels = depth.get("sell", []) if isinstance(depth, dict) else []

        best_bid = float(buy_levels[0].get("price", 0.0) or 0.0) if buy_levels else 0.0
        best_ask = float(sell_levels[0].get("price", 0.0) or 0.0) if sell_levels else 0.0
        vol = int(item.get("tradeVolume", item.get("volume", 0)) or 0)

        # Cache exchange circuit limits
        up_c = float(item.get("upperCircuit", 0.0) or 0.0)
        low_c = float(item.get("lowerCircuit", 0.0) or 0.0)
        if up_c > 0 and low_c > 0:
            self._circuit_cache[clean] = (low_c, up_c)

        # Parse exchange feed timestamp (FAIL CLOSED IF MISSING — do NOT invent from local time)
        feed_str = item.get("exchFeedTime") or item.get("exchTradeTime")
        feed_dt = None
        if feed_str:
            for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
                try:
                    feed_dt = datetime.strptime(feed_str, fmt).replace(tzinfo=IST)
                    break
                except Exception:
                    pass

        now_ist = datetime.now(IST)
        if feed_dt is not None:
            age_sec = (now_ist - feed_dt).total_seconds()
            is_time_fresh = (-15.0 <= age_sec <= 15.0)
            quote_ts = feed_dt
        else:
            is_time_fresh = False
            quote_ts = now_ist

        if is_index:
            # Index quotes do not have order-book depth on the exchange.
            # Validity requires real exchange timestamp and positive LTP.
            is_fresh = bool(feed_dt is not None and is_time_fresh and ltp > 0.0)
        else:
            # Stock quotes require real exchange timestamp, positive LTP, and valid top-of-book depth
            is_valid_depth = (best_bid > 0.0 and best_ask > 0.0 and best_ask >= best_bid)
            is_fresh = bool(feed_dt is not None and is_time_fresh and ltp > 0.0 and is_valid_depth)

        return Quote(
            symbol=clean,
            exchange=exchange,
            last_price=ltp,
            bid=best_bid,
            ask=best_ask,
            volume=vol,
            timestamp=quote_ts,
            is_fresh=is_fresh,
        )

    def get_quotes_batch(self, symbols: list[str], exchange: str = "NSE") -> dict[str, Quote]:
        """Fetch live quote snapshots for multiple symbols in a single batch request."""
        quotes: dict[str, Quote] = {}
        now_ist = datetime.now(IST)

        clean_symbols = [s.replace("-EQ", "").upper() for s in symbols]
        tokens_by_exchange: dict[str, dict[str, str]] = {}
        for sym in clean_symbols:
            eff_exch = "MCX" if sym.startswith(("SILVER", "GOLD")) else exchange
            tok = self.resolve_token(sym, eff_exch)
            if tok:
                if eff_exch not in tokens_by_exchange:
                    tokens_by_exchange[eff_exch] = {}
                tokens_by_exchange[eff_exch][tok] = sym
            else:
                quotes[sym] = Quote(
                    symbol=sym,
                    exchange=eff_exch,
                    last_price=0.0,
                    bid=0.0,
                    ask=0.0,
                    volume=0,
                    timestamp=now_ist,
                    is_fresh=False,
                )

        if not tokens_by_exchange or not self._smart_api:
            for sym in clean_symbols:
                if sym not in quotes:
                    eff_exch = "MCX" if sym.startswith(("SILVER", "GOLD")) else exchange
                    quotes[sym] = Quote(
                        symbol=sym,
                        exchange=eff_exch,
                        last_price=0.0,
                        bid=0.0,
                        ask=0.0,
                        volume=0,
                        timestamp=now_ist,
                        is_fresh=False,
                    )
            return quotes

        total_tokens = sum(len(m) for m in tokens_by_exchange.values())
        if self._rate_limiter:
            acquired = self._rate_limiter.acquire("quotes", timeout=2.0)
            if not acquired:
                logger.warning("Rate limit timeout acquiring quotes token for batch (%d symbols)", total_tokens)
                for sym in clean_symbols:
                    if sym not in quotes:
                        eff_exch = "MCX" if sym.startswith(("SILVER", "GOLD")) else exchange
                        quotes[sym] = Quote(
                            symbol=sym,
                            exchange=eff_exch,
                            last_price=0.0,
                            bid=0.0,
                            ask=0.0,
                            volume=0,
                            timestamp=now_ist,
                            is_fresh=False,
                        )
                return quotes

        for exch, exch_tokens in tokens_by_exchange.items():
            token_list = list(exch_tokens.keys())
            chunk_size = 50
            for i in range(0, len(token_list), chunk_size):
                chunk = token_list[i : i + chunk_size]
                try:
                    if hasattr(self._smart_api, "getMarketData"):
                        m_data = self._smart_api.getMarketData(
                            mode="FULL",
                            exchangeTokens={exch: chunk},
                        )
                        if m_data and m_data.get("status") and "data" in m_data:
                            fetched = m_data["data"].get("fetched", [])
                            for item in fetched:
                                tok = str(item.get("symbolToken", ""))
                                sym = exch_tokens.get(tok)
                                if not sym:
                                    ts = str(item.get("tradingSymbol", "")).replace("-EQ", "").upper()
                                    if ts in clean_symbols:
                                        sym = ts
                                if not sym and len(chunk) == 1 and len(clean_symbols) == 1:
                                    sym = clean_symbols[0]
                                if sym:
                                    quotes[sym] = self._parse_market_item(item, sym, exch)
                except Exception as e:
                    logger.error("Error in getMarketData batch chunk (%s): %s", exch, e)

        # Fallback for any symbols missing in batch response
        for sym in clean_symbols:
            if sym not in quotes:
                quotes[sym] = Quote(
                    symbol=sym,
                    exchange=exchange,
                    last_price=0.0,
                    bid=0.0,
                    ask=0.0,
                    volume=0,
                    timestamp=now_ist,
                    is_fresh=False,
                )

        return quotes

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Quote:
        """Fetch live quote snapshot from Angel One with true market depth and exchange timestamp."""
        res = self.get_quotes_batch([symbol], exchange=exchange)
        clean = symbol.replace("-EQ", "").upper()
        return res.get(
            clean,
            Quote(
                symbol=clean,
                exchange=exchange,
                last_price=0.0,
                bid=0.0,
                ask=0.0,
                volume=0,
                timestamp=datetime.now(IST),
                is_fresh=False,
            ),
        )

    def get_historical(
        self,
        symbol: str,
        interval: str,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
        exchange: str = "NSE",
    ) -> list[Candle]:
        """Fetch historical candles from Angel One."""
        clean = symbol.replace("-EQ", "").upper()
        eff_exch = "MCX" if clean.startswith(("SILVER", "GOLD")) else exchange
        token = self.resolve_token(clean, eff_exch)

        if from_dt is None:
            from_dt = datetime.now(IST) - timedelta(days=5)
        if to_dt is None:
            to_dt = datetime.now(IST)

        # Normalize interval ("5min" -> "FIVE_MINUTE", "1min" -> "ONE_MINUTE")
        angel_interval = "FIVE_MINUTE"
        if interval in ("1min", "1m", "minute"):
            angel_interval = "ONE_MINUTE"
        elif interval in ("15min", "15m"):
            angel_interval = "FIFTEEN_MINUTE"
        elif interval in ("1d", "day"):
            angel_interval = "ONE_DAY"

        # Determine bar bucket for clock boundary invalidation
        now_ts = datetime.now(IST).timestamp()
        if angel_interval == "FIVE_MINUTE":
            bar_bucket = int(now_ts) // 300
        elif angel_interval == "ONE_MINUTE":
            bar_bucket = int(now_ts) // 60
        elif angel_interval == "FIFTEEN_MINUTE":
            bar_bucket = int(now_ts) // 900
        else:
            bar_bucket = int(now_ts) // 300

        # Check short-term candle cache to prevent burst exhaustion
        cache_key = (clean, interval)
        now_mono = time.monotonic()
        if cache_key in self._candle_cache:
            cached_t, cached_bucket, cached_candles = self._candle_cache[cache_key]
            if (bar_bucket == cached_bucket) and (now_mono - cached_t) < self._candle_cache_ttl and cached_candles:
                return cached_candles

        # Check error backoff cooldown (e.g. rate limit 429)
        if cache_key in self._error_backoff:
            if now_mono < self._error_backoff[cache_key]:
                if cache_key in self._candle_cache and self._candle_cache[cache_key][2]:
                    return self._candle_cache[cache_key][2]
                return []
            else:
                del self._error_backoff[cache_key]

        if self._smart_api:
            if self._rate_limiter:
                acquired = self._rate_limiter.acquire("historical", timeout=4.0)
                if not acquired:
                    logger.warning("Rate limit timeout acquiring historical token for %s", clean)
                    if cache_key in self._candle_cache and self._candle_cache[cache_key][2]:
                        logger.info("Serving stale cached candles for %s due to rate limit timeout", clean)
                        return self._candle_cache[cache_key][2]
                    return []

            try:
                hist_params = {
                    "exchange": eff_exch,
                    "symboltoken": token,
                    "interval": angel_interval,
                    "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
                    "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
                }
                res = self._smart_api.getCandleData(hist_params)
                # Space calls to respect Angel One rate limit
                time.sleep(0.35)
                if res and res.get("status") and "data" in res:
                    candles = []
                    for row in res["data"]:
                        t_str = row[0]
                        dt = datetime.fromisoformat(t_str.replace("Z", "+00:00")).astimezone(IST)
                        now_ist = datetime.now(IST)
                        int_upper = str(interval).upper()
                        if "1MIN" in int_upper or "1M" in int_upper or "ONE_MIN" in int_upper or int_upper == "MINUTE":
                            tf_min = 1
                        elif "3MIN" in int_upper or "3M" in int_upper or "THREE_MIN" in int_upper:
                            tf_min = 3
                        elif "15MIN" in int_upper or "15M" in int_upper or "FIFTEEN" in int_upper:
                            tf_min = 15
                        elif "30MIN" in int_upper or "30M" in int_upper or "THIRTY" in int_upper:
                            tf_min = 30
                        elif "60MIN" in int_upper or "60M" in int_upper or "1H" in int_upper or "HOUR" in int_upper:
                            tf_min = 60
                        elif "1D" in int_upper or "DAY" in int_upper:
                            tf_min = 1440
                        else:
                            tf_min = 5
                        is_done = now_ist >= (dt + timedelta(minutes=tf_min))
                        candles.append(
                            Candle(
                                timestamp=dt,
                                open=float(row[1]),
                                high=float(row[2]),
                                low=float(row[3]),
                                close=float(row[4]),
                                volume=int(row[5]),
                                is_complete=is_done,
                            )
                        )
                    if candles:
                        self._candle_cache[cache_key] = (now_mono, bar_bucket, candles)
                    return candles
            except Exception as e:
                logger.error("Angel One get_historical error for %s: %s", clean, e)
                self._error_backoff[cache_key] = now_mono + 10.0  # 10s cooldown
                if cache_key in self._candle_cache and self._candle_cache[cache_key][2]:
                    return self._candle_cache[cache_key][2]

        return []

    def resolve_active_mcx_futures(
        self,
        symbol_roots: list[str],
        min_days_to_expiry: int = 5,
        as_of: Optional[date] = None,
        scrip_master: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, InstrumentInfo]:
        """Resolve active front-month MCX futures contracts and populate token cache."""
        today = as_of or datetime.now(IST).date()
        scrip_list = scrip_master or []
        if not scrip_list and self._smart_api and self._auth_token and hasattr(self._smart_api, "searchScrip"):
            for root in symbol_roots:
                try:
                    res = self._smart_api.searchScrip("MCX", root)
                    if isinstance(res, dict) and res.get("status") and isinstance(res.get("data"), list):
                        scrip_list.extend(res["data"])
                except Exception as e:
                    logger.warning("searchScrip failed for root %s: %s", root, e)

        resolved_raw = resolve_active_mcx_futures(scrip_list, symbol_roots, min_days_to_expiry, today)
        results: dict[str, InstrumentInfo] = {}

        for root, raw in resolved_raw.items():
            tradingsymbol = str(raw.get("symbol", "") or raw.get("tradingsymbol", "")).upper()
            token = str(raw.get("token", "") or raw.get("symboltoken", "") or raw.get("instrument_token", "")).strip()
            if tradingsymbol and token:
                self._token_cache[tradingsymbol] = token
                self._token_cache[root] = token

            exp_date = parse_expiry_date(raw.get("expiry") or raw.get("tradingsymbol") or raw.get("symbol"))
            dte = (exp_date - today).days if exp_date else 999
            order_lot = int(raw.get("lotsize", 1) or 1)
            lot_mult = get_mcx_lot_multiplier(tradingsymbol or root)

            info = InstrumentInfo(
                symbol=tradingsymbol or root,
                exchange="MCX",
                symbol_root=root,
                tick_size=1.0,
                lot_size=order_lot,
                order_lot_size=order_lot,
                lot_multiplier=lot_mult,
                expiry_date=exp_date,
                days_to_expiry=dte,
                instrument_token=token,
                mis_available=True,
            )
            results[root] = info

        return results

    def get_instrument_info(self, symbol: str, exchange: Optional[str] = None) -> InstrumentInfo:
        """Fetch instrument info with verified circuit limits from exchange."""
        clean = symbol.replace("-EQ", "").upper()
        if exchange is None:
            exchange = "MCX" if clean.startswith(("SILVER", "GOLD")) else "NSE"
        token = self.resolve_token(clean, exchange)
        low_c, up_c = self._circuit_cache.get(clean, (0.0, 0.0))
        tick = 1.0 if exchange == "MCX" else 0.05
        lot_mult = get_mcx_lot_multiplier(clean) if exchange == "MCX" else 1

        return InstrumentInfo(
            symbol=clean,
            exchange=exchange,
            tick_size=tick,
            lot_size=1,
            order_lot_size=1,
            lot_multiplier=lot_mult,
            instrument_token=token,
            upper_circuit=up_c,
            lower_circuit=low_c,
            mis_available=True,
        )
