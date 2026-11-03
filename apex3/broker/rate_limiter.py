"""Token-bucket rate limiter for APEX-3 broker API calls.

Implements per-category rate limiting based on Kite Connect's
documented API rate limits. Thread-safe for concurrent usage
across strategy and data-feed threads.

Default rates from Kite docs:
- orders: 10 requests/second
- quotes: 1 request/second
- historical: 3 requests/second
- general: 10 requests/second
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default rate limits per category (requests per second)
_DEFAULT_RATES: dict[str, float] = {
    "orders": 10.0,
    "quotes": 1.0,
    "historical": 3.0,
    "general": 10.0,
}

# Default timeout waiting for a token (seconds)
_DEFAULT_TIMEOUT: float = 5.0


class RateLimitTimeoutError(Exception):
    """Raised when acquiring a token from the rate limiter times out."""


@dataclass
class _TokenBucket:
    """Single token bucket for one rate-limit category.

    Tokens refill at `rate` tokens per second, up to `capacity`.
    Thread-safe via an internal lock.
    """
    rate: float              # Tokens added per second
    capacity: float          # Maximum tokens in bucket
    tokens: float = 0.0      # Current available tokens
    last_refill: float = 0.0 # Timestamp of last refill (time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Initialize the bucket to full capacity."""
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill.

        Must be called while holding self.lock.
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def try_acquire(self, timeout: float = _DEFAULT_TIMEOUT) -> bool:
        """Try to acquire one token, blocking up to `timeout` seconds.

        Args:
            timeout: Maximum seconds to wait for a token.
                If 0, returns immediately without blocking.

        Returns:
            True if a token was acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout

        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

                # Calculate wait time for next token
                wait_time = (1.0 - self.tokens) / self.rate if self.rate > 0 else timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                sleep_time = min(wait_time, remaining)

            # Sleep outside the lock to allow other threads to proceed
            time.sleep(sleep_time)


class RateLimiter:
    """Multi-category token-bucket rate limiter.

    Manages separate rate-limit buckets for different API endpoint
    categories. Thread-safe for concurrent access.

    Args:
        rates: Dict mapping category names to requests-per-second.
            Defaults to Kite Connect's documented limits.
        default_timeout: Default timeout in seconds for acquire().

    Example:
        >>> limiter = RateLimiter()
        >>> if limiter.acquire("orders"):
        ...     broker.place_order(request)
        >>> else:
        ...     logger.warning("Rate limit timeout for orders")
    """

    def __init__(
        self,
        rates: Optional[dict[str, float]] = None,
        default_timeout: float = _DEFAULT_TIMEOUT,
        **kwargs: float,
    ) -> None:
        self._default_timeout = default_timeout
        self._rates = dict(_DEFAULT_RATES)
        if rates:
            self._rates.update(rates)
        for k, v in kwargs.items():
            cat = k.replace("_per_sec", "").replace("_per_second", "")
            if not cat.endswith("s"):
                cat += "s"
            self._rates[cat] = float(v)

        # Initialize a token bucket per category
        self._buckets: dict[str, _TokenBucket] = {}
        for category, rate in self._rates.items():
            self._buckets[category] = _TokenBucket(
                rate=rate,
                capacity=rate,  # Burst capacity = 1 second of tokens
            )

        logger.info(
            "RateLimiter initialized with categories: %s",
            ", ".join(f"{k}={v}/s" for k, v in self._rates.items()),
        )

    def acquire(self, category: str, timeout: Optional[float] = None) -> bool:
        """Acquire a rate-limit token for the given category.

        Blocks until a token is available or the timeout expires.
        If the category is unknown, falls back to the "general" bucket.

        Args:
            category: API endpoint category (e.g. "orders", "quotes",
                      "historical", "general").
            timeout: Max seconds to wait. Defaults to the limiter's
                     default_timeout.

        Returns:
            True if a token was acquired and the call may proceed.
            False if the timeout expired — the caller should back off
            or skip the API call.
        """
        if timeout is None:
            timeout = self._default_timeout

        # Check singular/plural aliases
        if category not in self._buckets and (category + "s") in self._buckets:
            category = category + "s"

        # Fall back to "general" for unknown categories
        if category not in self._buckets:
            logger.debug(
                "Unknown rate-limit category '%s', using 'general'",
                category,
            )
            category = "general"

        # Ensure the "general" bucket exists as final fallback
        if category not in self._buckets:
            logger.warning(
                "No rate-limit bucket for '%s' and no 'general' fallback. "
                "Allowing request without rate limiting.",
                category,
            )
            return True

        bucket = self._buckets[category]
        acquired = bucket.try_acquire(timeout)

        if not acquired:
            logger.warning(
                "Rate limit timeout: category='%s', timeout=%.1fs. "
                "Consider reducing request frequency.",
                category,
                timeout,
            )

        return acquired

    def add_category(self, category: str, rate: float) -> None:
        """Add or update a rate-limit category.

        Args:
            category: Category name.
            rate: Requests per second allowed.
        """
        self._rates[category] = rate
        self._buckets[category] = _TokenBucket(rate=rate, capacity=rate)
        logger.info("Rate-limit category added/updated: %s=%s/s", category, rate)

    def get_rates(self) -> dict[str, float]:
        """Return the current rate configuration.

        Returns:
            Dict mapping category names to requests-per-second.
        """
        return dict(self._rates)
