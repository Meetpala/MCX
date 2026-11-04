"""APEX-3 Macro Scraper and News Provider.

Thread-safe background daemon that polls high-impact economic calendar events,
scrapes breaking commodity news wires (RSS / XML / JSON), calculates economic surprise scores
via the Bullion Inverse Law, and provides zero-latency in-memory state to the StrategyEngine.
"""
from __future__ import annotations

import collections
import hashlib
import logging
import os
import re
import threading
import time as time_mod
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from apex3.news.models import MacroEvent, NewsHeadline, MacroRegimeState

logger = logging.getLogger("apex3.news")

# Rotating modern desktop browser User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# High-impact macro keywords
HIGH_IMPACT_KEYWORDS = [
    "CPI",
    "NON-FARM",
    "PAYROLL",
    "FOMC",
    "FED CHAIR",
    "POWELL",
    "PCE",
    "JOBLESS CLAIMS",
    "INITIAL CLAIMS",
    "UNEMPLOYMENT",
    "ISM",
    "GDP",
    "PPI",
    "RETAIL SALES",
    "INTEREST RATE",
]

# Bullion Polarity Lexicon
BULLISH_KEYWORDS: dict[str, float] = {
    "rate cut": 0.65,
    "dovish": 0.60,
    "ceasefire collapses": 0.70,
    "missile strike": 0.75,
    "escalates": 0.55,
    "safe-haven": 0.50,
    "central bank gold buying": 0.60,
    "inflation cools": 0.65,
    "jobless claims rise": 0.50,
    "dollar slides": 0.60,
    "yields drop": 0.60,
}

BEARISH_KEYWORDS: dict[str, float] = {
    "rate hike": -0.65,
    "hawkish": -0.60,
    "hot inflation": -0.70,
    "cpi beats": -0.75,
    "payrolls surge": -0.70,
    "dollar surges": -0.65,
    "yields spike": -0.60,
    "ceasefire agreed": -0.60,
    "cme raises margin": -0.50,
}

SHOCK_WORDS = [
    "missile strike",
    "ceasefire collapses",
    "war",
    "emergency",
    "rate hike",
    "cpi beats",
    "hot inflation",
    "surge",
    "plunge",
    "breaking",
    "attack",
    "default",
]


def parse_economic_number(val_str: Optional[str]) -> Optional[float]:
    """Parse numeric floats from strings stripping %, K, M, B."""
    if val_str is None:
        return None
    cleaned = str(val_str).strip()
    if not cleaned or cleaned.lower() in ("none", "null", "-", "tbd", "n/a"):
        return None
    multiplier = 1.0
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    elif cleaned.endswith("K") or cleaned.endswith("k"):
        multiplier = 1_000.0
        cleaned = cleaned[:-1].strip()
    elif cleaned.endswith("M") or cleaned.endswith("m"):
        multiplier = 1_000_000.0
        cleaned = cleaned[:-1].strip()
    elif cleaned.endswith("B") or cleaned.endswith("b"):
        multiplier = 1_000_000_000.0
        cleaned = cleaned[:-1].strip()
    try:
        return float(cleaned.replace(",", "")) * multiplier
    except (ValueError, TypeError):
        return None


def calculate_economic_surprise(event: MacroEvent) -> float:
    """Compute normalized surprise score using the Bullion Inverse Law.

    - Higher-than-expected US CPI / NFP / GDP = Hawkish USD = Bearish Bullion (score = -0.80)
    - Lower-than-expected US CPI / NFP / GDP = Dovish USD = Bullish Bullion (score = +0.80)
    - Inverse indicators (Unemployment, Jobless Claims):
      Higher actual = Weaker USD = Bullish Bullion (+0.80)
      Lower actual = Stronger USD = Bearish Bullion (-0.80)
    """
    if event.actual is None or event.forecast is None:
        return 0.0
    diff = event.actual - event.forecast
    if abs(diff) < 1e-7:
        return 0.0

    if not event.inverse_for_usd:
        # Standard: higher actual = hawkish USD = bearish bullion
        return -0.80 if diff > 0 else 0.80
    else:
        # Inverse: higher actual (e.g. jobless claims) = weaker USD = bullish bullion
        return 0.80 if diff > 0 else -0.80


def score_headline_lexicon(title: str) -> tuple[float, bool]:
    """Score headline using deterministic bullion polarity lexicon.

    Returns:
        (sentiment_score: float in [-1.0, 1.0], is_breaking_shock: bool)
    """
    lower = title.lower()
    score = 0.0
    matches = 0

    for kw, val in BULLISH_KEYWORDS.items():
        if kw in lower:
            score += val
            matches += 1

    for kw, val in BEARISH_KEYWORDS.items():
        if kw in lower:
            score += val
            matches += 1

    final_score = max(-1.0, min(1.0, score / max(1, matches))) if matches > 0 else 0.0

    is_shock = any(sw in lower for sw in SHOCK_WORDS) or abs(final_score) >= 0.65
    return final_score, is_shock


class MacroNewsProvider:
    """Thread-safe background daemon providing zero-latency in-memory macro and news state."""

    def __init__(self, config: Any = None):
        self._lock = threading.RLock()
        self.config = config

        # Resolve config parameters
        news_cfg = getattr(config, "news", None) if config else None
        if isinstance(config, dict) and "news" in config:
            news_cfg = config["news"]

        self.enabled = bool(getattr(news_cfg, "enabled", True) if news_cfg else True)
        self.calendar_poll_interval = int(getattr(news_cfg, "calendar_poll_interval_sec", 60) if news_cfg else 60)
        self.wire_poll_interval = int(getattr(news_cfg, "wire_poll_interval_sec", 15) if news_cfg else 15)
        self.pre_event_blackout_min = float(getattr(news_cfg, "pre_event_blackout_min", 15.0) if news_cfg else 15.0)
        self.post_event_blackout_min = float(getattr(news_cfg, "post_event_blackout_min", 5.0) if news_cfg else 5.0)
        self.macro_block_threshold = float(getattr(news_cfg, "macro_block_threshold", 0.50) if news_cfg else 0.50)
        self.catalyst_boost_threshold = float(getattr(news_cfg, "catalyst_boost_threshold", 0.60) if news_cfg else 0.60)

        # In-memory thread caches
        self._events: dict[str, MacroEvent] = {}
        self._headlines: list[NewsHeadline] = []
        self._dedup_hashes: collections.OrderedDict[str, bool] = collections.OrderedDict()
        self._max_dedup = 2000

        # Cached macro state
        self._cached_state: MacroRegimeState = MacroRegimeState(
            in_blackout_window=False,
            minutes_to_next_high_event=None,
            active_event_title="",
            macro_score=0.0,
            catalyst_aligned_buy=False,
            catalyst_aligned_sell=False,
            last_updated_utc=datetime.now(timezone.utc),
        )

        # HTTP session with strict 3.0s timeout and retries
        self._ua_index = 0
        self._session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Daemon thread control
        self._running = False
        self._cal_thread: Optional[threading.Thread] = None
        self._wire_thread: Optional[threading.Thread] = None

        if self.enabled:
            self.start()

    def _get_headers(self) -> dict[str, str]:
        ua = USER_AGENTS[self._ua_index % len(USER_AGENTS)]
        self._ua_index += 1
        return {
            "User-Agent": ua,
            "Accept": "application/json, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def start(self):
        """Starts background polling daemon threads."""
        with self._lock:
            if self._running:
                return
            self._running = True

            self._cal_thread = threading.Thread(target=self._run_calendar_loop, daemon=True, name="MacroCalPoll")
            self._cal_thread.start()

            self._wire_thread = threading.Thread(target=self._run_wire_loop, daemon=True, name="MacroWirePoll")
            self._wire_thread.start()
            logger.info("MacroNewsProvider daemon threads started.")

    def stop(self):
        """Stops background polling daemon threads."""
        with self._lock:
            self._running = False

    def add_event(self, event: MacroEvent):
        """Thread-safe injection of a MacroEvent (used in testing and feeds)."""
        with self._lock:
            self._events[event.event_id] = event
            self._recalculate_state_locked(datetime.now(timezone.utc))

    def add_headline(self, headline: NewsHeadline):
        """Thread-safe injection of a NewsHeadline (used in testing and feeds)."""
        with self._lock:
            if headline.headline_id in self._dedup_hashes:
                return
            self._dedup_hashes[headline.headline_id] = True
            if len(self._dedup_hashes) > self._max_dedup:
                self._dedup_hashes.popitem(last=False)
            self._headlines.append(headline)
            self._recalculate_state_locked(datetime.now(timezone.utc))

    def get_macro_state(self, now_utc: Optional[datetime] = None) -> MacroRegimeState:
        """Zero-latency read interface (< 0.1ms) strictly from in-memory state under lock."""
        t_now = now_utc or datetime.now(timezone.utc)
        if t_now.tzinfo is None:
            t_now = t_now.replace(tzinfo=timezone.utc)
        else:
            t_now = t_now.astimezone(timezone.utc)

        with self._lock:
            # Recompute state quickly for the given timestamp
            return self._recalculate_state_locked(t_now)

    def _recalculate_state_locked(self, now_utc: datetime) -> MacroRegimeState:
        """Evaluates blackout windows, event surprises, and news sentiment."""
        in_blackout = False
        active_title = ""
        mins_to_next: Optional[float] = None
        min_future_diff: float = float("inf")

        # 1. Evaluate Scheduled High-Impact Macro Events
        recent_surprise_scores: list[float] = []

        for evt in self._events.values():
            if evt.impact != "HIGH" and evt.currency != "USD":
                continue

            evt_ts = evt.timestamp_utc
            if evt_ts.tzinfo is None:
                evt_ts = evt_ts.replace(tzinfo=timezone.utc)
            else:
                evt_ts = evt_ts.astimezone(timezone.utc)

            diff_sec = (evt_ts - now_utc).total_seconds()
            diff_min = diff_sec / 60.0

            # Future event tracking
            if diff_min > 0:
                if diff_min < min_future_diff:
                    min_future_diff = diff_min
                    mins_to_next = diff_min

            # Blackout Window: [event - pre_blackout, event + post_blackout]
            if -self.pre_event_blackout_min <= -diff_min <= self.post_event_blackout_min:
                in_blackout = True
                active_title = evt.title

            # Check Economic Surprise if released within the past 120 minutes
            if -120.0 <= diff_min <= 0.0 and evt.actual is not None and evt.forecast is not None:
                surprise = calculate_economic_surprise(evt)
                if abs(surprise) > 0.0:
                    recent_surprise_scores.append(surprise)

        # 2. Evaluate News Wire Headlines within last 30 minutes
        cutoff_30m = now_utc - timedelta(minutes=30)
        recent_headline_scores: list[float] = []

        # Retain only headlines within last 2 hours in memory
        cutoff_2h = now_utc - timedelta(hours=2)
        valid_headlines = []
        for h in self._headlines:
            h_ts = h.published_utc
            if h_ts.tzinfo is None:
                h_ts = h_ts.replace(tzinfo=timezone.utc)
            else:
                h_ts = h_ts.astimezone(timezone.utc)

            if h_ts >= cutoff_2h:
                valid_headlines.append(h)
                if h_ts >= cutoff_30m and abs(h.bullion_sentiment) > 0.0:
                    recent_headline_scores.append(h.bullion_sentiment)

        self._headlines = valid_headlines

        # 3. Aggregate Macro Score (-1.0 to +1.0)
        macro_score = 0.0
        if recent_surprise_scores:
            # Most recent or maximum conviction economic surprise dominates
            macro_score = recent_surprise_scores[-1]
        elif recent_headline_scores:
            # Average recent wire sentiment
            macro_score = sum(recent_headline_scores) / len(recent_headline_scores)

        macro_score = max(-1.0, min(1.0, round(macro_score, 4)))

        # 4. Catalyst Alignment Flags
        catalyst_buy = (macro_score >= self.catalyst_boost_threshold) and (not in_blackout)
        catalyst_sell = (macro_score <= -self.catalyst_boost_threshold) and (not in_blackout)

        state = MacroRegimeState(
            in_blackout_window=in_blackout,
            minutes_to_next_high_event=mins_to_next,
            active_event_title=active_title,
            macro_score=macro_score,
            catalyst_aligned_buy=catalyst_buy,
            catalyst_aligned_sell=catalyst_sell,
            last_updated_utc=now_utc,
        )
        self._cached_state = state
        return state

    def _run_calendar_loop(self):
        """Background thread polling economic calendar feeds every 60s."""
        while self._running:
            try:
                self._poll_economic_calendar()
            except Exception as e:
                logger.warning("Error in economic calendar poll: %s", e)
            time_mod.sleep(self.calendar_poll_interval)

    def _poll_economic_calendar(self):
        """Fetch and parse live economic calendar feeds with strict 3.0s timeout."""
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        try:
            resp = self._session.get(url, headers=self._get_headers(), timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        country = str(item.get("country", "")).strip().upper()
                        impact = str(item.get("impact", "")).strip().capitalize()
                        title = str(item.get("title", "")).strip()
                        date_str = item.get("date")

                        if not title or not date_str:
                            continue

                        title_upper = title.upper()
                        is_high_impact = (impact == "High") or any(kw in title_upper for kw in HIGH_IMPACT_KEYWORDS)
                        if country not in ("USD", "ALL") and not is_high_impact:
                            continue

                        # Parse ISO datetime
                        try:
                            # e.g., 2026-11-04T08:30:00-05:00
                            dt = datetime.fromisoformat(date_str)
                            dt_utc = dt.astimezone(timezone.utc)
                        except Exception:
                            continue

                        actual_val = parse_economic_number(item.get("actual"))
                        forecast_val = parse_economic_number(item.get("forecast"))
                        previous_val = parse_economic_number(item.get("previous"))

                        is_inverse = any(w in title_upper for w in ("UNEMPLOYMENT", "JOBLESS", "INITIAL CLAIMS"))
                        evt_id = hashlib.sha256(f"{title}_{dt_utc.isoformat()}".encode()).hexdigest()[:16]

                        event = MacroEvent(
                            event_id=evt_id,
                            timestamp_utc=dt_utc,
                            currency="USD" if country in ("USD", "ALL") else country,
                            impact="HIGH" if is_high_impact else "MEDIUM",
                            title=title,
                            actual=actual_val,
                            forecast=forecast_val,
                            previous=previous_val,
                            inverse_for_usd=is_inverse,
                        )
                        self.add_event(event)
        except Exception as exc:
            logger.debug("Failed to fetch FairEconomy calendar: %s", exc)

    def _run_wire_loop(self):
        """Background thread polling breaking news wires every 15s."""
        while self._running:
            try:
                self._poll_breaking_wires()
            except Exception as e:
                logger.warning("Error in breaking wires poll: %s", e)
            time_mod.sleep(self.wire_poll_interval)

    def _poll_breaking_wires(self):
        """Poll RSS and JSON wires for breaking macro and bullion headlines."""
        rss_urls = [
            ("https://www.investing.com/rss/news_11.rss", "Investing_Commodities"),
            ("https://www.investing.com/rss/news_1.rss", "Investing_Forex"),
        ]

        for url, source in rss_urls:
            try:
                resp = self._session.get(url, headers=self._get_headers(), timeout=3.0)
                if resp.status_code == 200:
                    self._parse_rss_feed(resp.text, source)
            except Exception as e:
                logger.debug("Failed to fetch RSS wire %s: %s", url, e)

        # Optional Finnhub news endpoint if API key present
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        if finnhub_key:
            try:
                fh_url = f"https://finnhub.io/api/v1/news?category=general&token={finnhub_key}"
                resp = self._session.get(fh_url, headers=self._get_headers(), timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        now_utc = datetime.now(timezone.utc)
                        for item in data[:20]:
                            headline = item.get("headline", "")
                            t_stamp = item.get("datetime")
                            if headline:
                                pub_utc = datetime.fromtimestamp(t_stamp, tz=timezone.utc) if t_stamp else now_utc
                                self._process_raw_headline(headline, pub_utc, "Finnhub")
            except Exception as fh_e:
                logger.debug("Failed to fetch Finnhub news: %s", fh_e)

    def _parse_rss_feed(self, xml_text: str, source: str):
        """Parses RSS XML items into NewsHeadline models."""
        try:
            root = ET.fromstring(xml_text)
            now_utc = datetime.now(timezone.utc)
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                pub_elem = item.find("pubDate")
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.strip()
                    pub_utc = now_utc
                    if pub_elem is not None and pub_elem.text:
                        try:
                            from email.utils import parsedate_to_datetime
                            pub_utc = parsedate_to_datetime(pub_elem.text).astimezone(timezone.utc)
                        except Exception:
                            pub_utc = now_utc
                    self._process_raw_headline(title, pub_utc, source)
        except Exception as xml_err:
            logger.debug("Error parsing RSS feed from %s: %s", source, xml_err)

    def _process_raw_headline(self, title: str, pub_utc: datetime, source: str):
        """Scores and deduplicates raw headline."""
        norm_title = re.sub(r"\s+", " ", title).strip()
        h_id = hashlib.sha256(norm_title.encode("utf-8")).hexdigest()

        with self._lock:
            if h_id in self._dedup_hashes:
                return

        score, is_shock = score_headline_lexicon(norm_title)
        if abs(score) > 0.0 or is_shock:
            headline = NewsHeadline(
                headline_id=h_id,
                published_utc=pub_utc,
                source=source,
                title=norm_title,
                bullion_sentiment=score,
                is_breaking_shock=is_shock,
            )
            self.add_headline(headline)

