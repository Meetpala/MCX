"""Fetch 1-year 5-minute MCX historical data for top liquid Mini/Micro contracts from Angel One."""
from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apex3.broker.angel_data_provider import AngelDataProvider
from apex3.broker.rate_limiter import RateLimiter
from apex3.config import load_config

IST = ZoneInfo("Asia/Kolkata")
MULTI_CACHE = ROOT / "data" / "mcx_multi_1y_cache.pkl"
BASE_CACHE = ROOT / "data" / "mcx_5m_cache.pkl"
NEW_ROOTS = ["CRUDEOILM", "NATGASMINI", "ZINCMINI", "ALUMINI"]


def main(days_back: int = 370):
    cfg = load_config(str(ROOT / "apex3" / "config.yaml"))
    s = cfg.secrets
    a = AngelDataProvider(
        api_key=s.angel_api_key,
        client_code=s.angel_client_code,
        pin=s.angel_pin,
        totp_secret=s.angel_totp_secret,
        rate_limiter=RateLimiter(rates={"orders": 10.0, "quotes": 5.0, "historical": 3.0}),
    )

    out = {}
    if BASE_CACHE.exists():
        base = pickle.load(open(BASE_CACHE, "rb"))
        for k in ("SILVERMIC", "GOLDGUINEA", "GOLDTEN"):
            if k in base:
                out[k] = base[k]
    if MULTI_CACHE.exists():
        prev = pickle.load(open(MULTI_CACHE, "rb"))
        out.update(prev)

    now = datetime.now(IST)
    for root in NEW_ROOTS:
        # Resolve front-month contract via Angel searchScrip
        res = a._smart_api.searchScrip("MCX", root)
        sym = None
        if isinstance(res, dict) and res.get("status") and res.get("data"):
            futs = [
                x for x in res["data"]
                if isinstance(x, dict)
                and str(x.get("tradingsymbol", "")).upper().startswith(root)
                and str(x.get("tradingsymbol", "")).upper().endswith("FUT")
            ]
            # Pick front-month with >= 5 days to expiry
            parsed = []
            for item in futs:
                tsym = str(item["tradingsymbol"]).upper()
                tail = tsym[len(root) : -3]
                try:
                    exp = datetime.strptime(tail, "%d%b%y").date()
                    if (exp - now.date()).days >= 4:
                        parsed.append((exp, tsym, str(item["symboltoken"])))
                except Exception:
                    pass
            if parsed:
                parsed.sort()
                _, sym, tok = parsed[0]
                a._token_cache[sym] = tok
                print(f"Resolved {root} -> {sym} (token={tok})", flush=True)
        if not sym:
            print(f"Could not resolve {root}", flush=True)
            continue

        bars = {}
        if root in out and "candles" in out[root]:
            for c in out[root]["candles"]:
                bars[c.timestamp] = c

        end = now
        while end > now - timedelta(days=days_back):
            start = end - timedelta(days=28)
            in_win = [k for k in bars if start <= k <= end]
            latest = max(in_win) if in_win else None
            if len(in_win) >= 1500 and latest is not None and (end - latest) <= timedelta(days=3):
                end = start
                continue
            chunk = []
            for attempt in range(3):
                try:
                    chunk = a.get_historical(sym, "5minute", start, end, exchange="MCX")
                except Exception as e:
                    print(root, "err:", e, flush=True)
                    chunk = []
                if chunk:
                    break
                time.sleep(1.5 * (attempt + 1))
            time.sleep(0.5)
            for c in chunk:
                bars[c.timestamp] = c
            end = start

        cs = [bars[k] for k in sorted(bars)]
        out[root] = {"symbol": sym, "candles": cs}
        print(
            f"SAVED {root}: {sym} | {len(cs):,} bars | "
            f"{cs[0].timestamp.date() if cs else None} -> {cs[-1].timestamp.date() if cs else None}",
            flush=True,
        )
        pickle.dump(out, open(MULTI_CACHE, "wb"))


if __name__ == "__main__":
    main()
