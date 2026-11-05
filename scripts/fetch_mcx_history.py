"""Download and cache MCX 5m history from Angel One (read-only market data) for backtesting."""
from __future__ import annotations
import pickle, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from apex3.config import load_config
from apex3.broker.rate_limiter import RateLimiter
from apex3.broker.angel_data_provider import AngelDataProvider

IST = ZoneInfo("Asia/Kolkata")
ROOTS = ["SILVERMIC", "GOLDGUINEA", "GOLDTEN"]
CACHE = ROOT / "data" / "mcx_5m_cache.pkl"


def main(days_back: int = 240):
    cfg = load_config(str(ROOT / "apex3" / "config.yaml"))
    s = cfg.secrets
    a = AngelDataProvider(api_key=s.angel_api_key, client_code=s.angel_client_code, pin=s.angel_pin,
                          totp_secret=s.angel_totp_secret,
                          rate_limiter=RateLimiter(rates={"orders": 10.0, "quotes": 5.0, "historical": 3.0}))
    res = a.resolve_active_mcx_futures(ROOTS)
    now = datetime.now(IST)
    out = {}
    for r in ROOTS:
        sym = res[r].symbol
        bars = {}
        end = now
        while end > now - timedelta(days=days_back):
            start = end - timedelta(days=25)
            chunk = []
            for attempt in range(4):
                try:
                    chunk = a.get_historical(sym, "5minute", start, end, exchange="MCX")
                except Exception as e:
                    print(r, "chunk error", e)
                    chunk = []
                if chunk:
                    break
                time.sleep(2.0 * (attempt + 1))
            time.sleep(1.0)
            for c in chunk:
                bars[c.timestamp] = c
            end = start
        out[r] = {"symbol": sym, "candles": [bars[k] for k in sorted(bars)]}
        cs = out[r]["candles"]
        print(r, sym, len(cs), cs[0].timestamp if cs else None, "->", cs[-1].timestamp if cs else None)
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(out, open(CACHE, "wb"))


if __name__ == "__main__":
    main()
