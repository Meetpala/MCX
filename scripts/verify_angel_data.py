"""Check whether Angel One API key is sending live quotes and historical data for MCX commodities."""
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apex3.config import load_config
from apex3.broker.rate_limiter import RateLimiter
from apex3.broker.angel_data_provider import AngelDataProvider

IST = ZoneInfo("Asia/Kolkata")


def check_angel_data():
    cfg = load_config(os.path.join(ROOT, "apex3", "config.yaml"))
    s = cfg.secrets

    print("=" * 80)
    print("ANGEL ONE SMARTAPI COMMODITY LIVE & HISTORICAL DATA VERIFICATION")
    print("=" * 80)
    now_ist = datetime.now(IST)
    print(f"Timestamp (IST)       : {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market Status         : {'OPEN' if (now_ist.weekday() < 5 and (9, 0) <= (now_ist.hour, now_ist.minute) <= (23, 55)) else 'CLOSED (Normal trading hours: 09:00 - 23:55 IST)'}")
    print(f"Client Code           : {s.angel_client_code}")
    print(f"API Key Configured    : {bool(s.angel_api_key)}")
    print(f"TOTP Secret Configured: {bool(s.angel_totp_secret)}")

    rl = RateLimiter(rates={"orders": 10.0, "quotes": 5.0, "historical": 3.0})
    provider = AngelDataProvider(
        api_key=s.angel_api_key,
        client_code=s.angel_client_code,
        pin=s.angel_pin,
        totp_secret=s.angel_totp_secret,
        rate_limiter=rl,
    )

    is_valid = provider.is_session_valid()
    feed_token = provider.get_feed_token()
    print(f"\n1. AUTHENTICATION & SESSION STATUS:")
    print(f"   [+] Session Valid         : {is_valid}")
    print(f"   [+] Feed Token Available  : {bool(feed_token)} ({str(feed_token)[:10]}... if feed_token else '')")

    if not is_valid:
        print("[-] Authentication failed. Cannot proceed with data checks.")
        return

    # Check 1: Resolving Commodities on MCX
    roots = ["SILVERMIC", "GOLDGUINEA", "GOLDTEN", "CRUDEOIL", "NATURALGAS", "COPPER"]
    print(f"\n2. RESOLVING ACTIVE MCX FRONT-MONTH FUTURES CONTRACTS:")
    resolved = provider.resolve_active_mcx_futures(roots)
    for r in roots:
        inst = resolved.get(r)
        if inst:
            print(f"   [+] {r:11s} -> {inst.symbol:22s} | Token: {inst.instrument_token:8s} | Expiry: {inst.expiry_date} | LotMult: {inst.lot_multiplier}")
        else:
            print(f"   [-] {r:11s} -> NOT FOUND")

    # Check 2: Live Quote & Market Depth (MCX)
    print(f"\n3. LIVE QUOTE & MARKET DATA FEED (MCX):")
    test_symbols = ["SILVERMIC", "GOLDGUINEA", "CRUDEOIL", "NATURALGAS"]
    for r in test_symbols:
        inst = resolved.get(r)
        if not inst:
            continue
        try:
            q = provider.get_quote(inst.symbol, exchange="MCX")
            if q:
                print(f"   [+] {inst.symbol}:")
                print(f"       LTP      : INR {q.last_price:,.2f}")
                print(f"       Bid / Ask: INR {q.bid:,.2f} / INR {q.ask:,.2f}")
                print(f"       Volume   : {q.volume:,}")
                print(f"       Feed Time: {q.timestamp} (Fresh: {q.is_fresh})")
            else:
                print(f"   [-] {inst.symbol}: No quote received.")
        except Exception as qe:
            print(f"   [-] {inst.symbol} Quote Error: {qe}")

    # Check 3: Historical 5-Minute Candle Data Feed (MCX)
    print(f"\n4. HISTORICAL 5-MINUTE CANDLE DATA FEED (MCX):")
    for r in ["SILVERMIC", "GOLDGUINEA", "CRUDEOIL"]:
        inst = resolved.get(r)
        if not inst:
            continue
        try:
            from_dt = now_ist - timedelta(days=7)
            c_5m = provider.get_historical(inst.symbol, "5minute", from_dt=from_dt, to_dt=now_ist, exchange="MCX")
            print(f"   [+] {inst.symbol} (5m): {len(c_5m)} candles returned.")
            if c_5m:
                first_b = c_5m[0]
                last_b = c_5m[-1]
                print(f"       Oldest bar in chunk: {first_b.timestamp.strftime('%Y-%m-%d %H:%M')} | O={first_b.open} H={first_b.high} L={first_b.low} C={first_b.close} V={first_b.volume}")
                print(f"       Latest bar in chunk: {last_b.timestamp.strftime('%Y-%m-%d %H:%M')} | O={last_b.open} H={last_b.high} L={last_b.low} C={last_b.close} V={last_b.volume}")
        except Exception as he:
            print(f"   [-] {inst.symbol} 5m History Error: {he}")

    # Check 4: Historical 1-Day (Daily) Candle Data Feed (MCX)
    print(f"\n5. HISTORICAL DAILY (1-DAY) CANDLE DATA FEED (MCX):")
    for r in ["SILVERMIC", "GOLDGUINEA"]:
        inst = resolved.get(r)
        if not inst:
            continue
        try:
            from_dt = now_ist - timedelta(days=90)
            c_1d = provider.get_historical(inst.symbol, "day", from_dt=from_dt, to_dt=now_ist, exchange="MCX")
            print(f"   [+] {inst.symbol} (Daily): {len(c_1d)} daily bars returned.")
            if c_1d:
                first_b = c_1d[0]
                last_b = c_1d[-1]
                print(f"       Oldest daily bar: {first_b.timestamp.strftime('%Y-%m-%d')} | O={first_b.open} H={first_b.high} L={first_b.low} C={first_b.close} V={first_b.volume}")
                print(f"       Latest daily bar: {last_b.timestamp.strftime('%Y-%m-%d')} | O={last_b.open} H={last_b.high} L={last_b.low} C={last_b.close} V={last_b.volume}")
        except Exception as he:
            print(f"   [-] {inst.symbol} Daily History Error: {he}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    check_angel_data()

