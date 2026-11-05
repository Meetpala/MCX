"""End-to-End Live Broker Smoke Test for Dual-Broker Architecture (Angel One + Zerodha Kite).

Tests:
1. Angel One SmartAPI (Market Data):
   - Authenticate via TOTP
   - Resolve active front-month contracts for SILVERMIC, GOLDGUINEA, GOLDTEN
   - Query live MCX quote depth (bid/ask)
   - Fetch 5-minute historical candles
2. Zerodha Kite Connect (Execution & Margins):
   - Check session validity
   - Query live commodity margin (kite.margins(segment='commodity'))
   - Query live order margin estimation for 1 lot of active SILVERMIC
   - Query orderbook and positions reconciliation
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from apex3.config import load_config
from apex3.broker.rate_limiter import RateLimiter
from apex3.broker.angel_data_provider import AngelDataProvider
from apex3.broker.kite_adapter import KiteAdapter
from apex3.models import OrderRequest, OrderType, Product, Side

IST = ZoneInfo("Asia/Kolkata")


def run_broker_verification():
    print("=" * 80)
    print("AURUM-APEX: DUAL-BROKER E2E SMOKE & READINESS TEST")
    print("=" * 80)

    cfg = load_config(str(WORKSPACE_ROOT / "apex3" / "config.yaml"))
    rl = RateLimiter(rates={"orders": 10.0, "quotes": 5.0, "historical": 3.0})

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: Angel One SmartAPI Market Data Test
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[PART 1/2] Connecting to Angel One SmartAPI (Market Data Feed)...")
    angel_key = cfg.secrets.angel_api_key or os.environ.get("ANGEL_API_KEY", "")
    angel_client = cfg.secrets.angel_client_code or os.environ.get("ANGEL_CLIENT_CODE", "")
    angel_pin = cfg.secrets.angel_pin or os.environ.get("ANGEL_PIN", "")
    angel_totp = cfg.secrets.angel_totp_secret or os.environ.get("ANGEL_TOTP_SECRET", "")

    if not (angel_key and angel_client and angel_pin and angel_totp):
        print("[-] Angel One credentials missing in .env / config; skipping live Angel login.")
    else:
        try:
            angel = AngelDataProvider(
                api_key=angel_key,
                client_code=angel_client,
                pin=angel_pin,
                totp_secret=angel_totp,
                rate_limiter=rl,
            )
            is_valid = angel.is_session_valid()
            print(f"[+] Angel One Session Valid: {is_valid}")

            # Resolve active front-month commodity futures
            print("[*] Resolving active MCX futures for ['SILVERMIC', 'GOLDGUINEA', 'GOLDTEN']...")
            resolved = angel.resolve_active_mcx_futures(["SILVERMIC", "GOLDGUINEA", "GOLDTEN"])
            for root, inst in resolved.items():
                print(f"    - {root} -> {inst.symbol} (Token: {inst.instrument_token}, Expiry: {inst.expiry_date}, Multiplier: {inst.lot_multiplier})")

            # Fetch live quote for front-month contract
            test_sym = resolved.get("SILVERMIC", resolved.get("GOLDGUINEA", None))
            if test_sym:
                sym_str = test_sym.symbol
                print(f"[*] Fetching live quote for {sym_str} (exchange='MCX')...")
                q = angel.get_quote(sym_str, exchange="MCX")
                if q:
                    print(f"    Last Price: INR {q.last_price:,.2f} | Bid: INR {q.bid:,.2f} | Ask: INR {q.ask:,.2f} | Volume: {q.volume:,}")
                    print(f"    Quote Fresh: {q.is_fresh} (Timestamp: {q.timestamp})")
                else:
                    print("    [-] Quote unavailable (Market may be closed or token inactive)")

                # Fetch 5-minute historical candles
                now_ist = datetime.now(IST)
                from_dt = now_ist - timedelta(days=5)
                print(f"[*] Fetching 5m historical candles for {sym_str}...")
                candles = angel.get_historical(sym_str, "5minute", from_dt, now_ist, exchange="MCX")
                print(f"    Fetched {len(candles)} 5m candles successfully.")
                if candles:
                    latest = candles[-1]
                    print(f"    Latest 5m bar: {latest.timestamp.strftime('%Y-%m-%d %H:%M')} | O={latest.open} H={latest.high} L={latest.low} C={latest.close} V={latest.volume}")
        except Exception as ae:
            print(f"[-] Angel One Error: {ae}")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: Zerodha Kite Connect Execution & Margin Test
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[PART 2/2] Connecting to Zerodha Kite Connect (Order Execution & Margins)...")
    kite_key = cfg.secrets.kite_api_key or os.environ.get("KITE_API_KEY", "")
    kite_token = cfg.secrets.kite_access_token or os.environ.get("KITE_ACCESS_TOKEN", "")

    if not (kite_key and kite_token):
        print("[-] Zerodha Kite credentials missing in .env; skipping live Kite login.")
    else:
        try:
            kite = KiteAdapter(
                api_key=kite_key,
                access_token=kite_token,
                live_enabled=False,  # Read-only verification
                rate_limiter=rl,
            )
            is_kite_valid = kite.is_session_valid()
            print(f"[+] Zerodha Kite Session Valid: {is_kite_valid}")

            if is_kite_valid:
                # Margin Query for Commodity Segment
                print("[*] Fetching live commodity margin ledger (segment='commodity')...")
                margins = kite.get_margins(segment="commodity")
                print(f"    Available Cash / Live Balance : INR {margins.equity_available:,.2f}")
                print(f"    Utilized Margin (Debits)     : INR {margins.equity_used:,.2f}")
                print(f"    Net Account Value            : INR {margins.net_value:,.2f}")

                # Order Margin Estimation for 1 lot SILVERMIC
                resolved_kite = kite.resolve_active_mcx_futures(["SILVERMIC"])
                kite_inst = resolved_kite.get("SILVERMIC")
                if kite_inst:
                    sample_sym = kite_inst.symbol
                    print(f"[*] Querying exact order margin estimate for 1 lot {sample_sym}...")
                    dummy_order = OrderRequest(
                        symbol=sample_sym,
                        exchange="MCX",
                        side=Side.BUY,
                        quantity=1,
                        price=95000.0,
                        product=Product.MIS,
                        order_type=OrderType.LIMIT,
                    )
                    order_est = kite.estimate_order_margin(dummy_order)
                    print(f"    Required Margin: INR {order_est.required_margin:,.2f} (Effective Leverage: {order_est.leverage_available:.2f}x)")

                # Reconcile Orders & Positions
                print("[*] Reconciling live open orders and MIS positions on Zerodha...")
                open_orders = kite.get_orders()
                positions = kite.get_positions()
                print(f"    Open / Pending Orders: {len(open_orders)}")
                print(f"    Active MIS Positions  : {len([p for p in positions if getattr(p, 'product', Product.MIS) == Product.MIS])}")
        except Exception as ke:
            print(f"[-] Zerodha Error: {ke}")

    print("\n" + "=" * 80)
    print("BROKER READINESS AUDIT COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    run_broker_verification()

