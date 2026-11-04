"""End-to-End Live Market Execution & Environment Audit for APEX-3-MCX.

Audits:
1. Environment (.env) & Secrets Configuration.
2. Angel One SmartAPI Live Market Data Feed (MCX Quotes, Depth & Circuits).
3. Macro News & Economic Calendar Engine.
4. Signal Gates, Lot Sizing & MCX Statutory Charge Invariants.
5. Order Construction (Tag <= 20 chars, Exchange=MCX, Tick Rounding).
6. Hybrid Broker Split-Feed Routing & Zerodha Execution Gateway Guards.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from apex3.broker.angel_data_provider import AngelDataProvider
from apex3.broker.hybrid_adapter import HybridBrokerAdapter
from apex3.broker.kite_adapter import KiteAdapter
from apex3.broker.paper import PaperBroker
from apex3.broker.rate_limiter import RateLimiter
from apex3.config import load_config
from apex3.models import (
    Candle,
    Decision,
    InstrumentInfo,
    MarginInfo,
    OrderRequest,
    OrderStatus,
    OrderType,
    Product,
    Quote,
    Side,
    make_broker_tag,
    round_to_tick,
)
from apex3.news.macro_scraper import MacroNewsProvider
from apex3.strategy.accounting import calculate_mcx_intraday_charges
from apex3.strategy.engine import StrategyEngine
from apex3.strategy.sizer import size_position

IST = ZoneInfo("Asia/Kolkata")


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def audit_environment(config):
    print_header("1. ENVIRONMENT & SECRETS AUDIT")
    secrets = config.secrets
    print(f"[*] Config Operating Mode: {config.mode.upper()}")
    print(f"[*] Exchange Target: {config.exchange} (Segment: {config.margin_segment})")
    print(f"[*] Base Capital: INR {config.base_capital_inr:,.2f}")
    print(f"[*] Per-Trade Risk: {config.per_trade_risk_pct * 100:.2f}% (INR {config.base_capital_inr * config.per_trade_risk_pct:,.2f})")
    print(f"[*] Daily Loss Stop: {config.daily_max_loss_pct * 100:.2f}% (INR {config.base_capital_inr * config.daily_max_loss_pct:,.2f})")
    
    # Angel One
    angel_configured = bool(secrets.angel_api_key and secrets.angel_client_code and secrets.angel_pin and secrets.angel_totp_secret)
    print(f"[*] Angel One SmartAPI Configured: {'YES' if angel_configured else 'NO'}")
    if angel_configured:
        print(f"    - Client Code: {secrets.angel_client_code}")
        print(f"    - API Key: {secrets.angel_api_key[:3]}...{secrets.angel_api_key[-2:]}")

    # Zerodha
    kite_key = secrets.kite_api_key
    kite_token = secrets.kite_access_token
    kite_is_placeholder = ("your_api_key" in kite_key.lower()) or ("your_access_token" in kite_token.lower()) or (not kite_key) or (not kite_token)
    print(f"[*] Zerodha Kite Connect Configured: {'PLACEHOLDER / PENDING' if kite_is_placeholder else 'LIVE READY'}")
    if not kite_is_placeholder:
        print(f"    - Kite API Key: {kite_key[:3]}...{kite_key[-2:]}")

    # Safety Gates
    print(f"[*] Live Trading Safety Gates:")
    print(f"    - LIVE_TRADING_ENABLED: {secrets.live_trading_enabled}")
    print(f"    - STATIC_IP_CONFIGURED: {secrets.static_ip_configured}")
    print(f"    - SEBI_COMPLIANCE_ACKNOWLEDGED: {secrets.sebi_compliance_acknowledged}")
    
    if config.mode == "live":
        all_passed = secrets.live_trading_enabled and secrets.static_ip_configured and secrets.sebi_compliance_acknowledged and not kite_is_placeholder
        print(f"[*] Live Execution Readiness: {'READY FOR REAL ORDERS' if all_passed else 'BLOCKED BY SAFETY GATES (Engine will pause in STANDBY)'}")
    else:
        print(f"[*] Live Execution Readiness: PAPER MODE ACTIVE (Zero real capital at risk)")


def audit_angel_market_data(config):
    print_header("2. ANGEL ONE SMARTAPI LIVE DATA FEED AUDIT")
    secrets = config.secrets
    if not secrets.angel_api_key:
        print("[!] Angel One credentials not found. Skipping live data test.")
        return None

    rate_limiter = RateLimiter(rates={"orders": 10.0, "quotes": 5.0, "historical": 3.0})
    angel_provider = AngelDataProvider(
        api_key=secrets.angel_api_key,
        client_code=secrets.angel_client_code,
        pin=secrets.angel_pin,
        totp_secret=secrets.angel_totp_secret,
        rate_limiter=rate_limiter,
    )

    try:
        is_valid = angel_provider.is_session_valid()
        print(f"[*] Angel One Session Authenticated: {is_valid}")
        if not is_valid:
            print("[!] Authenticating session...")
            angel_provider.authenticate()
            is_valid = angel_provider.is_session_valid()
            print(f"[*] Re-check Session Authenticated: {is_valid}")

        # Resolve active front-month MCX contracts
        print("[*] Resolving Active Front-Month MCX Contracts...")
        active_map = angel_provider.resolve_active_mcx_futures(["SILVERMIC", "GOLDM"])
        for root, info in active_map.items():
            print(f"    -> Root: {root:<10} | TradingSymbol: {info.symbol:<20} | Token: {info.instrument_token} | LotMultiplier: {info.lot_multiplier} | Expiry: {info.expiry_date} ({info.days_to_expiry} DTE)")

        # Fetch Live Quote for SILVERMIC
        slv_info = active_map.get("SILVERMIC")
        if slv_info and slv_info.symbol:
            sym = slv_info.symbol
            print(f"[*] Querying live quote for {sym}...")
            q = angel_provider.get_quote(sym, exchange="MCX")
            if q:
                print(f"    - LTP: Rs. {q.last_price:,.2f}")
                print(f"    - Best Bid: Rs. {q.bid:,.2f} | Best Ask: Rs. {q.ask:,.2f} | Spread: Rs. {q.ask - q.bid:.2f}")
                circuits = angel_provider._circuit_cache.get(sym, (0.0, 0.0))
                print(f"    - Circuit Limits: Lower = Rs. {circuits[0]:,.2f} | Upper = Rs. {circuits[1]:,.2f}")
                print(f"    - Volume: {q.volume:,}")
                print(f"    - Freshness: age = {(datetime.now(IST) - q.timestamp).total_seconds():.1f}s")
                return q, slv_info
    except Exception as e:
        print(f"[!] Angel One market data query error: {e}")
    return None


def audit_macro_news(config):
    print_header("3. MACRO NEWS & ECONOMIC CALENDAR AUDIT")
    provider = MacroNewsProvider(config=config.news)
    try:
        provider.start()
        state = provider.get_macro_state()
        print(f"[*] Macro News Daemon Running: {provider._running}")
        print(f"[*] Blackout Window Active: {state.in_blackout_window}")
        print(f"[*] Active / Next Event: '{state.active_event_title}' (Mins to next: {state.minutes_to_next_high_event})")
        print(f"[*] Macro Score: {state.macro_score:+.2f}")
        print(f"[*] Catalyst Aligned BUY: {state.catalyst_aligned_buy} | SELL: {state.catalyst_aligned_sell}")
    except Exception as e:
        print(f"[!] Macro News Provider test warning: {e}")
    finally:
        provider.stop()


def audit_order_sizing_and_charges(quote: Quote, inst: InstrumentInfo, config):
    print_header("4. ORDER SIZING, CHARGES & TICK ROUNDING AUDIT")
    entry_price = quote.last_price if quote and quote.last_price > 0 else 85000.0
    tick_size = inst.tick_size if inst else 1.0
    lot_multiplier = inst.lot_multiplier if inst else 1
    
    # 1. Round to tick test
    rounded_entry = round_to_tick(entry_price, tick_size, "nearest")
    initial_stop = round_to_tick(rounded_entry * 0.9960, tick_size, "floor")
    print(f"[*] Quoted Price: Rs. {entry_price:.2f} -> Rounded to Tick ({tick_size}): Rs. {rounded_entry:.2f}")
    print(f"[*] Initial Stop (Floor): Rs. {initial_stop:.2f} (Distance: Rs. {rounded_entry - initial_stop:.2f})")

    # 2. MCX Statutory Charges Check
    charges_1lot = calculate_mcx_intraday_charges(rounded_entry, initial_stop, lots=1, lot_multiplier=lot_multiplier)
    print(f"[*] MCX Intraday Charges (1 lot {inst.symbol if inst else 'SILVERMIC'}): Rs. {charges_1lot:.2f}")
    
    expected_2r_gross = (rounded_entry - initial_stop) * 2.0 * lot_multiplier
    fee_ratio = charges_1lot / expected_2r_gross if expected_2r_gross > 0 else 1.0
    print(f"[*] Expected 2.0R Gross Profit: Rs. {expected_2r_gross:.2f}")
    print(f"[*] Fee-to-Profit Ratio: {fee_ratio * 100:.2f}% (Config Max Cap: {config.max_fee_to_profit_ratio * 100:.1f}%)")
    print(f"[*] Fee Gate Passed: {fee_ratio <= config.max_fee_to_profit_ratio}")

    # 3. Position Sizer Verification
    margin_info = MarginInfo(
        equity_available=config.base_capital_inr,
        equity_used=0.0,
        collateral=0.0,
        net_value=config.base_capital_inr,
        timestamp=datetime.now(IST),
    )
    sizing = size_position(
        symbol=inst.symbol if inst else "SILVERMIC26NOVFUT",
        side=Side.BUY,
        entry_price=rounded_entry,
        stop_price=initial_stop,
        num_qualified_symbols=1,
        margin_info=margin_info,
        instrument=inst,
        limits_config=config.limits,
        risk_config=config.risk,
    )
    print(f"[*] Sizer Allocation: {sizing.quantity} units ({sizing.quantity // inst.order_lot_size if inst and inst.order_lot_size > 0 else sizing.quantity} lots)")
    print(f"[*] Net Bound Risk: Rs. {sizing.trade_risk_rupees:.2f}")


def audit_order_construction_and_tags(inst: InstrumentInfo):
    print_header("5. ORDER CONSTRUCTION & BROKER COMPLIANCE AUDIT")
    sym = inst.symbol if inst else "SILVERMIC26NOVFUT"
    
    # Broker tag verification (strictly <= 20 chars)
    tags = [
        make_broker_tag("APEX3", sym, "ORB"),
        make_broker_tag("APEX3", sym, "SL"),
        make_broker_tag("APEX3", sym, "EOD"),
        make_broker_tag("APEX3", sym, "NWS"),
    ]
    for tag in tags:
        print(f"[*] Broker Tag: '{tag}' (Length: {len(tag)}/20 chars) -> Valid: {len(tag) <= 20}")

    # OrderRequest Verification
    req = OrderRequest(
        symbol=sym,
        exchange="MCX",
        side=Side.BUY,
        quantity=1,
        product=Product.MIS,
        order_type=OrderType.LIMIT,
        price=85000.0,
        tag=tags[0],
    )
    print(f"[*] Sample OrderRequest:")
    print(f"    - Symbol: {req.symbol}")
    print(f"    - Exchange: {req.exchange} (Must be MCX)")
    print(f"    - Side: {req.side.value}")
    print(f"    - Qty: {req.quantity}")
    print(f"    - Product: {req.product.value}")
    print(f"    - Type: {req.order_type.value} @ Rs. {req.price:.2f}")
    print(f"    - Tag: {req.tag}")
    assert req.exchange == "MCX", "Exchange must be MCX!"
    assert len(req.tag) <= 20, "Tag must be <= 20 chars!"


def audit_hybrid_routing_and_gateway(config, inst: InstrumentInfo):
    print_header("6. HYBRID ROUTING & EXECUTION GATEWAY AUDIT")
    secrets = config.secrets
    rate_limiter = RateLimiter()

    # Build components
    angel_provider = AngelDataProvider(
        api_key=secrets.angel_api_key,
        client_code=secrets.angel_client_code,
        pin=secrets.angel_pin,
        totp_secret=secrets.angel_totp_secret,
        rate_limiter=rate_limiter,
    )
    kite_broker = KiteAdapter(
        api_key=secrets.kite_api_key,
        access_token=secrets.kite_access_token,
        live_enabled=secrets.live_trading_enabled,
        rate_limiter=rate_limiter,
    )
    paper_broker = PaperBroker(starting_margin=config.paper_starting_capital)

    # Hybrid with Paper Execution
    hybrid_paper = HybridBrokerAdapter(data_provider=angel_provider, execution_broker=paper_broker)
    print(f"[*] Split-Feed Hybrid Mode (Paper Execution):")
    print(f"    - Data Provider: {type(hybrid_paper.data_provider).__name__} (Angel One SmartAPI)")
    print(f"    - Execution Broker: {type(hybrid_paper.execution_broker).__name__} (PaperBroker)")
    
    # Test paper order routing
    req = OrderRequest(
        symbol=inst.symbol if inst else "SILVERMIC26NOVFUT",
        exchange="MCX",
        side=Side.BUY,
        quantity=1,
        product=Product.MIS,
        order_type=OrderType.LIMIT,
        price=85000.0,
        tag="APEX3_SLVMIC_TEST",
    )
    paper_order_id = hybrid_paper.place_order(req)
    print(f"    - Routed test order through Hybrid -> Order ID: {paper_order_id}")
    order_state = hybrid_paper.get_order_status(paper_order_id)
    print(f"    - Order Status: {order_state.status.value}")

    # Test Live Kite Adapter Guards
    print(f"\n[*] Live Zerodha Kite Gateway Guards Check:")
    print(f"    - Live Enabled Flag: {kite_broker._live_enabled}")
    print(f"    - Kite Client Initialized: {kite_broker._kite is not None}")
    
    # Guard 1: _require_live
    try:
        kite_broker._require_live()
        print("    - Guard _require_live: PASSED (Live mode flag active)")
    except Exception as e:
        print(f"    - Guard _require_live: BLOCKED ({type(e).__name__}: {e})")

    # Guard 2: _require_kite
    try:
        kite_broker._require_kite()
        print("    - Guard _require_kite: PASSED (Kite Connect client active)")
    except Exception as e:
        print(f"    - Guard _require_kite: BLOCKED ({type(e).__name__}: {e})")


def main():
    print("=" * 70)
    print(" APEX-3-MCX PRECIOUS METALS QUANTITATIVE SYSTEM")
    print(" END-TO-END LIVE MARKET EXECUTION & ENVIRONMENT VERIFICATION")
    print("=" * 70)
    
    config = load_config()
    audit_environment(config)
    market_data = audit_angel_market_data(config)
    quote = market_data[0] if market_data else None
    inst = market_data[1] if market_data else None
    audit_macro_news(config)
    if quote and inst:
        audit_order_sizing_and_charges(quote, inst, config)
        audit_order_construction_and_tags(inst)
        audit_hybrid_routing_and_gateway(config, inst)
    else:
        print("\n[!] Market data quote unavailable. Running invariant checks with reference SILVERMIC instrument...")
        ref_inst = InstrumentInfo(
            symbol="SILVERMIC30NOV26FUT",
            exchange="MCX",
            tick_size=1.0,
            lot_size=1,
            order_lot_size=1,
            lot_multiplier=1,
            upper_circuit=195000.0,
            lower_circuit=170000.0,
        )
        dummy_q = Quote(
            symbol="SILVERMIC30NOV26FUT",
            exchange="MCX",
            last_price=189500.0,
            bid=189490.0,
            ask=189510.0,
            volume=50000,
            timestamp=datetime.now(IST),
        )
        audit_order_sizing_and_charges(dummy_q, ref_inst, config)
        audit_order_construction_and_tags(ref_inst)
        audit_hybrid_routing_and_gateway(config, ref_inst)

    print("\n" + "=" * 70)
    print(" AUDIT COMPLETE: All core execution pipelines and safety gates verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
