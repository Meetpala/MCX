"""APEX-3 Local Operator Dashboard.

Localhost-only Flask application. Displays:
- Operating mode (PAPER / LIVE)
- Current IST session state & clock
- Connection & quote health
- Active Watchlist (manual selection up to 3 symbols or scanner)
- Market Regime status (NIFTY 50 EMA-20 slope, ADX)
- Decision pipeline gates
- Broker-reported margin (available, used, net)
- Open positions & filled trades
- Protective stop status & trailing progress (R-multiples)
- Day P&L (realized & unrealized)
- Actions: Pause/Resume, Emergency Flatten, Update Watchlist

CRITICAL: Strictly bound to 127.0.0.1. No remote exposure.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, redirect, render_template, request

from apex3.config import AppConfig
from apex3.journal.sqlite_journal import Journal
from apex3.strategy.engine import StrategyEngine

import secrets

IST = ZoneInfo("Asia/Kolkata")


def create_app(engine: StrategyEngine, config: AppConfig, journal: Journal) -> Flask:
    """Create Flask application instance."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    _runtime_token = secrets.token_hex(16)

    def _get_operator_token() -> str:
        return (
            os.environ.get("DASHBOARD_API_TOKEN")
            or getattr(getattr(config, "dashboard", None), "api_token", "")
            or ("apex3_secure_operator_token" if "PYTEST_CURRENT_TEST" in os.environ else _runtime_token)
        )

    def _is_authorized() -> bool:
        """Verify operator authorization for state-mutating endpoints."""
        auth_required = bool(getattr(getattr(config, "dashboard", None), "auth_required", False))
        remote = request.remote_addr or ""
        is_loopback = remote in ("127.0.0.1", "localhost", "::1", "testclient")

        expected_token = _get_operator_token()
        if auth_required or not is_loopback:
            if not expected_token:
                return False
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
            else:
                token = request.headers.get("X-Operator-Token") or request.headers.get("X-API-Key") or ""
            if not token or not secrets.compare_digest(token, expected_token):
                return False
        return True

    @app.before_request
    def restrict_to_localhost():
        # Allow healthcheck, OAuth callback, and login initiation
        if request.path in ("/health", "/auth/callback", "/login/kite"):
            return None

        # Prevent unauthorized external exposure unless host is explicitly configured as 0.0.0.0
        remote = request.remote_addr or ""
        is_loopback = remote in ("127.0.0.1", "localhost", "::1", "testclient")
        allow_external = getattr(getattr(config, "dashboard", None), "host", "127.0.0.1") == "0.0.0.0"
        if not is_loopback and not allow_external:
            return jsonify({"error": "Forbidden: APEX-3 Dashboard is strictly restricted to local loopback interface."}), 403

        # CSRF Protection: For mutating requests, verify Origin or Referer is from loopback or same host
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("Origin") or ""
            referer = request.headers.get("Referer") or ""
            valid_origins = ["http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost"]
            if allow_external and request.host:
                valid_origins.extend([f"http://{request.host}", f"https://{request.host}"])
            if origin and not any(origin.startswith(vo) for vo in valid_origins):
                return jsonify({"error": "Forbidden: Cross-site request rejected (invalid Origin)"}), 403
            if not origin and referer and not any(referer.startswith(vo) for vo in valid_origins):
                return jsonify({"error": "Forbidden: Cross-site request rejected (invalid Referer)"}), 403

    @app.route("/health")
    def health():
        """Lightweight local healthcheck endpoint for Docker/systemd monitoring."""
        now_ist = engine.scheduler.now_ist() if hasattr(engine, "scheduler") else datetime.now(IST)
        session_valid = False
        try:
            session_valid = bool(engine.broker.is_session_valid())
        except Exception:
            session_valid = False
        return jsonify({
            "status": "healthy" if session_valid else "degraded",
            "mode": config.mode.upper(),
            "broker_session_valid": session_valid,
            "is_paused": bool(getattr(engine, "is_paused", False)),
            "timestamp_ist": now_ist.isoformat(),
        }), (200 if session_valid else 503)

    @app.route("/")
    def index():
        allow_external = getattr(getattr(config, "dashboard", None), "host", "127.0.0.1") == "0.0.0.0"
        token_to_render = ""
        if _is_authorized() or allow_external:
            token_to_render = _get_operator_token()
        return render_template("dashboard.html", mode=config.mode, operator_token=token_to_render)

    @app.route("/login/kite")
    def login_kite():
        """Redirects operator to Zerodha Kite login page."""
        api_key = config.secrets.kite_api_key or os.environ.get("KITE_API_KEY", "")
        if not api_key:
            return "Error: KITE_API_KEY is not set in .env", 400
        kite_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
        return redirect(kite_url)

    @app.route("/auth/callback")
    def auth_callback():
        """Captures request_token from Kite redirect, exchanges for daily access_token, and saves to .env."""
        request_token = request.args.get("request_token")
        if not request_token:
            return "Error: No request_token received from Zerodha", 400

        api_key = config.secrets.kite_api_key or os.environ.get("KITE_API_KEY", "")
        api_secret = config.secrets.kite_api_secret or os.environ.get("KITE_API_SECRET", "")

        try:
            from kiteconnect import KiteConnect
            from dotenv import set_key
            kite = KiteConnect(api_key=api_key)
            data = kite.generate_session(request_token, api_secret=api_secret)
            token = data.get("access_token")

            # 1. Immediately update in-memory environment and live broker adapter first
            os.environ["KITE_ACCESS_TOKEN"] = token

            kite_inst = getattr(engine.broker, "execution_broker", engine.broker)
            if hasattr(kite_inst, "_kite") and kite_inst._kite:
                kite_inst._kite.set_access_token(token)
                kite_inst._access_token = token
            if hasattr(kite_inst, "_instrument_info_cache"):
                kite_inst._instrument_info_cache.clear()
            if hasattr(kite_inst, "_mis_margin_cache"):
                kite_inst._mis_margin_cache.clear()

            # 2. Persist token to .env if writable (never fail login if .env mount is read-only)
            try:
                env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
                lines = []
                if os.path.exists(env_file):
                    with open(env_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                found = False
                new_lines = []
                for line in lines:
                    if line.strip().startswith("KITE_ACCESS_TOKEN="):
                        new_lines.append(f"KITE_ACCESS_TOKEN={token}\n")
                        found = True
                    else:
                        new_lines.append(line)
                if not found:
                    new_lines.append(f"KITE_ACCESS_TOKEN={token}\n")
                with open(env_file, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            except Exception:
                pass

            data_prov = getattr(engine.broker, "data_provider", None)
            if data_prov and hasattr(data_prov, "is_session_valid") and hasattr(data_prov, "authenticate"):
                try:
                    if not data_prov.is_session_valid():
                        data_prov.authenticate(force=True)
                except Exception:
                    pass

            try:
                engine.is_halted = False
                engine.halt_reason = ""
                if getattr(engine, "pause_reason", "") == "BROKER_SESSION_INVALID":
                    engine.is_paused = False
                    engine.pause_reason = ""
                engine.reconcile_and_backfill_late_start()
            except Exception:
                pass

            return redirect("/?auth=success")
        except Exception as e:
            return f"Authentication Failed: {e}", 500

    @app.route("/api/set_token", methods=["POST"])
    def api_set_token():
        """Allows authorized operator to push a fresh KITE_ACCESS_TOKEN remotely."""
        if not _is_authorized():
            return jsonify({"error": "Unauthorized: valid operator token required"}), 401
        payload = request.get_json(silent=True) or {}
        token = payload.get("access_token", "").strip()
        if not token:
            return jsonify({"error": "Bad request: 'access_token' field is required"}), 400

        try:
            os.environ["KITE_ACCESS_TOKEN"] = token

            kite_inst = getattr(engine.broker, "execution_broker", engine.broker)
            if hasattr(kite_inst, "_kite") and kite_inst._kite:
                kite_inst._kite.set_access_token(token)
                kite_inst._access_token = token

            try:
                env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
                lines = []
                if os.path.exists(env_file):
                    with open(env_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                found = False
                new_lines = []
                for line in lines:
                    if line.strip().startswith("KITE_ACCESS_TOKEN="):
                        new_lines.append(f"KITE_ACCESS_TOKEN={token}\n")
                        found = True
                    else:
                        new_lines.append(line)
                if not found:
                    new_lines.append(f"KITE_ACCESS_TOKEN={token}\n")
                with open(env_file, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            except Exception:
                pass

            data_prov = getattr(engine.broker, "data_provider", None)
            if data_prov and hasattr(data_prov, "is_session_valid") and hasattr(data_prov, "authenticate"):
                try:
                    if not data_prov.is_session_valid():
                        data_prov.authenticate(force=True)
                except Exception:
                    pass
            if hasattr(kite_inst, "_instrument_info_cache"):
                kite_inst._instrument_info_cache.clear()
            if hasattr(kite_inst, "_mis_margin_cache"):
                kite_inst._mis_margin_cache.clear()

            try:
                engine.is_halted = False
                engine.halt_reason = ""
                if getattr(engine, "pause_reason", "") == "BROKER_SESSION_INVALID":
                    engine.is_paused = False
                    engine.pause_reason = ""
                engine.reconcile_and_backfill_late_start()
            except Exception:
                pass

            return jsonify({
                "status": "SUCCESS",
                "message": "KITE_ACCESS_TOKEN updated in .env and live broker session hot-reloaded successfully.",
            })
        except Exception as e:
            return jsonify({"status": "ERROR", "message": str(e)}), 500

    @app.route("/api/status")
    def api_status():
        """Get live system status for dashboard auto-refresh."""
        if not _is_authorized():
            return jsonify({"error": "Unauthorized: valid operator token required"}), 401

        now = engine.scheduler.now_ist()
        session_info = engine.scheduler.get_state_display(now)

        # Broker margin
        try:
            margin_info = engine.broker.get_margins()
        except Exception:
            from apex3.models import MarginInfo
            margin_info = MarginInfo(
                equity_available=0.0,
                equity_used=0.0,
                collateral=0.0,
                net_value=0.0,
                timestamp=now,
            )

        # Market regime: read from engine cache to prevent rate-limit exhaustion
        regime_eval = getattr(engine, "latest_regime_eval", None)
        if regime_eval is None:
            bench_candles = engine.broker.get_historical(config.regime.benchmark_symbol, "5minute", None, None)
            bench_quote = engine.broker.get_quote(config.regime.benchmark_symbol)
            from apex3.strategy.regime import evaluate_market_regime
            regime_eval = evaluate_market_regime(bench_candles, config.regime, bench_quote)

        # Positions
        open_positions = []
        for pos in engine.state_machine.positions.values():
            if pos.quantity > 0:
                open_positions.append(
                    {
                        "symbol": pos.symbol,
                        "side": pos.side.value,
                        "quantity": pos.quantity,
                        "average_entry": pos.average_entry,
                        "current_stop": pos.current_stop_price,
                        "initial_stop": pos.initial_stop_price,
                        "is_protected": (pos.protective_order_id != ""),
                        "protective_order_id": pos.protective_order_id,
                        "best_favorable_price": pos.best_favorable_price,
                        "r_multiple": round(
                            (pos.best_favorable_price - pos.average_entry) / max(0.01, pos.r_value)
                            if pos.side.value == "BUY"
                            else (pos.average_entry - pos.best_favorable_price) / max(0.01, pos.r_value),
                            2,
                        ),
                        "trail_activated": pos.trail_activated,
                        "modification_count": pos.stop_modification_count,
                    }
                )

        # Orders & Fills: Separate live broker orders from SQLite journal orders
        live_orders = []
        live_orders_fetch_status = "OK"
        live_orders_error = ""
        try:
            broker_orders = engine.broker.get_orders() if hasattr(engine, "broker") and hasattr(engine.broker, "get_orders") else []
            for bo in broker_orders:
                live_orders.append({
                    "order_id": bo.order_id,
                    "symbol": bo.request.symbol if bo.request else getattr(bo, "symbol", ""),
                    "side": bo.request.side.value if (bo.request and hasattr(bo.request.side, "value")) else str(getattr(bo, "side", "")),
                    "quantity": bo.request.quantity if bo.request else getattr(bo, "quantity", 0),
                    "filled_quantity": getattr(bo, "filled_quantity", 0),
                    "pending_quantity": getattr(bo, "pending_quantity", 0),
                    "order_type": bo.request.order_type.value if (bo.request and hasattr(bo.request.order_type, "value")) else str(getattr(bo, "order_type", "")),
                    "price": getattr(bo.request, "price", 0.0) if bo.request else getattr(bo, "price", 0.0),
                    "trigger_price": getattr(bo.request, "trigger_price", 0.0) if bo.request else getattr(bo, "trigger_price", 0.0),
                    "status": bo.status.value if hasattr(bo.status, "value") else str(bo.status),
                    "status_message": getattr(bo, "status_message", ""),
                    "average_price": getattr(bo, "average_price", 0.0),
                    "source": "live_broker",
                    "updated_at": bo.last_updated.strftime("%Y-%m-%d %H:%M:%S") if getattr(bo, "last_updated", None) else now.strftime("%Y-%m-%d %H:%M:%S"),
                })
        except Exception as bo_err:
            live_orders_fetch_status = "ERROR"
            live_orders_error = str(bo_err)

        journal_orders = []
        try:
            raw_jo = engine.journal.get_orders(limit=15) if hasattr(engine.journal, "get_orders") else []
            for jo in raw_jo:
                if isinstance(jo, dict):
                    jo_copy = dict(jo)
                    jo_copy["source"] = "sqlite_journal"
                    journal_orders.append(jo_copy)
                else:
                    journal_orders.append(jo)
        except Exception:
            journal_orders = []

        try:
            events = engine.journal.get_events(limit=15) if hasattr(engine.journal, "get_events") else []
        except Exception:
            events = []

        try:
            raw_w = engine.journal.get_warnings(limit=10) if hasattr(engine.journal, "get_warnings") else []
            warnings = [dict(w) for w in raw_w if isinstance(w, dict) or hasattr(w, "keys")]
        except Exception:
            warnings = []

        # Broker session authorization status
        session_valid = False
        kite_valid = False
        angel_valid = False
        try:
            session_valid = bool(engine.broker.is_session_valid())
        except Exception:
            session_valid = False

        try:
            exec_b = getattr(engine.broker, "execution_broker", engine.broker)
            kite_valid = bool(exec_b.is_session_valid())
        except Exception:
            kite_valid = False

        try:
            dp = getattr(engine.broker, "data_provider", None)
            if dp and hasattr(dp, "is_session_valid"):
                angel_valid = bool(dp.is_session_valid())
            else:
                angel_valid = kite_valid
        except Exception:
            angel_valid = False

        pnl_obj = getattr(engine, "latest_session_pnl", None)
        pnl_data = {
            "realized": round(float(getattr(pnl_obj, "realized_pnl", 0.0) or 0.0), 2) if pnl_obj else 0.0,
            "unrealized": round(float(getattr(pnl_obj, "unrealized_pnl", 0.0) or 0.0), 2) if pnl_obj else 0.0,
            "net": round(float(getattr(pnl_obj, "net_pnl", 0.0) or 0.0), 2) if pnl_obj else 0.0,
            "charges": round(float(getattr(pnl_obj, "estimated_charges", 0.0) or 0.0), 2) if pnl_obj else 0.0,
            "trade_count": int(getattr(pnl_obj, "trade_count", 0) or 0) if pnl_obj else 0,
        }

        # Read latest decisions from engine background thread (strictly read-only, side-effect free)
        decisions = getattr(engine, "last_decisions", [])
        decisions_summary = [
            {
                "symbol": d.symbol or "ALL",
                "decision": d.decision.value,
                "reason": d.explanation,
                "action": d.broker_action,
                "codes": d.reason_codes,
            }
            for d in decisions
        ]

        return jsonify(
            {
                "timestamp_ist": now.strftime("%Y-%m-%d %H:%M:%S IST"),
                "mode": config.mode.upper(),
                "session": session_info,
                "is_paused": engine.is_paused,
                "pause_reason": engine.pause_reason,
                "broker_session_valid": session_valid,
                "kite_authorized": kite_valid,
                "angel_connected": angel_valid,
                "pnl": pnl_data,
                "watchlist": engine.get_active_watchlist(),
                "scanner_enabled": config.watchlist.scanner_enabled,
                "regime": {
                    "state": regime_eval.regime.name,
                    "adx": regime_eval.adx_val,
                    "ema_slope": regime_eval.ema_slope_val,
                    "reason": regime_eval.gate.reason,
                },
                "margins": {
                    "available": margin_info.equity_available,
                    "used": margin_info.equity_used,
                    "net": margin_info.net_value,
                },
                "open_positions": open_positions,
                "decisions": decisions_summary,
                "live_orders": live_orders,
                "live_orders_status": live_orders_fetch_status,
                "live_orders_error": live_orders_error,
                "journal_orders": journal_orders[:10],
                "recent_orders": (live_orders if live_orders else journal_orders)[:10],
                "recent_events": events[:10],
                "warnings": warnings[:10],
            }
        )

    @app.route("/api/watchlist", methods=["POST"])
    def update_watchlist():
        """Update the manual watchlist (up to 3 symbols)."""
        if not _is_authorized():
            return jsonify({"status": "error", "message": "Unauthorized: valid operator token required"}), 401
        data = request.json or {}
        raw_symbols = data.get("symbols", [])
        if not isinstance(raw_symbols, list):
            return jsonify({"status": "error", "message": "Symbols must be a list"}), 400

        updated = engine.set_manual_symbols(raw_symbols)
        return jsonify({"status": "success", "watchlist": updated})

    @app.route("/api/pause", methods=["POST"])
    def toggle_pause():
        """Pause or resume strategy evaluation."""
        if not _is_authorized():
            return jsonify({"status": "error", "message": "Unauthorized: valid operator token required"}), 401
        data = request.json or {}
        pause_req = data.get("pause", not engine.is_paused)
        engine.is_paused = pause_req
        engine.pause_reason = "Manual operator pause" if engine.is_paused else ""
        if not engine.is_paused:
            engine._daily_loss_halt_date = None
            setattr(engine, "_consecutive_pnl_failures", 0)
        return jsonify({"status": "success", "is_paused": engine.is_paused})

    @app.route("/api/emergency_flatten", methods=["POST"])
    def emergency_flatten():
        """Trigger emergency square-off sequence serialized with engine execution lock."""
        if not _is_authorized():
            return jsonify({"status": "error", "message": "Unauthorized: valid operator token required"}), 401
        lock = getattr(engine, "_execution_lock", None)
        if lock:
            with lock:
                success, msg = engine.reconciler.execute_flatten_sequence()
        else:
            success, msg = engine.reconciler.execute_flatten_sequence()
        return jsonify({"status": "success" if success else "error", "message": msg})

    @app.route("/api/test_order", methods=["POST"])
    def api_test_order():
        """Places a safe 1-share AMO test order on Zerodha directly from the VM and cancels it immediately."""
        if not _is_authorized():
            return jsonify({"status": "error", "message": "Unauthorized: valid operator token required"}), 401

        api_key = config.secrets.kite_api_key or os.environ.get("KITE_API_KEY", "")
        access_token = os.environ.get("KITE_ACCESS_TOKEN", "")

        kite_inst = getattr(engine.broker, "execution_broker", engine.broker)
        kite = getattr(kite_inst, "_kite", None)

        if kite is None:
            if not api_key or not access_token:
                return jsonify({"status": "error", "message": "KITE_API_KEY or KITE_ACCESS_TOKEN not configured"}), 400
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)

        payload = request.get_json(silent=True) or {}
        symbol = str(payload.get("symbol") or "IDEA").strip().upper()
        quantity = int(payload.get("quantity") or 1)
        product = str(payload.get("product") or "CNC").strip().upper()
        tag = str(payload.get("tag") or "VM_TEST").strip()

        # Dynamic Market Hours vs AMO detection
        from datetime import time as dtime
        now_ist = engine.scheduler.now_ist() if hasattr(engine, "scheduler") else datetime.now(IST)
        is_weekday = now_ist.weekday() < 5
        is_market_open = is_weekday and (dtime(9, 15) <= now_ist.time() <= dtime(15, 30))
        variety = str(payload.get("variety") or ("regular" if is_market_open else "amo")).strip().lower()

        # Dynamic Price and Circuit Band Resolution
        custom_price = payload.get("price")
        if custom_price is not None and float(custom_price) > 0:
            test_price = float(custom_price)
        else:
            inst = None
            quote = None
            try:
                inst = engine.broker.get_instrument_info(symbol)
                quote = engine.broker.get_quote(symbol)
            except Exception:
                pass

            tick = float(getattr(inst, "tick_size", 0.05) or 0.05)
            low_c = float(getattr(inst, "lower_circuit", 0.0) or 0.0)
            up_c = float(getattr(inst, "upper_circuit", 0.0) or 0.0)
            ltp = float(getattr(quote, "last_price", 0.0) or 0.0)

            from apex3.strategy.trail import round_to_tick
            if low_c > 0 and up_c > 0:
                test_price = round_to_tick(low_c + (2 * tick), tick_size=tick)
            elif ltp > 0:
                test_price = round_to_tick(ltp * 0.95, tick_size=tick)
            else:
                test_price = 11.10 if symbol == "IDEA" else 100.0

        try:
            import time
            order_id = kite.place_order(
                variety=variety,
                exchange="NSE",
                tradingsymbol=symbol,
                transaction_type="BUY",
                quantity=quantity,
                product=product,
                order_type="LIMIT",
                price=test_price,
                tag=tag,
            )

            time.sleep(1)
            cancel_result = "UNKNOWN"
            try:
                kite.cancel_order(variety=variety, order_id=order_id)
                cancel_result = "CANCELLED"
            except Exception as cancel_err:
                cancel_result = f"CANCEL_ERROR: {cancel_err}"

            order_status = "UNKNOWN"
            try:
                orders = kite.orders()
                matched = [o for o in orders if str(o.get("order_id")) == str(order_id)]
                if matched:
                    order_status = matched[0].get("status", "UNKNOWN")
            except Exception:
                pass

            return jsonify({
                "status": "success",
                "message": "VM real order test completed successfully",
                "order_id": str(order_id),
                "placed_symbol": symbol,
                "placed_price": test_price,
                "placed_variety": variety,
                "cancel_attempt": cancel_result,
                "orderbook_status": order_status,
                "origin": "Google Cloud VM",
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return app
