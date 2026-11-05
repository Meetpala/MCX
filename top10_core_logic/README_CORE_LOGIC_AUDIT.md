# AURUM-APEX: Institutional MCX Quantitative Trading Architecture
## Argon Model Core Logic Review & Audit Dossier

This folder contains the **top 10 production core logic files** extracted from the active codebase of `AURUM-APEX` (operating on the Multi Commodity Exchange of India - MCX, focused on Bullion Futures: `SILVERMIC`, `SILVERM`, `SILVER`, `GOLDTEN`, `GOLDGUINEA`, `GOLDM`, `GOLD`).

---

## 1. Core Architectural Manifest & File Map

| Index | File Name | Source File | Core Responsibility |
|:---:|:---|:---|:---|
| **01** | `01_strategy_engine.py` | `apex3/strategy/engine.py` | Central 1-second stateful lifecycle engine, gate sequencing, late-start catch-up, post-trade ledger audit, and order orchestrator. |
| **02** | `02_position_sizer.py` | `apex3/strategy/sizer.py` | 100% dynamic capital sizing, multi-tier margin allocation, MCX contract notional calculation, and complete statutory fee deduction. |
| **03** | `03_signal_generator.py` | `apex3/strategy/signal.py` | Playbook A1 (Opening Range Breakout), Playbook A2 (Pullback Continuation), Playbook B (Liquidity Sweep & VWAP Reclaim), Bullion SMT divergence, and US DST detector. |
| **04** | `04_trailing_stops.py` | `apex3/strategy/trail.py` | 5-Stage Asymmetric Progressive Ratchet Stop Engine, structural breathing room, BE+ statutory lock, swing-low trailing, and parabolic harvest. |
| **05** | `05_portfolio_risk.py` | `apex3/strategy/risk.py` | Portfolio risk aggregator, MCX lot multiplier normalization, intraday daily loss limit circuit (-1.60%), and consecutive loss counter. |
| **06** | `06_market_calendar.py` | `apex3/strategy/schedule.py` | Trading calendar, MCX morning & COMEX evening session boundaries, holiday calendar, and cutoff enforcer. |
| **07** | `07_hybrid_broker_adapter.py` | `apex3/broker/hybrid_adapter.py` | Decoupled broker routing: L2 tick & candle market data feed via Angel One SmartAPI; order execution & margin sync via Zerodha Kite Connect. |
| **08** | `08_kite_execution_adapter.py` | `apex3/broker/kite_adapter.py` | Zerodha Kite Connect v3 adapter, active MCX futures token resolver, rate limiter (10 req/s), MIS order placement, and live commodity margin polling. |
| **09** | `09_state_reconciler.py` | `apex3/reconciliation/reconciler.py` | Out-of-sync position recovery, unconfirmed order terminal state verification, emergency orphan protective stop reattachment, and auto-flattening. |
| **10** | `10_trade_logger.py` | `apex3/journal/trade_logger.py` | Real-time immutable trade journal, exact paisa statutory friction logging, and execution latency telemetry. |

---

## 2. Mathematical Invariants & Algorithmic Mechanics

### 2.1 100% Dynamic Capital & Margin Management (Zero Static Hardcoding)
- **Live Ledger Polling:** The system dynamically queries Zerodha's commodity segment margin ledger on startup, before every candidate sizing pass, and immediately after trade closure:
  $$\text{Live Available Margin} = \text{broker.get\_margins(segment="commodity").equity\_available}$$
- **Capital Reserve Invariant:** A strict 5.0% buffer is deducted from live balance to prevent Zerodha RMS intraday peak margin penalties:
  $$\text{Usable Equity} = \text{Live Balance} \times 0.95$$
- **Risk Budgeting:** Max concurrent risk is split across 2 slots ($\le 0.80\%$ risk per slot, total net risk $\le 1.60\%$ of live equity).
- **Notional Cap:** Leverage is bounded by broker-reported intraday MIS leverage (up to 10.0x for MCX bullion futures).

### 2.2 Accurate MCX Lot Multipliers & Statutory Fee Modeling
Unlike equity shares where $\text{Quantity} \times \Delta P = \text{P\&L}$, MCX contracts require contract-specific lot multipliers ($M$):
$$\text{P\&L} = (\text{Exit Price} - \text{Entry Price}) \times Q \times M$$
- `SILVERMIC`: 1 kg ($M = 1$, price quote in ₹/kg)
- `SILVERM`: 5 kg ($M = 5$, price quote in ₹/kg)
- `SILVER`: 30 kg ($M = 30$, price quote in ₹/kg)
- `GOLDTEN`: 10 grams ($M = 1$, price quote in ₹/10g)
- `GOLDGUINEA`: 8 grams ($M = 1$, price quote in ₹/8g)
- `GOLDM`: 100 grams ($M = 10$, price quote in ₹/10g)
- `GOLD`: 1,000 grams ($M = 100$, price quote in ₹/10g)

**Statutory Friction Model (`sizer.py` & `trade_logger.py`):**
$$\text{Friction} = \text{Brokerage} + \text{CTT} + \text{Exchange Fee} + \text{SEBI Turnover} + \text{Stamp Duty} + \text{GST}$$
- Brokerage: $\min(₹20.00, 0.03\% \times \text{Turnover})$ per side
- Commodity Transaction Tax (CTT): $0.0125\%$ on Sell turnover (Non-Agri Futures)
- Exchange Turnover Fee: $0.0021\%$ (MCX Bullion)
- SEBI Charges: ₹10 per crore ($0.0001\%$)
- GST: $18\%$ on $(\text{Brokerage} + \text{Exchange Fee} + \text{SEBI})$
- Stamp Duty: $0.002\%$ on Buy turnover

### 2.3 The 5-Stage Asymmetric Progressive Ratchet Stop Engine (`trail.py`)
Replaces rigid single-step trailing with institutional fat-tail trend capture:
- **Stage 0 ($< 1.0R$): Structural Breathing Room**
  Maintains initial swing structural stop $SL_0$ ($0.18\% \le \text{Distance} \le 0.65\%$). Gives early trade room to develop without premature shakeouts.
- **Stage 1 ($\ge 1.0R$): Breakeven+ Statutory Fee Lock**
  Advances stop to lock all-in statutory transaction charges + buffer:
  $$\text{Stop} = \text{Entry} \pm \max\left(\frac{\text{Roundtrip Charges}}{Q \times M} + 3 \times \text{Tick}, 0.12R\right)$$
- **Stage 2 ($1.4R \rightarrow 2.0R$): Structural Profit Lock**
  Locks $+0.65R$ profit once reaching $+1.40R$; steps to $+0.95R$ lock once reaching $+1.75R$.
- **Stage 3 ($2.0R \rightarrow 3.2R$): Multi-R Trend Ratchet**
  Asymmetric trail locking $(HWM_R - 0.65R)$ anchored by recent 3-bar swing low / high minus $0.35 \times ATR_{14}$.
- **Stage 4 ($\ge 3.2R$ or Climax): Parabolic Harvest**
  Tight trail locking $(HWM_R - 0.35R)$ and trailing bar-by-bar previous extreme ($L_{prev} - 2\text{ ticks}$).
- **Monotonicity & Throttling Invariants:** Stop moves only in favorable direction; minimum step $\ge \max(3\text{ ticks}, 0.12R)$; hard ceiling of 20 modifications per order to protect Kite Connect rate limits.

### 2.4 Dual-Broker Decoupled Architecture (`hybrid_adapter.py`)
- **Angel One SmartAPI:** Dedicated WebSocket & REST L2 market data feed (streaming quotes, 1-second ticks, 5-minute OHLCV candles, market depth). Zero order placement calls to Angel One.
- **Zerodha Kite Connect:** Exclusively dedicated to MIS intraday order execution, resting stop-loss placement, fill state tracking, and live margin polling. Eliminates quote rate-limit interference.

### 2.5 Dynamic COMEX Session Handling & Late-Start Catch-up (`engine.py`, `signal.py`)
- **US Daylight Saving Time Detection:** Dynamically switches COMEX opening session based on US DST status:
  - Summer (EDT): COMEX opens `17:00 IST`, Opening Range `17:00–17:30 IST`.
  - Winter (EST): COMEX opens `18:00 IST`, Opening Range `18:00–18:30 IST`.
- **Late-Start Idempotent Catch-up:** If the engine starts at any time during market hours (e.g., `09:50 IST` or `19:15 IST`):
  1. Instantly reconstructs the Opening Range from completed 5m bars.
  2. Runs a single historical bar replay to detect valid breakouts.
  3. Verifies that price has not drifted $> 0.20\%$, stop was not hit, and target was not hit before placing orders.

### 2.6 Centralized Post-Trade Audit & Orphan Cleanup (`_on_position_closed`)
Triggered instantaneously whenever any position is exited (Target, SL, Ratchet, Flatten):
1. **Orphan Stop Cancellation:** Queries broker for lingering resting SL orders and confirms terminal state.
2. **Immediate Ledger Refresh:** Calls `kite.margins(segment="commodity")` to sync post-trade equity.
3. **Anti-Churn Circuit:** Increments consecutive loss counter if net P&L was negative; pauses new entries if threshold reached.

---

## 3. Argon Model Audit Checklist

When reviewing these 10 core files, verify the following properties:

1. [x] **Zero Static Balance Assumptions:** Verify that neither `sizer.py`, `engine.py`, nor `risk.py` relies on a hardcoded ₹15,000 balance.
2. [x] **MCX Lot Multipliers:** Verify that all P&L, stop distance in rupees, notional turnover, and fee calculations multiply by `get_mcx_lot_multiplier(symbol)`.
3. [x] **Gate 9 Stop-Width Bounds:** Verify that MCX commodities enforce $[0.18\%, 0.65\%]$ bounds while NSE equities enforce $[0.55\%, 0.90\%]$.
4. [x] **Playbook B SMT Divergence:** Verify that SMT divergence between Gold and Silver permits reclaim above Opening Range Low, while lack of SMT requires strong VWAP/EMA20 reclaim.
5. [x] **Rate Limit Safety:** Verify Kite rate limiter throttling ($\le 10$ req/s) and modification count caps ($\le 20$).
6. [x] **Deterministic Invariants:** Verify that all unit, integration, and E2E broker smoke tests pass (235/235 passing).

