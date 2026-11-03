# Institutional-Grade MCX Precious Metals Quantitative Trading System
**Broker Architecture:** Zerodha Kite Connect v3/v4 (Single-Ledger & Commodity Segment)  
**Target Instruments:** MCX Gold (GOLD, GOLDM, GOLDTEN) & MCX Silver (SILVER, SILVERM, SILVERMIC)  
**Execution Target:** Antigravity IDE Agent Autonomous Deployment  

---

## 1. Executive Diagnostic: Why Broad Scanners & Aggressive Stops Fail on MCX

Before deploying capital in live MCX futures, we must diagnose why conventional retail algorithms lose money and why a small, evidence-backed, regime-gated strategy outperforms broad multi-commodity scanners.

### 1.1 The Multi-Commodity Scanner Trap (Crude + NatGas + Copper + Bullion)
Scanning 5–8 MCX commodities simultaneously introduces three fatal structural flaws:
1. **Microstructure & Regime Divergence:**
   - **Crude Oil (`CRUDEOIL` / `CRUDEOILM`) & Natural Gas (`NATURALGAS`):** Driven by OPEC+ headlines, EIA inventory shocks, and weather models. Price action is heavily mean-reverting with violent intraday stop-hunts ($2\%$ to $4\%$ whipsaws within minutes) and requires a steep $31.25\%$ NRML margin on Zerodha (~Rs. 2,83,540 per standard Crude lot).
   - **Precious Metals (`GOLD` / `SILVER`):** Driven by US Real Yields (10Y TIPS), the US Dollar Index ($DXY$), central bank flows, and London/COMEX liquidity windows. Once an institutional trend initiates during the London or New York overlap, precious metals exhibit high serial autocorrelation (clean intraday trend continuation) with significantly lower margin requirements ($9.25\%$ for Gold, $15.52\%$ to $15.83\%$ for Silver).
2. **Hidden Cross-Asset Correlation Spike:**
   - During a sudden $DXY$ or US Treasury yield spike at 18:00 IST / 19:00 IST, Gold, Silver, Copper, and Zinc all react to the exact same macro impulse simultaneously. A broad scanner triggers concurrent positions across multiple metals, silently multiplying your single-event risk from $1\%$ to $4\%–5\%$ of account equity.
3. **Margin Starvation & Reject Cascades:**
   - Holding a marginal setup in Crude Oil locks up over $31\%$ of contract value, starving the account of free margin when an A+ high-expectancy breakout occurs in `SILVERM` or `GOLDM`.

### 1.2 The Aggressive (Tight) Stop-Loss Fallacy
- Placing an arbitrarily tight stop-loss (e.g., $0.15\%$ or a static Rs. 80 on Gold / Rs. 150 on Silver) places your exit order directly inside the 1-Standard-Deviation ($1\sigma$) market-maker liquidity sweep zone.
- On MCX, domestic liquidity pools are routinely swept by $0.3\times$ to $0.6\times \text{ATR}_{14}$ right as London (13:30 IST) or COMEX (18:00/19:00 IST) opens, right before price moves $2.5\times$ to $4.0\times \text{ATR}_{14}$ in the intended direction.
- **The Fix:** Instead of an aggressive initial stop, use a **Volatility-Adjusted Structural Stop** ($1.5\times \text{ATR}_{14}$) paired with smaller initial lot sizing (`GOLDM`/`GOLDTEN` or `SILVERM`/`SILVERMIC`), and deploy an **Asymmetric Ratchet Trailing Stop-Loss (TSL)** that trails "only a little ahead" once the trade achieves structural validation.

### 1.3 The Fixed Target Fallacy
Static 1:1 or 1:1.5 take-profit orders cap your right-tail winners on trending COMEX sessions while leaving you exposed to full $-1.0R$ stops on choppy days. By eliminating hard profit caps and using a **4-Stage Breathing Ratchet Trail**, the strategy captures full multi-hour institutional trends while locking in incremental profit floor by floor.

---

## 2. MCX Instrument Matrix & Capital Tier Routing (Live Reference Specs)

The system dynamically selects the exact MCX contract based on live margin fetched from Zerodha (`kite.margins("commodity")`) so that **Risk per Trade never exceeds $1.0\%$ of Total Available Capital** and **Margin Utilized never exceeds $65\%$ of Free Margin**.

| Commodity Tier | Symbol | Lot Size | Quote Unit | Tick Value ($\Delta$Rs. 1) | Approx. NRML Margin % | Ideal Account Capital Tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gold Micro/Ten** | `GOLDTEN` | $10\text{ g}$ | $10\text{ g}$ | $\text{Rs. } 1$ | $9.25\%$ (~Rs. 13,150) | $\text{Rs. } 35,000 - \text{Rs. } 1,50,000$ |
| **Gold Mini** | `GOLDM` | $100\text{ g}$ | $10\text{ g}$ | $\text{Rs. } 10$ | $9.25\%$ (~Rs. 1,29,700) | $\text{Rs. } 2,50,000 - \text{Rs. } 15,00,000$ |
| **Gold Standard** | `GOLD` | $1\text{ kg}$ | $10\text{ g}$ | $\text{Rs. } 100$ | $9.25\%$ (~Rs. 12,96,700) | $> \text{Rs. } 25,00,000$ |
| **Silver Micro** | `SILVERMIC` | $1\text{ kg}$ | $1\text{ kg}$ | $\text{Rs. } 1$ | $15.52\%$ (~Rs. 28,800) | $\text{Rs. } 50,000 - \text{Rs. } 2,50,000$ |
| **Silver Mini** | `SILVERM` | $5\text{ kg}$ | $1\text{ kg}$ | $\text{Rs. } 5$ | $15.52\%$ (~Rs. 1,44,080) | $\text{Rs. } 3,00,000 - \text{Rs. } 18,00,000$ |
| **Silver Standard** | `SILVER` | $30\text{ kg}$ | $1\text{ kg}$ | $\text{Rs. } 30$ | $15.83\%$ (~Rs. 8,72,600) | $> \text{Rs. } 20,00,000$ |

**Critical Delivery & Tender Period Rule:**  
MCX Gold and Silver contracts are compulsory staggered physical delivery contracts. Zerodha squares off expiring precious metal contracts at 22:30 IST five trading days prior to expiry. The algorithm must automatically roll over its active `instrument_token` to the next month's contract **$7$ calendar days before the current contract's expiry date**.

---

## 3. The Strategy: Volatility-Normalized VWAP-ATR Regime Breakout & Pullback Continuation (VVR-BPC)

A small, high-precision, evidence-backed strategy operating on a **15-Minute Regime Filter** and a **5-Minute Execution Trigger**, strictly gated by a **Market Condition Waiting Engine**.

### 3.1 Mathematical Indicators & Definitions

- **Session-Anchored VWAP ($VWAP_t$) & Standard Deviation Bands ($\sigma_t$):**  
  Reset every morning at 09:00 IST:
  $$VWAP_t = \frac{\sum_{i=1}^{t} P_i \cdot V_i}{\sum_{i=1}^{t} V_i}, \quad P_i = \frac{H_i + L_i + C_i}{3}$$
  $$\sigma_t = \sqrt{\frac{\sum_{i=1}^{t} V_i \cdot (P_i - VWAP_t)^2}{\sum_{i=1}^{t} V_i}}$$

- **True Range & Wilder's Average True Range ($ATR_{14}$ on 5m):**
  $$TR_t = \max\left(H_t - L_t, \vert{}H_t - C_{t-1}\vert{}, \vert{}L_t - C_{t-1}\vert{}\right), \quad ATR_{14, t} = \frac{13 \cdot ATR_{14, t-1} + TR_t}{14}$$

- **Choppiness Index ($CI_{14}$ on 15m — The Anti-Whip Filter):**
  $$CI_{14} = 100 \times \frac{\log_{10}\left(\frac{\sum_{i=1}^{14} TR_i}{\max(H_{1..14}) - \min(L_{1..14})}\right)}{\log_{10}(14)}$$
  *$CI_{14} > 55$: sideways chop. $CI_{14} < 48$: clean directional expansion.*

- **Trend Strength ($ADX_{14}$ on 15m)** and **$EMA_{20}, EMA_{50}$ on 5m & 15m**.

### 3.2 State 0: The "Wait for Market Condition" Gate (REGIME_WAITING)
The system **MUST NOT TRADE** if ANY of the following conditions are true:
1. **Chop Gate:** 15-minute $CI_{14} > 52$ AND 15-minute $ADX_{14} < 20$.
2. **VWAP Magnet Trap:** Current price $C_t \in [VWAP_t - 0.35\sigma_t, \; VWAP_t + 0.35\sigma_t]$ with 5-minute $EMA_{20}$ and $EMA_{50}$ intertwined ($\frac{\vert{}EMA_{20} - EMA_{50}\vert{}}{C_t} < 0.0008$).
3. **Volatility Compression / Dead Tape:** Current 5-minute $ATR_{14} < 0.65 \times \text{SMA}(ATR_{14}, 50)$ OR current 5-minute tick volume $V_t < 0.8 \times \text{SMA}(V, 20)$.
4. **USD/INR Distortion Guard:** If MCX movement is driven solely by USD/INR spikes while spot ($XAUUSD$ / $XAGUSD$) is flat/opposing, block trend continuation.

### 3.3 Bullish Regime & Long Entry Rules (BULL_ACTIVE)
All conditions must align at the close of a 5-minute candle ($t$):
- **Macro/15m Trend Alignment:** 15m $C_t > VWAP_t$, 15m $EMA_{20} > EMA_{50}$, and $ADX_{14} \ge 20$ (or rising for 3 consecutive bars with $CI_{14} < 50$).
- **Trigger A — Institutional Compression Breakout:**
  - Price breaks and closes above Session Anchor High (09:00–09:45 Morning ORB High or 17:30–18:15 / 18:30–19:15 Pre-COMEX High).
  - Candle body ratio $\frac{C_t - O_t}{H_t - L_t} \ge 0.60$.
  - Volume confirmation: $V_t \ge 1.4 \times \text{SMA}(V, 20)$.
  - Overextension Guard: $C_t \le VWAP_t + 2.0\sigma_t$, and 5m $RSI_{14} \in [55, 73]$.
- **Trigger B — First Institutional Pullback Continuation:**
  - After a confirmed breakout above $VWAP_t + 1.0\sigma_t$, price pulls back gently on declining volume ($V_{\text{pullback}} < \text{SMA}(V, 20)$) into dynamic support $[5\text{m } EMA_{20}, \; VWAP_t + 0.5\sigma_t]$.
  - Triggers when 5m bullish engulfing or hammer closes above prior candle's high ($C_t > H_{t-1}$) while holding above 5m $EMA_{50}$.

### 3.4 Bearish Regime & Short Entry Rules (BEAR_ACTIVE)
All conditions must align at the close of a 5-minute candle ($t$):
- **Macro/15m Trend Alignment:** 15m $C_t < VWAP_t$, 15m $EMA_{20} < EMA_{50}$, and $ADX_{14} \ge 20$ (or rising for 3 consecutive bars with $CI_{14} < 50$).
- **Trigger A — Institutional Breakdown:**
  - Price breaks and closes below Session Anchor Low (Morning ORB Low or Pre-COMEX Low).
  - Candle body ratio $\frac{O_t - C_t}{H_t - L_t} \ge 0.60$.
  - Volume confirmation: $V_t \ge 1.4 \times \text{SMA}(V, 20)$.
  - Overextension Guard: $C_t \ge VWAP_t - 2.0\sigma_t$, and 5m $RSI_{14} \in [27, 45]$.
- **Trigger B — First Bearish Pullback Continuation:**
  - After confirmed breakdown below $VWAP_t - 1.0\sigma_t$, price retraces on low volume into resistance zone $[5\text{m } EMA_{20}, \; VWAP_t - 0.5\sigma_t]$.
  - Triggers when 5m bearish rejection candle closes below prior low ($C_t < L_{t-1}$) while holding below 5m $EMA_{50}$.

---

## 4. Verified Intraday Timing Architecture & Seasonal DST Engine

### 4.1 Dynamic US DST Schedule Detector
- **US Summer Time** (Second Sunday in March to First Sunday in November):
  - MCX Market Hours: 09:00 IST to 23:30 IST.
  - US COMEX Open Impact: 17:50 IST to 18:30 IST.
- **US Winter Time** (First Sunday in November to Second Sunday in March):
  - MCX Market Hours: 09:00 IST to 23:55 IST.
  - US COMEX Open Impact: 18:50 IST to 19:30 IST.

### 4.2 Intraday Execution & Cutoff Windows

| Time Window (Summer / Winter IST) | System State | Operational Rule |
| :--- | :--- | :--- |
| **09:00 – 09:15** | `OPENING_STABILIZATION` | NO TRADES. Absorb overnight gap; build 15m ORB and initial VWAP anchor. |
| **09:15 – 11:30** | `WINDOW_1_ASIA_INDIA` | ACTIVE. Morning ORB Breakouts or gap-and-go pullbacks if $V_t \ge 1.5\times$. |
| **11:30 – 13:30** | `MIDDAY_CHOP_GUARD` | RESTRICTED. Trail open positions only. Block new entries unless $ADX_{14} > 28$ & $V_t > 2.0\times$. |
| **13:30 – 16:30** | `WINDOW_2_LONDON_OPEN` | ACTIVE. High-probability trend emergence on London Bullion Market (LBMA) desk flows. |
| **16:30 – 17:45 / 18:45** | `PRE_NY_CONSOLIDATION` | WAIT/BUILD ANCHOR. Record Pre-COMEX high/low box. Tighten trailing stops. |
| **17:45 – 21:45 / 18:45 – 22:15**| `WINDOW_3_COMEX_PRIME` | PRIME EXECUTION. Highest volume and cleanest directional moves of the day. |
| **22:15 – 23:10 / 22:40 – 23:35**| `NO_NEW_ENTRY_CUTOFF` | STRICT ENTRY BLOCK ($T_{\text{close}} - 75\text{m}$). Existing profitable positions continue trailing only. |
| **23:10 – 23:30 / 23:35 – 23:55**| `HARD_AUTO_SQUAREOFF` | MANDATORY INTRADAY FLATTEN ($T_{\text{close}} - 20\text{m}$). Cancel pending orders, exit open positions via marketable Limit. |

**Product-Type Optimization:**  
Always place intraday orders using `product="NRML"` while enforcing your own hard-coded intraday square-off at $T_{\text{close}} - 20\text{ mins}$ (23:10 IST Summer / 23:35 IST Winter). This avoids Zerodha's MIS penalty (Rs. 50 + 18% GST call-and-trade penalty) and broker RMS race conditions while remaining 100% intraday.

---

## 5. Initial Stop-Loss & The 4-Stage "Little Ahead" Ratchet Trailing Engine

### 5.1 Initial Stop-Loss ($SL_0$) Calculation
- **Longs:** $SL_0 = \min(L_{\text{signal\_candle}} - 2\text{ ticks}, \; P_{\text{entry}} - 1.5 \times ATR_{14})$ (Capped at maximum distance of $2.2 \times ATR_{14}$).
- **Shorts:** $SL_0 = \max(H_{\text{signal\_candle}} + 2\text{ ticks}, \; P_{\text{entry}} + 1.5 \times ATR_{14})$ (Capped at maximum distance of $2.2 \times ATR_{14}$).
- **Unit Risk ($1R$):** $1R = |P_{\text{entry}} - SL_0|$.

### 5.2 Dynamic All-In Breakeven Buffer ($\Delta_{\text{cost}}$)
$$\Delta_{\text{cost}} = \left(P_{\text{entry}} \times 0.00065\right) + \frac{\text{Rs. } 48}{\text{Lot Multiplier}} + 2\text{ ticks}$$
- **Long True Breakeven ($BE^+$):** $BE^+ = P_{\text{entry}} + \Delta_{\text{cost}}$
- **Short True Breakeven ($BE^-$):** $BE^- = P_{\text{entry}} - \Delta_{\text{cost}}$

### 5.3 4-Stage Trailing Logic ($R_{\text{peak}} = \frac{|P_{\text{peak}} - P_{\text{entry}}|}{1R}$)
Stop-loss $SL_t$ is strictly monotonic (Longs: $SL_t = \max(SL_{t-1}, SL_{\text{cand}})$; Shorts: $SL_t = \min(SL_{t-1}, SL_{\text{cand}})$):
- **Stage 0 — Initial Breathing Room ($0 \le R_{\text{peak}} < 1.0R$):** $SL_t = SL_0$.
- **Stage 1 — Cost-Plus Risk Elimination ($1.0R \le R_{\text{peak}} < 1.5R$):** Ratchet $SL_t$ to $BE^+$ ($P_{\text{entry}} \pm \Delta_{\text{cost}}$). Net risk becomes $0.00$.
- **Stage 2 — "Little Ahead" Step-Ratchet ($1.5R \le R_{\text{peak}} < 3.0R$):**
  $$SL_{\text{candidate, Long}} = \max\left(BE^{+} + 0.60 \cdot (R_{\text{peak}} - 1.0) \cdot 1R, \;\; P_{\text{peak}} - 1.25 \times ATR_{14}\right)$$
  $$SL_{\text{candidate, Short}} = \min\left(BE^{-} - 0.60 \cdot (R_{\text{peak}} - 1.0) \cdot 1R, \;\; P_{\text{peak}} + 1.25 \times ATR_{14}\right)$$
- **Stage 3 — Parabolic Climax Lock ($R_{\text{peak}} \ge 3.0R$ OR $C_t$ pierces $VWAP_t \pm 2.5\sigma_t$ OR 5m $RSI_{14} > 78 / < 22$):**
  $$SL_{\text{candidate, Long}} = \max\left(L_{t-1} - 2\text{ ticks}, \;\; P_{\text{peak}} - 0.65 \times ATR_{14}\right)$$
  $$SL_{\text{candidate, Short}} = \min\left(H_{t-1} + 2\text{ ticks}, \;\; P_{\text{peak}} + 0.65 \times ATR_{14}\right)$$

**Zerodha 25-Modification Limit Protection:**
Only call `kite.modify_order()` when:
1. $|SL_{\text{new}} - SL_{\text{current}}| \ge \max(0.20 \times ATR_{14}, \; 5\text{ ticks})$
2. $\ge 4\text{ seconds}$ elapsed since last modification
3. If `mod_count == 22`, cancel existing SL and place fresh SL order after cancellation confirmation.

---

## 6. Live Margin & Fault-Tolerant Execution Engine

- Query `kite.margins("commodity")` (or Single Ledger `margins["equity"]["net"]`).
  $$\text{Usable Margin} = 0.65 \times \max(\text{margins}[\text{"commodity"}][\text{"available"}][\text{"live\_balance"}], \; \text{margins}[\text{"commodity"}][\text{"net"}])$$
- Query `kite.order_margins(order_params)` (`POST /margins/orders`).
- Position Sizing:
  $$\text{Max Lots by Risk} = \left\lfloor \frac{\text{Account Equity} \times 0.01}{1R \times \text{Lot Multiplier}} \right\rfloor$$
  $$\text{Max Lots by Margin} = \left\lfloor \frac{\text{Usable Margin}}{\text{Single Lot Required Margin}} \right\rfloor$$
  $$\text{Final Lots} = \min(\text{Max Lots by Risk}, \; \text{Max Lots by Margin})$$
  *If Final Lots == 0 for standard/mini, automatically step down to micro (`GOLDTEN` / `SILVERMIC`).*

- **Order State Handling & Failure Modes:**
  - Entry via Marketable Limit (Buy: LTP + 0.10%, Sell: LTP - 0.10%).
  - 5-second Partial-Fill Timeout: If unfilled at 5.0s, cancel remaining quantity. Set active quantity to filled quantity and immediately place exchange SL order for filled quantity.
  - Exchange SL-L Order: Trigger = $SL_0$, Limit = $SL_0 \mp 0.35\%$ slippage buffer.
  - Emergency Marketable Exit: If SL placement is rejected (e.g. LTP crossed trigger), fire immediate marketable exit limit order at LTP $\mp 0.50\%$ within 100ms.
  - 10-second Reconciler: Compares local positions with `kite.positions()["net"]` and pending SL orders.

---

## 7. Definition of Correctness (6 Core Invariants)

1. **Positive Net Expectancy Invariant:** Net P&L = Gross P&L - Total Costs > 0 on all Stage 1+ exits.
2. **Monotonic Trailing Invariant:** $SL(t_2) \ge SL(t_1)$ for Longs; $SL(t_2) \le SL(t_1)$ for Shorts.
3. **Zero Naked Exposure Invariant:** $\Delta t_{\text{naked}} < 1.5\text{ seconds}$.
4. **Zero Overnight Carry Invariant:** $0$ open positions and orders at $T_{\text{close}} - 18\text{m}$.
5. **Daily Circuit Breaker Invariant:** Realized + Unrealized Loss $\le -2.0\%$ or 2 stop-outs $\rightarrow$ `LOCKED_FOR_DAY`.
6. **Pre-Close Cutoff Invariant:** Zero new entries at or after $T_{\text{close}} - 75\text{m}$.

