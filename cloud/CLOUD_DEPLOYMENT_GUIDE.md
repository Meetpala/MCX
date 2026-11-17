# APEX-3 MCX Cloud Deployment & Broker Architecture Guide

## 1. End-to-End Cloud-to-Broker Architecture

```
+-------------------------------------------------------------------------+
|  CLOUD VM (Oracle Always-Free ARM / GCP e2-micro / AWS t3.micro)        |
|  Region: Mumbai (ap-south-1 / asia-south1) | Static Public IPv4         |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  | Docker Container: apex3-mcx-engine (TZ=Asia/Kolkata)              |  |
|  |                                                                   |  |
|  |  1. AngelDataProvider (Auto-TOTP via pyotp)                       |  |
|  |     -> Connects outbound HTTPS/WSS to Angel One SmartAPI          |  |
|  |     -> Zero manual daily login needed! Auto-renews session        |  |
|  |     -> Feeds live quotes + 5m/1d historical candles (MCX)         |  |
|  |                                                                   |  |
|  |  2. StrategyEngine + Reconciler + TradeLogger (info.log + SQLite) |  |
|  |     -> Late-start backfill works at ANY time (09:50, 11:30, 18:45)|  |
|  |     -> Progressive 4-stage ratchet + dynamic live margin sizing   |  |
|  |                                                                   |  |
|  |  3. KiteAdapter (Zerodha Personal Free API)                       |  |
|  |     -> Whitelisted Static Cloud IP (SEBI 1 Apr 2026 Compliance)   |  |
|  |     -> Places LIMIT / MARKET / SL / SL-M orders on MCX            |  |
|  |     -> Daily OAuth Access Token updated via Dashboard or .env     |  |
|  |                                                                   |  |
|  |  4. Flask Operator Dashboard (0.0.0.0:5000 -> Reverse Proxy/HTTPS)|  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

## 2. Why Static IP in Cloud Is Mandatory (SEBI Retail Algo Rule — 1 Apr 2026)
* Zerodha Kite requires your cloud VM's **static IPv4 address** (AWS Elastic IP / GCP Static External IP / Oracle Reserved Public IP) to be whitelisted in your Kite Developer Console.
* Set `STATIC_IP_CONFIRMED=true` and `SEBI_COMPLIANCE_ACKNOWLEDGED=true` in `.env` before enabling `LIVE_TRADING_ENABLED=true`.

## 3. Recommended Zero-Cost / Low-Cost Cloud Hosting (Crucial for ₹30k Capital!)
> **Important Economics:** Never spend ₹1,000/month (₹12,000/yr) on a cloud server for a ₹30,000 trading account — server fees would wipe out profits. Use a free-tier Mumbai/Hyderabad VM:
* **Option A (Best — ₹0/month):** Oracle Cloud Always-Free VM (Hyderabad/Mumbai) with Reserved Static Public IPv4.
* **Option B (₹0/month):** Google Cloud Always-Free `e2-micro` (or low-cost `asia-south1` Mumbai VM).
* **Option C (~₹450/month):** Hetzner / DigitalOcean / AWS Lightsail Mumbai with Static IP.

## 4. Step-by-Step Cloud Turn-On (5 Minutes)

### Step 1: Provision Ubuntu 22.04/24.04 VM with Static IP
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
```

### Step 2: Upload Code & Configure `.env`
```bash
git clone <your-private-repo-url> apex3-mcx && cd apex3-mcx
cp .env.example .env
nano .env
```
Fill in:
* `ANGEL_API_KEY`, `ANGEL_CLIENT_CODE`, `ANGEL_PIN`, `ANGEL_TOTP_SECRET` (auto-logs in every day!)
* `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`
* `LIVE_TRADING_ENABLED=false` (first test in `paper` mode for 5 sessions, then switch to `true`)
* `STATIC_IP_CONFIRMED=true` & `SEBI_COMPLIANCE_ACKNOWLEDGED=true`

### Step 3: Start Container (Auto-Restarts on Reboot/Crash)
```bash
# Start in Paper Mode (Live Angel One Data + Simulated Fills)
APEX3_MODE=paper APEX3_SYMBOLS=SILVERMIC docker compose up -d --build

# Check live logs
docker compose logs -f apex3-mcx
```

### Step 4: Switch to Live Mode (After Daily Kite Login)
```bash
# Update KITE_ACCESS_TOKEN in .env (Zerodha tokens expire daily at 06:00 IST), then:
APEX3_MODE=live APEX3_SYMBOLS=SILVERMIC docker compose up -d
```

## 5. Starting Late (e.g., 09:50 IST, 11:30 IST, or 18:45 IST)
* You do **NOT** need to start before 09:00 AM.
* Whenever you boot or refresh the Kite token (even at **09:50 IST** or **18:45 IST**):
  1. `AngelDataProvider` automatically logs in with TOTP and fetches today's full 5-minute candles from `09:00 IST`.
  2. `StrategyEngine.reconcile_and_backfill_late_start()` reconstructs the exact Morning (`09:00–09:15`) or COMEX (`17:00–17:30` EDT / `18:00–18:30` EST) Opening Range.
  3. It reconciles any open positions/orders on Zerodha, re-attaches missing stop-losses, runs a 1-time catch-up scan within the 0.15% drift guard, and immediately trades all fresh setups for the remainder of the session!
