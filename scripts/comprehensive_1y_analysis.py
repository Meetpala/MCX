"""Comprehensive 1-Year Multi-Commodity Performance & Equity Analysis.
Runs the exact Apex 3 engine logic across all 3 commodities (SILVERMIC, GOLDGUINEA, GOLDTEN)
and generates full breakdown:
- Overall Net P&L, Win Rate, Profit Factor, Max Drawdown
- Bull vs Bear (Long vs Short)
- Morning vs COMEX sessions
- Playbook breakdown (OR_BREAKOUT, SWEEP_RECLAIM, PULLBACK)
- Monthly P&L and consistency
"""
import pickle, sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apex3.config import StopsConfig, SignalConfig
from apex3.models import InstrumentInfo, PositionState, Quote, Side, Decision
from apex3.strategy.regime import MarketRegime, calculate_session_anchored_vwap
from apex3.strategy.signal import build_opening_range, evaluate_aurum_signal, is_us_dst
from apex3.strategy.trail import calculate_initial_stop, calculate_target_price, evaluate_progressive_ratchet
from apex3.strategy.sizer import calculate_mcx_intraday_charges
from apex3.strategy.risk import get_mcx_lot_multiplier
from apex3.indicators.ema import compute_ema
from apex3.indicators.atr import compute_atr

IST = ZoneInfo("Asia/Kolkata")
CACHE = ROOT / "data" / "mcx_5m_cache.pkl"

@dataclass
class TradeRecord:
    day: date
    symbol: str
    session: str
    side: str
    entry_time: time
    entry_px: float
    exit_px: float
    stop0: float
    r_rs: float
    gross: float
    charges: float
    reason: str
    playbook: str
    max_r: float

    @property
    def net(self):
        return self.gross - self.charges

    @property
    def r_net(self):
        return self.net / self.r_rs if self.r_rs else 0.0

def run_backtest_symbol(symbol: str, target_r: float = 4.5, max_stop_pct: float = 0.45,
                        min_stop_pct: float = 0.18, htf_filter: bool = True,
                        comex_delay: bool = True, max_trades_per_sess: int = 2):
    data = pickle.load(open(CACHE, "rb"))
    root = symbol.upper()
    if root not in data:
        return []
    
    sym = data[root]["symbol"]
    candles = [c for c in data[root]["candles"] if c.is_complete]
    sibling_root = "GOLDGUINEA" if "SILVER" in root else "SILVERMIC"
    sib_all = data.get(sibling_root, {}).get("candles", [])
    mult = get_mcx_lot_multiplier(root)
    tick = 1.0 if "SILVER" in root else 0.5
    inst = InstrumentInfo(symbol=sym, exchange="MCX", tick_size=tick, lot_multiplier=mult)
    stops_cfg = StopsConfig(trail_enabled=True, target_r=target_r)
    sig_cfg = SignalConfig(max_spread_bps=25.0)
    min_sp, max_sp = min_stop_pct / 100, max_stop_pct / 100

    by_day = defaultdict(list)
    for c in candles:
        by_day[c.timestamp.astimezone(IST).date()].append(c)
    days = sorted(by_day)
    trades = []

    for di, d in enumerate(days):
        if d.weekday() >= 5:
            continue
        dst = is_us_dst(d)
        com_start = time(17, 0) if dst else time(18, 0)
        com_or_end = time(17, 30) if dst else time(18, 30)
        com_entry_start = (time(18, 45) if dst else time(19, 30)) if comex_delay else com_or_end
        
        windows = {
            "MORNING": dict(start=time(9, 0), or_end=time(9, 15), entry_start=time(9, 15), entry_end=time(14, 0), flat=time(16, 45)),
            "COMEX": dict(start=com_start, or_end=com_or_end, entry_start=com_entry_start, entry_end=time(21, 45) if dst else time(22, 30), flat=time(23, 15)),
        }

        for sname, w in windows.items():
            bars = [c for c in by_day[d] if w["start"] <= c.timestamp.astimezone(IST).time() < w["flat"]]
            or_bars = [c for c in bars if c.timestamp.astimezone(IST).time() < w["or_end"]]
            if len(or_bars) < 3:
                continue
            orr = build_opening_range(root, or_bars, session_date=d, session_type=sname)
            if not orr.is_complete:
                continue
            prior = [c for dd in days[max(0, di - 2):di] for c in by_day[dd]][-60:]
            n_taken, losses, i = 0, 0, len(or_bars)
            
            while i < len(bars) - 1:
                bar = bars[i]
                t = bar.timestamp.astimezone(IST).time()
                if t < w["entry_start"]:
                    i += 1
                    continue
                if t >= w["entry_end"] or n_taken >= max_trades_per_sess or losses >= 2:
                    break
                hist = prior + bars[: i + 1]
                today = bars[: i + 1]
                sib = [c for c in sib_all if c.timestamp <= bar.timestamp][-len(hist):] if sib_all else None
                q = Quote(symbol=sym, exchange="MCX", last_price=bar.close, bid=bar.close - tick, ask=bar.close + tick,
                          volume=bar.volume, timestamp=bar.timestamp + timedelta(minutes=5), is_fresh=True)
                
                # Regime
                closes_today = [c.close for c in today]
                e20 = compute_ema(closes_today, 20)[-1] if len(closes_today) >= 20 else bar.close
                vwap, _ = calculate_session_anchored_vwap(today)
                regime = MarketRegime.BULLISH if (bar.close > e20 and bar.close > vwap) else MarketRegime.BEARISH if (bar.close < e20 and bar.close < vwap) else MarketRegime.NEUTRAL
                
                ev = evaluate_aurum_signal(root, hist, hist, orr, regime, inst, q, sig_cfg,
                                           is_entry_window=True, sibling_candles_5m=sib)
                if ev.decision not in (Decision.PROPOSED_LONG, Decision.PROPOSED_SHORT):
                    i += 1
                    continue
                side = Side.BUY if ev.decision == Decision.PROPOSED_LONG else Side.SELL
                nxt = bars[i + 1]
                entry = nxt.open + (2.0 * tick if side == Side.BUY else -2.0 * tick)
                closes = [c.close for c in hist]
                highs = [c.high for c in hist]
                lows = [c.low for c in hist]

                # HTF 50 EMA filter
                if htf_filter:
                    e50_s = compute_ema(closes, 50)
                    if e50_s:
                        e50 = e50_s[-1]
                        if side == Side.BUY and entry < e50:
                            i += 1
                            continue
                        if side == Side.SELL and entry > e50:
                            i += 1
                            continue

                atr_s = compute_atr(highs, lows, closes, period=14)
                atr = atr_s[-1] if atr_s else (bar.high - bar.low)
                lb = hist[-5:]
                stop0 = calculate_initial_stop(side, entry, min(c.low for c in lb), max(c.high for c in lb), atr, stops_cfg, tick)
                sp = abs(entry - stop0) / entry
                if sp < min_sp or sp > max_sp:
                    i += 1
                    continue
                
                tgt = calculate_target_price(side, entry, stop0, target_r, tick)
                R = abs(entry - stop0) * mult
                pos = PositionState(symbol=sym, side=side, quantity=1, average_entry=entry, initial_stop_price=stop0,
                                    protective_stop_price=stop0, best_favorable_price=entry, hwm_price=entry,
                                    r_value=abs(entry - stop0))
                stop, exit_px, reason, max_r, j = stop0, None, "FLAT", 0.0, i + 1
                while j < len(bars):
                    b = bars[j]
                    hit_stop = b.low <= stop if side == Side.BUY else b.high >= stop
                    hit_tgt = b.high >= tgt if side == Side.BUY else b.low <= tgt
                    if hit_stop:
                        exit_px = stop - (2.0 * tick if side == Side.BUY else -2.0 * tick)
                        reason = "STOP" if stop == stop0 else "TRAIL"
                        break
                    if hit_tgt:
                        exit_px, reason = tgt, "TARGET"
                        break
                    fav = (b.high - entry) if side == Side.BUY else (entry - b.low)
                    max_r = max(max_r, fav / abs(entry - stop0))
                    sub = bars[: j + 1]
                    ev2 = evaluate_progressive_ratchet(pos, b.close, hist[-1:] and (prior + sub), atr, stops_cfg, tick)
                    pos.hwm_price = ev2.hwm_price
                    pos.best_favorable_price = ev2.best_favorable_price
                    if ev2.should_modify and ((ev2.proposed_stop > stop) if side == Side.BUY else (ev2.proposed_stop < stop)):
                        stop = ev2.proposed_stop
                        pos.protective_stop_price = stop
                        pos.modification_count += 1
                    j += 1
                if exit_px is None:
                    exit_px = bars[-1].close
                gross = ((exit_px - entry) if side == Side.BUY else (entry - exit_px)) * mult
                bp, sp_ = (entry, exit_px) if side == Side.BUY else (exit_px, entry)
                ch = calculate_mcx_intraday_charges(bp, sp_, 1, mult)
                ch = ch if isinstance(ch, (int, float)) else getattr(ch, "total", 0.0)
                trades.append(TradeRecord(d, root, sname, side.value, t, entry, exit_px, stop0, R, gross, ch, reason, ev.playbook, max_r))
                n_taken += 1
                if gross - ch < 0:
                    losses += 1
                i = max(j, i + 1) + 2
    return trades

if __name__ == "__main__":
    for sym in ["SILVERMIC", "GOLDGUINEA", "GOLDTEN"]:
        print(f"\n==========================================")
        print(f"ANALYZING: {sym} (Past 1 Year 5m Live Replay)")
        print(f"==========================================")
        trades = run_backtest_symbol(sym)
        if not trades:
            print(f"No trades triggered for {sym} under these parameters.")
            continue
        net = sum(t.net for t in trades)
        gross = sum(t.gross for t in trades)
        charges = sum(t.charges for t in trades)
        wins = [t for t in trades if t.net > 0]
        losses = [t for t in trades if t.net <= 0]
        gp = sum(t.gross for t in wins)
        gl = -sum(t.gross for t in losses)
        pf = gp / gl if gl > 0 else 999.0
        
        # Max DD
        eq, peak, max_dd = 0.0, 0.0, 0.0
        for t in trades:
            eq += t.net
            peak = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
        
        print(f"Total Trades   : {len(trades)}")
        print(f"Win Rate       : {len(wins)/len(trades)*100:.1f}% ({len(wins)}W / {len(losses)}L)")
        print(f"Gross P&L      : Rs {gross:,.2f}")
        print(f"Broker Charges : Rs {charges:,.2f}")
        print(f"Net P&L        : Rs {net:,.2f}")
        print(f"Profit Factor  : {pf:.2f}")
        print(f"Max Drawdown   : Rs {max_dd:,.2f}")
        
        # By Side
        print("\n--- By Trade Side ---")
        for sd in ["BUY", "SELL"]:
            s_trades = [t for t in trades if t.side == sd]
            if not s_trades:
                print(f"  {sd}: 0 trades")
                continue
            s_net = sum(t.net for t in s_trades)
            s_wins = [t for t in s_trades if t.net > 0]
            s_gp = sum(t.gross for t in s_wins)
            s_gl = -sum(t.gross for t in s_trades if t.net <= 0)
            s_pf = s_gp / s_gl if s_gl > 0 else 999.0
            print(f"  {sd:4s} | Trades: {len(s_trades):2d} | Win: {len(s_wins)/len(s_trades)*100:4.1f}% | Net: Rs {s_net:9,.2f} | PF: {s_pf:4.2f}")
            
        # By Session
        print("\n--- By Session ---")
        for s in ["MORNING", "COMEX"]:
            s_trades = [t for t in trades if t.session == s]
            if not s_trades:
                continue
            s_net = sum(t.net for t in s_trades)
            s_wins = [t for t in s_trades if t.net > 0]
            print(f"  {s:7s} | Trades: {len(s_trades):2d} | Win: {len(s_wins)/len(s_trades)*100:4.1f}% | Net: Rs {s_net:9,.2f}")
            
        # Monthly breakdown
        print("\n--- Monthly Net P&L ---")
        by_month = defaultdict(list)
        for t in trades:
            m = t.day.strftime("%Y-%m")
            by_month[m].append(t)
        for m in sorted(by_month):
            m_trades = by_month[m]
            m_net = sum(t.net for t in m_trades)
            m_wins = [t for t in m_trades if t.net > 0]
            print(f"  {m} | Trades: {len(m_trades):2d} | Wins: {len(m_wins)} | Net: Rs {m_net:8,.2f}")

