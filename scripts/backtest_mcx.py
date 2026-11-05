"""Reproducible MCX backtest that drives the *live* signal / stop / ratchet / fee code on cached Angel One 5m data.

Usage:  python scripts/backtest_mcx.py [--sessions both|evening|morning] [--max-trades N] [--target-r R]
Conservative fills: entry at next-bar open + slippage ticks, stop checked before target inside a bar,
ratchet updates apply from the *next* bar, full statutory charges per round trip, 1 lot, no look-ahead.
"""
from __future__ import annotations
import argparse, pickle, sys, statistics as st
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
class Trade:
    day: date; session: str; side: str; entry: float; exit: float; stop0: float
    r_rs: float; gross: float; charges: float; reason: str; playbook: str; max_r: float

    @property
    def net(self): return self.gross - self.charges
    @property
    def r_net(self): return self.net / self.r_rs if self.r_rs else 0.0


def session_windows(d: date):
    dst = is_us_dst(d)
    com_start = time(17, 0) if dst else time(18, 0)
    com_or_end = time(17, 30) if dst else time(18, 30)
    return {
        "MORNING": dict(start=time(9, 0), or_end=time(9, 15), entry_end=time(14, 0), flat=time(16, 45)),
        "COMEX": dict(start=com_start, or_end=com_or_end, entry_end=time(21, 45) if dst else time(22, 30), flat=time(23, 15)),
    }


def regime_for(candles):
    closes = [c.close for c in candles]
    if len(closes) < 10:
        return MarketRegime.NEUTRAL
    ema = compute_ema(closes, 20)
    e = ema[-1] if ema else sum(closes) / len(closes)
    vwap, _ = calculate_session_anchored_vwap(candles)
    p = closes[-1]
    return MarketRegime.BULLISH if (p > e and p > vwap) else MarketRegime.BEARISH if (p < e and p < vwap) else MarketRegime.NEUTRAL


def run(args):
    data = pickle.load(open(CACHE, "rb"))
    root = "SILVERMIC"
    sym = data[root]["symbol"]
    candles = [c for c in data[root]["candles"] if c.is_complete]
    sib_all = data.get("GOLDGUINEA", {}).get("candles", [])
    mult = get_mcx_lot_multiplier(root)
    tick = 1.0
    inst = InstrumentInfo(symbol=sym, exchange="MCX", tick_size=tick, lot_multiplier=mult)
    stops_cfg = StopsConfig(trail_enabled=True, target_r=args.target_r)
    sig_cfg = SignalConfig(max_spread_bps=25.0)
    min_sp, max_sp = args.min_stop_pct / 100, args.max_stop_pct / 100

    by_day = defaultdict(list)
    for c in candles:
        by_day[c.timestamp.astimezone(IST).date()].append(c)
    days = sorted(by_day)
    trades: list[Trade] = []

    for di, d in enumerate(days):
        if d.weekday() >= 5:
            continue
        win = session_windows(d)
        for sname, w in win.items():
            if args.sessions == "evening" and sname != "COMEX": continue
            if args.sessions == "morning" and sname != "MORNING": continue
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
                if t >= w["entry_end"] or n_taken >= args.max_trades or losses >= args.max_losses:
                    break
                hist = prior + bars[: i + 1]
                today = bars[: i + 1]
                sib = [c for c in sib_all if c.timestamp <= bar.timestamp][-len(hist):] if sib_all else None
                q = Quote(symbol=sym, exchange="MCX", last_price=bar.close, bid=bar.close - tick, ask=bar.close + tick,
                          volume=bar.volume, timestamp=bar.timestamp + timedelta(minutes=5), is_fresh=True)
                ev = evaluate_aurum_signal(root, hist, hist, orr, regime_for(today), inst, q, sig_cfg,
                                           is_entry_window=True, sibling_candles_5m=sib)
                if ev.decision not in (Decision.PROPOSED_LONG, Decision.PROPOSED_SHORT):
                    i += 1; continue
                side = Side.BUY if ev.decision == Decision.PROPOSED_LONG else Side.SELL
                nxt = bars[i + 1]
                entry = nxt.open + (args.slip_ticks * tick if side == Side.BUY else -args.slip_ticks * tick)
                closes = [c.close for c in hist]; highs = [c.high for c in hist]; lows = [c.low for c in hist]
                atr_s = compute_atr(highs, lows, closes, period=14)
                atr = atr_s[-1] if atr_s else (bar.high - bar.low)
                lb = hist[-5:]
                stop0 = calculate_initial_stop(side, entry, min(c.low for c in lb), max(c.high for c in lb), atr, stops_cfg, tick)
                sp = abs(entry - stop0) / entry
                if sp < min_sp or sp > max_sp:
                    i += 1; continue
                tgt = calculate_target_price(side, entry, stop0, max(args.target_r, 3.5) if args.target_r else 3.5, tick)
                R = abs(entry - stop0) * mult
                pos = PositionState(symbol=sym, side=side, quantity=1, average_entry=entry, initial_stop_price=stop0,
                                    protective_stop_price=stop0, best_favorable_price=entry, hwm_price=entry,
                                    r_value=abs(entry - stop0))
                stop, exit_px, reason, max_r, j = stop0, None, "FLAT", 0.0, i + 1
                while j < len(bars):
                    b = bars[j]
                    hit_stop = b.low <= stop if side == Side.BUY else b.high >= stop
                    hit_tgt = b.high >= tgt if side == Side.BUY else b.low <= tgt
                    if hit_stop:  # stop first = conservative
                        exit_px = stop - (args.slip_ticks * tick if side == Side.BUY else -args.slip_ticks * tick)
                        reason = "STOP" if stop == stop0 else "TRAIL"; break
                    if hit_tgt:
                        exit_px, reason = tgt, "TARGET"; break
                    fav = (b.high - entry) if side == Side.BUY else (entry - b.low)
                    max_r = max(max_r, fav / abs(entry - stop0))
                    sub = bars[: j + 1]
                    ev2 = evaluate_progressive_ratchet(pos, b.close, hist[-1:] and (prior + sub), atr, stops_cfg, tick)
                    pos.hwm_price = ev2.hwm_price; pos.best_favorable_price = ev2.best_favorable_price
                    if ev2.should_modify and ((ev2.proposed_stop > stop) if side == Side.BUY else (ev2.proposed_stop < stop)):
                        stop = ev2.proposed_stop; pos.protective_stop_price = stop; pos.modification_count += 1
                    j += 1
                if exit_px is None:
                    exit_px = bars[-1].close
                gross = ((exit_px - entry) if side == Side.BUY else (entry - exit_px)) * mult
                bp, sp_ = (entry, exit_px) if side == Side.BUY else (exit_px, entry)
                ch = calculate_mcx_intraday_charges(bp, sp_, 1, mult)
                ch = ch if isinstance(ch, (int, float)) else getattr(ch, "total", 0.0)
                trades.append(Trade(d, sname, side.value, entry, exit_px, stop0, R, gross, ch, reason, ev.playbook, max_r))
                n_taken += 1
                if gross - ch < 0: losses += 1
                i = max(j, i + 1) + args.cooldown
    report(trades, args)
    return trades


def report(trades, args):
    print(f"\n=== {args.label} | sessions={args.sessions} max_trades={args.max_trades} target_r={args.target_r} "
          f"stop%=[{args.min_stop_pct},{args.max_stop_pct}] ===")
    if not trades:
        print("no trades"); return
    def block(ts, name):
        if not ts: print(f"{name}: no trades"); return
        net = sum(t.net for t in ts); gp = sum(t.gross for t in ts if t.gross > 0); gl = -sum(t.gross for t in ts if t.gross < 0)
        wins = [t for t in ts if t.net > 0]
        ch = sum(t.charges for t in ts)
        eq = peak = dd = 0.0
        for t in ts:
            eq += t.net; peak = max(peak, eq); dd = max(dd, peak - eq)
        print(f"{name:10s} n={len(ts):3d} win={len(wins)/len(ts)*100:4.1f}% gross=Rs{sum(t.gross for t in ts):9,.0f} "
              f"charges=Rs{ch:8,.0f} NET=Rs{net:9,.0f} PF={(gp/gl if gl else 0):.2f} avgR={st.mean(t.r_net for t in ts):+.2f} "
              f"avgRisk=Rs{st.mean(t.r_rs for t in ts):.0f} maxDD=Rs{dd:,.0f}")
    block(trades, "ALL")
    for s in ("MORNING", "COMEX"): block([t for t in trades if t.session == s], s)
    for p in sorted({t.playbook for t in trades}): block([t for t in trades if t.playbook == p], p)
    for sd in ("BUY", "SELL"): block([t for t in trades if t.side == sd], sd)
    for r in sorted({t.reason for t in trades}): block([t for t in trades if t.reason == r], r)
    cut = sorted({t.day for t in trades})[int(len({t.day for t in trades}) * 0.6)]
    block([t for t in trades if t.day < cut], f"IN<{cut}"); block([t for t in trades if t.day >= cut], f"OUT>={cut}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="both")
    ap.add_argument("--max-trades", type=int, default=2)
    ap.add_argument("--max-losses", type=int, default=2)
    ap.add_argument("--target-r", type=float, default=3.5)
    ap.add_argument("--min-stop-pct", type=float, default=0.18)
    ap.add_argument("--max-stop-pct", type=float, default=0.65)
    ap.add_argument("--slip-ticks", type=float, default=2.0)
    ap.add_argument("--cooldown", type=int, default=2)
    ap.add_argument("--label", default="run")
    run(ap.parse_args())

