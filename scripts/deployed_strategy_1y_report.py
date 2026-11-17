"""1-Year Audit Report for the Exact Deployed AURUM-APEX MCX Strategy (SILVERMIC).

Replays the exact live production code (`evaluate_aurum_signal`, `calculate_initial_stop`,
`evaluate_progressive_ratchet`, `calculate_mcx_intraday_charges`) over the past 1 year of
real Angel One 5-minute candles (`data/mcx_5m_cache.pkl`), strictly with:
- Zero look-ahead (daily candles only include completed days strictly prior to trade day)
- Next-bar open entry + 2-tick slippage; stop checked before target inside bars
- Exact MCX statutory charges (brokerage, CTT, exchange txn, SEBI, GST, stamp duty)
"""
from __future__ import annotations

import bisect
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apex3.config import SignalConfig, StopsConfig
from apex3.indicators.atr import compute_atr
from apex3.indicators.ema import compute_ema
from apex3.models import Candle, Decision, InstrumentInfo, PositionState, Quote, Side
from apex3.strategy.regime import MarketRegime, calculate_session_anchored_vwap
from apex3.strategy.risk import get_mcx_lot_multiplier
from apex3.strategy.signal import build_opening_range, evaluate_aurum_signal, is_us_dst
from apex3.strategy.sizer import calculate_mcx_intraday_charges
from apex3.strategy.trail import (
    calculate_initial_stop,
    calculate_target_price,
    evaluate_progressive_ratchet,
)

IST = ZoneInfo("Asia/Kolkata")
CACHE = ROOT / "data" / "mcx_5m_cache.pkl"


@dataclass
class TradeRow:
    day: date
    entry_time: time
    exit_time: time
    session: str
    side: str
    playbook: str
    entry_px: float
    exit_px: float
    stop0: float
    r_rs: float
    gross: float
    charges: float
    reason: str
    max_r: float

    @property
    def net(self) -> float:
        return self.gross - self.charges

    @property
    def r_net(self) -> float:
        return self.net / self.r_rs if self.r_rs else 0.0


def build_daily_candles(candles_5m: list[Candle]) -> dict[date, list[Candle]]:
    """Build completed daily candles and return a map: day -> completed daily candles strictly BEFORE day."""
    by_day: dict[date, list[Candle]] = defaultdict(list)
    for c in candles_5m:
        if getattr(c, "is_complete", True):
            by_day[c.timestamp.astimezone(IST).date()].append(c)

    sorted_days = sorted(by_day)
    daily_seq: list[tuple[date, Candle]] = []
    for d in sorted_days:
        bars = by_day[d]
        daily_seq.append(
            (
                d,
                Candle(
                    timestamp=datetime.combine(d, time(23, 55), tzinfo=IST),
                    open=bars[0].open,
                    high=max(b.high for b in bars),
                    low=min(b.low for b in bars),
                    close=bars[-1].close,
                    volume=sum(b.volume for b in bars),
                    is_complete=True,
                ),
            )
        )

    prior_daily_map: dict[date, list[Candle]] = {}
    acc: list[Candle] = []
    for d, dc in daily_seq:
        prior_daily_map[d] = list(acc)
        acc.append(dc)
    return prior_daily_map


def replay_deployed(
    root: str = "SILVERMIC",
    use_daily_macro_gate: bool = True,
    side_filter: str = "both",
    max_stop_pct: float = 0.45,
    min_stop_pct: float = 0.18,
    target_r: float = 4.5,
    comex_delay: bool = True,
) -> tuple[list[TradeRow], date, date, int]:
    data = pickle.load(open(CACHE, "rb"))
    sym = data[root]["symbol"]
    candles = [c for c in data[root]["candles"] if c.is_complete]
    sibling_root = "GOLDGUINEA" if "SILVER" in root else "SILVERMIC"
    sib_all = data.get(sibling_root, {}).get("candles", [])
    sib_ts = [c.timestamp for c in sib_all]
    mult = get_mcx_lot_multiplier(root)
    tick = 1.0
    inst = InstrumentInfo(symbol=sym, exchange="MCX", tick_size=tick, lot_multiplier=mult)
    stops_cfg = StopsConfig(trail_enabled=True, target_r=target_r)
    sig_cfg = SignalConfig(max_spread_bps=25.0)
    min_sp, max_sp = min_stop_pct / 100.0, max_stop_pct / 100.0

    prior_daily_map = build_daily_candles(candles)

    by_day: dict[date, list[Candle]] = defaultdict(list)
    for c in candles:
        by_day[c.timestamp.astimezone(IST).date()].append(c)
    days = sorted(by_day)
    trades: list[TradeRow] = []

    for di, d in enumerate(days):
        if d.weekday() >= 5:
            continue
        dst = is_us_dst(d)
        com_start = time(17, 0) if dst else time(18, 0)
        com_or_end = time(17, 30) if dst else time(18, 30)
        com_entry_start = (time(18, 45) if dst else time(19, 30)) if comex_delay else com_or_end
        windows = {
            "MORNING": dict(
                start=time(9, 0),
                or_end=time(9, 15),
                entry_start=time(9, 15),
                entry_end=time(14, 0),
                flat=time(16, 45),
            ),
            "COMEX": dict(
                start=com_start,
                or_end=com_or_end,
                entry_start=com_entry_start,
                entry_end=time(21, 45) if dst else time(22, 30),
                flat=time(23, 15),
            ),
        }
        d_candles = prior_daily_map.get(d, []) if use_daily_macro_gate else None

        for sname, w in windows.items():
            bars = [c for c in by_day[d] if w["start"] <= c.timestamp.astimezone(IST).time() < w["flat"]]
            or_bars = [c for c in bars if c.timestamp.astimezone(IST).time() < w["or_end"]]
            if len(or_bars) < 3:
                continue
            orr = build_opening_range(root, or_bars, session_date=d, session_type=sname)
            if not orr.is_complete:
                continue
            prior = [c for dd in days[max(0, di - 2) : di] for c in by_day[dd]][-60:]
            n_taken, losses, i = 0, 0, len(or_bars)

            while i < len(bars) - 1:
                bar = bars[i]
                t = bar.timestamp.astimezone(IST).time()
                if t < w["entry_start"]:
                    i += 1
                    continue
                if t >= w["entry_end"] or n_taken >= 2 or losses >= 2:
                    break
                hist = prior + bars[: i + 1]
                today = bars[: i + 1]
                if sib_all:
                    pos_s = bisect.bisect_right(sib_ts, bar.timestamp)
                    sib = sib_all[max(0, pos_s - len(hist)) : pos_s]
                else:
                    sib = None
                q = Quote(
                    symbol=sym,
                    exchange="MCX",
                    last_price=bar.close,
                    bid=bar.close - tick,
                    ask=bar.close + tick,
                    volume=bar.volume,
                    timestamp=bar.timestamp + timedelta(minutes=5),
                    is_fresh=True,
                )

                closes_today = [c.close for c in today]
                e20 = compute_ema(closes_today, 20)[-1] if len(closes_today) >= 20 else bar.close
                vwap, _ = calculate_session_anchored_vwap(today)
                regime = (
                    MarketRegime.BULLISH
                    if (bar.close > e20 and bar.close > vwap)
                    else MarketRegime.BEARISH
                    if (bar.close < e20 and bar.close < vwap)
                    else MarketRegime.NEUTRAL
                )

                ev = evaluate_aurum_signal(
                    root,
                    hist,
                    hist,
                    orr,
                    regime,
                    inst,
                    q,
                    sig_cfg,
                    is_entry_window=True,
                    sibling_candles_5m=sib,
                    daily_candles=d_candles,
                )
                if ev.decision not in (Decision.PROPOSED_LONG, Decision.PROPOSED_SHORT):
                    i += 1
                    continue

                side = Side.BUY if ev.decision == Decision.PROPOSED_LONG else Side.SELL
                if side_filter != "both" and side.value.lower() != side_filter.lower():
                    i += 1
                    continue

                nxt = bars[i + 1]
                entry = nxt.open + (2.0 * tick if side == Side.BUY else -2.0 * tick)
                closes = [c.close for c in hist]
                highs = [c.high for c in hist]
                lows = [c.low for c in hist]

                atr_s = compute_atr(highs, lows, closes, period=14)
                atr = atr_s[-1] if atr_s else (bar.high - bar.low)
                lb = hist[-5:]
                stop0 = calculate_initial_stop(
                    side, entry, min(c.low for c in lb), max(c.high for c in lb), atr, stops_cfg, tick
                )
                sp = abs(entry - stop0) / entry
                if sp < min_sp or sp > max_sp:
                    i += 1
                    continue

                tgt = calculate_target_price(side, entry, stop0, target_r, tick)
                R = abs(entry - stop0) * mult
                pos = PositionState(
                    symbol=sym,
                    side=side,
                    quantity=1,
                    average_entry=entry,
                    initial_stop_price=stop0,
                    protective_stop_price=stop0,
                    best_favorable_price=entry,
                    hwm_price=entry,
                    r_value=abs(entry - stop0),
                )
                stop, exit_px, exit_t, reason, max_r, j = stop0, None, bars[-1].timestamp.astimezone(IST).time(), "FLAT", 0.0, i + 1
                while j < len(bars):
                    b = bars[j]
                    hit_stop = b.low <= stop if side == Side.BUY else b.high >= stop
                    hit_tgt = b.high >= tgt if side == Side.BUY else b.low <= tgt
                    if hit_stop:
                        exit_px = stop - (2.0 * tick if side == Side.BUY else -2.0 * tick)
                        exit_t = b.timestamp.astimezone(IST).time()
                        reason = "STOP" if stop == stop0 else "TRAIL"
                        break
                    if hit_tgt:
                        exit_px, reason = tgt, "TARGET"
                        exit_t = b.timestamp.astimezone(IST).time()
                        break
                    fav = (b.high - entry) if side == Side.BUY else (entry - b.low)
                    max_r = max(max_r, fav / abs(entry - stop0))
                    sub = bars[: j + 1]
                    ev2 = evaluate_progressive_ratchet(pos, b.close, prior + sub, atr, stops_cfg, tick)
                    pos.hwm_price = ev2.hwm_price
                    pos.best_favorable_price = ev2.best_favorable_price
                    if ev2.should_modify and (
                        (ev2.proposed_stop > stop) if side == Side.BUY else (ev2.proposed_stop < stop)
                    ):
                        stop = ev2.proposed_stop
                        pos.protective_stop_price = stop
                        pos.modification_count += 1
                    j += 1

                if exit_px is None:
                    exit_px = bars[-1].close
                    exit_t = bars[-1].timestamp.astimezone(IST).time()

                gross = ((exit_px - entry) if side == Side.BUY else (entry - exit_px)) * mult
                bp, sp_ = (entry, exit_px) if side == Side.BUY else (exit_px, entry)
                ch = calculate_mcx_intraday_charges(bp, sp_, 1, mult)
                ch = float(ch if isinstance(ch, (int, float)) else getattr(ch, "total", 0.0))
                trades.append(
                    TradeRow(
                        day=d,
                        entry_time=nxt.timestamp.astimezone(IST).time(),
                        exit_time=exit_t,
                        session=sname,
                        side=side.value,
                        playbook=ev.playbook or "TREND",
                        entry_px=entry,
                        exit_px=exit_px,
                        stop0=stop0,
                        r_rs=R,
                        gross=gross,
                        charges=ch,
                        reason=reason,
                        max_r=max_r,
                    )
                )
                n_taken += 1
                if gross - ch < 0:
                    losses += 1
                i = max(j, i + 1) + 2

    return trades, days[0], days[-1], len(candles)


def summarize(label: str, trades: list[TradeRow], print_ledger: bool = False) -> None:
    print(f"\n==========================================================================================")
    print(f" {label}")
    print(f"==========================================================================================")
    if not trades:
        print("No trades triggered.")
        return

    gross = sum(t.gross for t in trades)
    charges = sum(t.charges for t in trades)
    net = sum(t.net for t in trades)
    wins = [t for t in trades if t.net > 0]
    losses = [t for t in trades if t.net <= 0]
    gp = sum(t.gross for t in trades if t.gross > 0)
    gl = -sum(t.gross for t in trades if t.gross < 0)
    pf = gp / gl if gl > 0 else 999.0
    eq = peak = max_dd = 0.0
    for t in trades:
        eq += t.net
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    print(f"Total Trades    : {len(trades)} ({len(wins)}W / {len(losses)}L | Win Rate: {len(wins)/len(trades)*100:.1f}%)")
    print(f"Gross P&L       : Rs {gross:+,.2f}")
    print(f"Statutory Fees  : Rs {charges:,.2f}")
    print(f"Net Realized PnL: Rs {net:+,.2f}")
    print(f"Profit Factor   : {pf:.2f}")
    print(f"Max Drawdown    : Rs {max_dd:,.2f}")
    print(f"Return on 30k   : {net/30000.0*100:+.2f}%")

    print("\n--- Breakdown by Side ---")
    for sd in ("BUY", "SELL"):
        st = [t for t in trades if t.side == sd]
        if not st:
            print(f"  {sd:4s} : 0 trades")
            continue
        sw = [t for t in st if t.net > 0]
        sgp = sum(t.gross for t in st if t.gross > 0)
        sgl = -sum(t.gross for t in st if t.gross < 0)
        spf = sgp / sgl if sgl > 0 else 999.0
        print(
            f"  {sd:4s} : n={len(st):2d} | Win={len(sw)/len(st)*100:4.1f}% | "
            f"Gross=Rs {sum(t.gross for t in st):+9,.2f} | Fees=Rs {sum(t.charges for t in st):7,.2f} | "
            f"Net=Rs {sum(t.net for t in st):+9,.2f} | PF={spf:.2f}"
        )

    print("\n--- Breakdown by Session ---")
    for sess in ("MORNING", "COMEX"):
        st = [t for t in trades if t.session == sess]
        if not st:
            continue
        sw = [t for t in st if t.net > 0]
        print(
            f"  {sess:7s} : n={len(st):2d} | Win={len(sw)/len(st)*100:4.1f}% | "
            f"Gross=Rs {sum(t.gross for t in st):+9,.2f} | Net=Rs {sum(t.net for t in st):+9,.2f}"
        )

    print("\n--- Breakdown by Exit Reason ---")
    for r in sorted({t.reason for t in trades}):
        st = [t for t in trades if t.reason == r]
        sw = [t for t in st if t.net > 0]
        print(
            f"  {r:6s} : n={len(st):2d} | Win={len(sw)/len(st)*100:4.1f}% | "
            f"Gross=Rs {sum(t.gross for t in st):+9,.2f} | Net=Rs {sum(t.net for t in st):+9,.2f}"
        )

    print("\n--- Month-by-Month Net P&L ---")
    by_m: dict[str, list[TradeRow]] = defaultdict(list)
    for t in trades:
        by_m[t.day.strftime("%Y-%m")].append(t)
    cum = 0.0
    for m in sorted(by_m):
        mt = by_m[m]
        mw = [t for t in mt if t.net > 0]
        mnet = sum(t.net for t in mt)
        cum += mnet
        print(
            f"  {m} | Trades: {len(mt):2d} ({len(mw):2d}W/{len(mt)-len(mw):2d}L) | "
            f"Gross: Rs {sum(t.gross for t in mt):+8,.2f} | Net: Rs {mnet:+8,.2f} | Cum: Rs {cum:+8,.2f}"
        )

    if print_ledger:
        print("\n--- Complete Trade-by-Trade Audit Ledger ---")
        print(
            f"{'#':>2} {'Date':10} {'Sess':7} {'Side':4} {'Playbook':13} {'InTime':5} {'Out':5} "
            f"{'Entry':>9} {'Exit':>9} {'Reason':6} {'Gross':>8} {'Fees':>6} {'Net':>8} {'CumNet':>9}"
        )
        cum_t = 0.0
        for idx, t in enumerate(trades, 1):
            cum_t += t.net
            print(
                f"{idx:2d} {t.day.isoformat():10} {t.session:7} {t.side:4} {t.playbook:13} "
                f"{t.entry_time.strftime('%H:%M'):5} {t.exit_time.strftime('%H:%M'):5} "
                f"{t.entry_px:9,.1f} {t.exit_px:9,.1f} {t.reason:6} "
                f"{t.gross:+8,.0f} {t.charges:6,.0f} {t.net:+8,.0f} {cum_t:+9,.0f}"
            )


if __name__ == "__main__":
    t_exact, d0, d1, n_bars = replay_deployed("SILVERMIC", use_daily_macro_gate=True, side_filter="both")
    print(f"Dataset: SILVERMIC | Range: {d0} -> {d1} | Completed 5m Candles: {n_bars:,}")
    summarize("1) EXACT DEPLOYED CODE (Gate 4B Daily 50-EMA ON | Both Sides)", t_exact, print_ledger=True)

    t_nodaily, _, _, _ = replay_deployed("SILVERMIC", use_daily_macro_gate=False, side_filter="both")
    summarize("2) DEPLOYED CORE ENGINE (5m EMA50 HTF | No Daily 50-EMA Gate | Both Sides)", t_nodaily, print_ledger=True)

    t_sell, _, _, _ = replay_deployed("SILVERMIC", use_daily_macro_gate=False, side_filter="sell")
    summarize("3) DEPLOYED CORE ENGINE (SELL-ONLY / Bearish Macro Regime)", t_sell, print_ledger=False)
