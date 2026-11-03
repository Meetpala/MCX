"""APEX-3: Conservative NSE Cash-Equity Intraday Trading System.

This system operates through Zerodha Kite Connect v3 for NSE cash equities
using MIS (Margin Intraday Squareoff) positions only.

WARNING: Intraday trading, leverage, stop orders, and short selling can
lose money quickly. A stop trigger does not guarantee an execution price
or a fill. This system makes no promise of returns.

The system starts in paper mode by default and cannot send real orders
unless explicitly enabled with all safety checks passing.
"""

__version__ = "0.1.0"
__strategy__ = "APEX-3"
