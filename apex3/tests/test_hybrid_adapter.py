"""Tests for AngelDataProvider and HybridBrokerAdapter."""
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from apex3.broker.angel_data_provider import AngelDataProvider
from apex3.broker.hybrid_adapter import HybridBrokerAdapter
from apex3.broker.interface import BrokerInterface
from apex3.models import (
    Candle,
    InstrumentInfo,
    MarginInfo,
    OrderMarginEstimate,
    OrderRequest,
    OrderState,
    OrderStatus,
    OrderType,
    PositionState,
    Product,
    Quote,
    Side,
)

IST = ZoneInfo("Asia/Kolkata")


class MockExecutionBroker(BrokerInterface):
    """Mock execution broker standing in for Zerodha."""

    def __init__(self):
        self.placed_orders = []
        self.cancelled_orders = []
        self.modified_orders = []

    def get_margins(self) -> MarginInfo:
        return MarginInfo(equity_available=15000.0, equity_used=0.0, timestamp=datetime.now(IST))

    def estimate_order_margin(self, request: OrderRequest) -> OrderMarginEstimate:
        return OrderMarginEstimate(required_margin=3000.0, charges=20.0, total=3020.0, leverage_available=4.0)

    def place_order(self, request: OrderRequest) -> str:
        self.placed_orders.append(request)
        return "ZERODHA_ORDER_12345"

    def modify_order(self, order_id: str, **kwargs) -> bool:
        self.modified_orders.append((order_id, kwargs))
        return True

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled_orders.append(order_id)
        return True

    def get_order_status(self, order_id: str) -> OrderState:
        return OrderState(
            order_id=order_id,
            request=OrderRequest("RELIANCE", "NSE", Side.BUY, 10, Product.MIS, OrderType.LIMIT, 2500.0),
            status=OrderStatus.FILLED,
            filled_quantity=10,
            pending_quantity=0,
            average_price=2500.0,
            last_updated=datetime.now(IST),
        )

    def get_orders(self) -> list[OrderState]:
        return []

    def get_trades(self) -> list[dict]:
        return []

    def get_positions(self) -> list[PositionState]:
        return []

    def is_session_valid(self) -> bool:
        return True

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Quote:
        raise NotImplementedError("Execution broker should not provide market quotes in hybrid mode")

    def get_historical(self, symbol: str, interval: str, from_dt: datetime, to_dt: datetime, exchange: str = "NSE") -> list[Candle]:
        raise NotImplementedError("Execution broker should not provide historical data in hybrid mode")

    def get_instrument_info(self, symbol: str, exchange: str = "NSE") -> InstrumentInfo:
        raise NotImplementedError("Execution broker should not provide instrument info in hybrid mode")


def test_hybrid_adapter_routing():
    """Verify that market data calls go to Angel One and order calls go strictly to Zerodha."""
    angel_provider = AngelDataProvider(api_key="", client_code="", pin="", totp_secret="")
    zerodha_broker = MockExecutionBroker()

    angel_provider.get_quote = MagicMock(return_value=Quote("RELIANCE", "NSE", 2500.0, 2499.5, 2500.5, 10000, datetime.now(IST), is_fresh=True))
    angel_provider.get_instrument_info = MagicMock(return_value=InstrumentInfo("RELIANCE", "NSE", tick_size=0.05))

    hybrid = HybridBrokerAdapter(data_provider=angel_provider, execution_broker=zerodha_broker)

    # 1. Market Data should route to AngelDataProvider
    quote = hybrid.get_quote("RELIANCE")
    assert quote.symbol == "RELIANCE"
    assert quote.last_price > 0

    inst = hybrid.get_instrument_info("RELIANCE")
    assert inst.symbol == "RELIANCE"
    assert inst.tick_size > 0

    # 2. Margins should route to Zerodha
    margin = hybrid.get_margins()
    assert margin.equity_available == 15000.0

    # 3. Order placement must route EXCLUSIVELY to Zerodha
    req = OrderRequest(
        symbol="RELIANCE",
        exchange="NSE",
        side=Side.BUY,
        quantity=5,
        product=Product.MIS,
        order_type=OrderType.LIMIT,
        price=2500.0,
    )
    order_id = hybrid.place_order(req)
    assert order_id == "ZERODHA_ORDER_12345"
    assert len(zerodha_broker.placed_orders) == 1
    assert zerodha_broker.placed_orders[0].symbol == "RELIANCE"

    # 4. Order cancellation must route EXCLUSIVELY to Zerodha
    res = hybrid.cancel_order("ZERODHA_ORDER_12345")
    assert res is True
    assert "ZERODHA_ORDER_12345" in zerodha_broker.cancelled_orders


def test_hybrid_adapter_get_quotes_batch_and_self_healing_session():
    """Verify HybridBrokerAdapter.get_quotes_batch() delegates to data_provider and is_session_valid self-heals."""
    mock_data_provider = MagicMock()
    mock_exec_broker = MagicMock()
    mock_exec_broker.is_session_valid.return_value = True

    now = datetime.now(IST)
    batch_resp = {
        "TRENT": Quote("TRENT", "NSE", 1000.0, 999.8, 1000.2, 5000, now, is_fresh=True),
        "JSWSTEEL": Quote("JSWSTEEL", "NSE", 900.0, 899.5, 900.5, 8000, now, is_fresh=True),
    }
    mock_data_provider.get_quotes_batch.return_value = batch_resp

    hybrid = HybridBrokerAdapter(data_provider=mock_data_provider, execution_broker=mock_exec_broker)

    # 1. get_quotes_batch routes directly to data_provider
    quotes = hybrid.get_quotes_batch(["TRENT", "JSWSTEEL"], exchange="NSE")
    assert len(quotes) == 2
    assert quotes["TRENT"].last_price == 1000.0
    mock_data_provider.get_quotes_batch.assert_called_once_with(["TRENT", "JSWSTEEL"], exchange="NSE")

    # 2. is_session_valid calls authenticate() if Angel One session drops, and self-heals
    mock_data_provider.is_session_valid.side_effect = [False, True]
    valid = hybrid.is_session_valid()
    assert valid is True
    mock_data_provider.authenticate.assert_called_once()


def test_candle_cache_invalidates_at_five_minute_boundary():
    """Verify that AngelDataProvider invalidates _candle_cache immediately when crossing a 5-minute clock boundary."""
    from unittest.mock import patch

    provider = AngelDataProvider(api_key="", client_code="", pin="", totp_secret="")
    provider._smart_api = MagicMock()

    mock_resp_1 = {
        "status": True,
        "data": [
            ["2026-03-20T09:40:00+05:30", 2500.0, 2510.0, 2495.0, 2505.0, 10000],
            ["2026-03-20T09:45:00+05:30", 2505.0, 2515.0, 2500.0, 2510.0, 12000],
        ],
    }
    mock_resp_2 = {
        "status": True,
        "data": [
            ["2026-03-20T09:40:00+05:30", 2500.0, 2510.0, 2495.0, 2505.0, 10000],
            ["2026-03-20T09:45:00+05:30", 2505.0, 2515.0, 2500.0, 2510.0, 12000],
            ["2026-03-20T09:50:00+05:30", 2510.0, 2525.0, 2508.0, 2520.0, 15000],
        ],
    }
    provider._smart_api.getCandleData.side_effect = [mock_resp_1, mock_resp_2]

    # Call 1: At 09:49:50 (bucket = 09:45 to 09:50)
    dt_1 = datetime(2026, 3, 20, 9, 49, 50, tzinfo=IST)
    with patch("apex3.broker.angel_data_provider.datetime") as mock_dt:
        mock_dt.now.return_value = dt_1
        mock_dt.fromisoformat = datetime.fromisoformat
        candles_1 = provider.get_historical("RELIANCE", "5minute")
        assert len(candles_1) == 2
        assert provider._smart_api.getCandleData.call_count == 1

        # Call 2: At 09:49:58 (same bucket) -> served from cache!
        dt_2 = datetime(2026, 3, 20, 9, 49, 58, tzinfo=IST)
        mock_dt.now.return_value = dt_2
        candles_cached = provider.get_historical("RELIANCE", "5minute")
        assert len(candles_cached) == 2
        assert provider._smart_api.getCandleData.call_count == 1  # No new API call

        # Call 3: Clock crosses boundary to 09:50:02 (new bucket!) -> cache invalidated immediately!
        dt_3 = datetime(2026, 3, 20, 9, 50, 2, tzinfo=IST)
        mock_dt.now.return_value = dt_3
        candles_new = provider.get_historical("RELIANCE", "5minute")
        assert len(candles_new) == 3
        assert provider._smart_api.getCandleData.call_count == 2  # New API call made!


