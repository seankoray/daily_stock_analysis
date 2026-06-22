from types import SimpleNamespace
import sys
import types

import numpy as np
import pandas as pd

if "fake_useragent" not in sys.modules:
    fake_useragent = types.ModuleType("fake_useragent")
    fake_useragent.UserAgent = type("UserAgent", (), {})
    sys.modules["fake_useragent"] = fake_useragent

from src.services.intraday_service import IntradayAnalysisService


def _bars(count: int = 60) -> pd.DataFrame:
    timestamps = pd.date_range("2026-06-22 09:35", periods=count, freq="5min")
    close = np.linspace(10.0, 10.8, count)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.full(count, 10000),
            "amount": close * 10000,
        }
    )


def test_aggregate_15m_uses_ohlcv_contract():
    result = IntradayAnalysisService.aggregate_15m(_bars(6))
    assert len(result) >= 2
    assert set(["open", "high", "low", "close", "volume", "amount"]).issubset(result.columns)
    assert result["volume"].sum() == 60000


def test_decision_caps_quantity_at_thirty_percent_and_board_lot():
    service = object.__new__(IntradayAnalysisService)
    service.MAX_POSITION_FRACTION = 0.30
    payload = service._decide(
        "600519",
        portfolio={
            "is_held": True,
            "sellable_quantity": 1000,
            "available_cash": 100000,
        },
        technical={
            "last_price": 10.8,
            "vwap": 10.4,
            "five_minute": {
                "available": True,
                "position": "above_upper",
                "lower": 10.0,
                "middle": 10.4,
                "upper": 10.7,
            },
            "fifteen_minute": {
                "available": True,
                "position": "upper_half",
                "middle_slope_pct": -0.1,
                "lower": 9.9,
            },
        },
        fresh=True,
        age_seconds=30,
        state=None,
    )
    assert payload["direction"] == "sell_then_buy"
    assert payload["suggested_quantity"] == 300


def test_completed_cycle_degrades_to_watch():
    service = object.__new__(IntradayAnalysisService)
    service.MAX_POSITION_FRACTION = 0.30
    payload = service._decide(
        "600519",
        portfolio={"is_held": True, "sellable_quantity": 1000, "available_cash": 100000},
        technical={},
        fresh=True,
        age_seconds=1,
        state=SimpleNamespace(cycle_count=1),
    )
    assert payload["status"] == "watch"
    assert "一轮" in payload["reason"]
