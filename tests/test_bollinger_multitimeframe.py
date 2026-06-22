from datetime import date

import numpy as np
import pandas as pd

from src.indicators.bollinger import MultiTimeframeBollAnalyzer
from src.stock_analyzer import StockTrendAnalyzer


def _daily_frame(days: int = 140, *, end: str = "2026-06-18") -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=days)
    close = np.linspace(10.0, 14.0, days) + np.sin(np.arange(days) / 5) * 0.25
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.linspace(1_000_000, 1_500_000, days),
            "amount": close * np.linspace(1_000_000, 1_500_000, days),
        }
    )


def test_daily_boll_uses_population_stddev():
    frame = _daily_frame()
    result = MultiTimeframeBollAnalyzer().analyze(frame)
    closes = frame["close"].tail(20)
    expected_middle = closes.mean()
    expected_upper = expected_middle + 2 * closes.std(ddof=0)
    assert result.daily.available
    assert result.daily.middle == pytest.approx(expected_middle)
    assert result.daily.upper == pytest.approx(expected_upper)


def test_weekly_boll_includes_and_marks_partial_week():
    result = MultiTimeframeBollAnalyzer().analyze(_daily_frame(end="2026-06-18"))
    assert result.weekly.available
    assert result.weekly.is_partial is True
    assert result.weekly.as_of == date(2026, 6, 19).isoformat()


def test_stock_analyzer_emits_v2_dual_scores():
    result = StockTrendAnalyzer().analyze(_daily_frame(), "600519")
    assert result.score_version == "v2"
    assert 0 <= result.medium_term_score <= 100
    assert 0 <= result.entry_timing_score <= 100
    assert result.daily_boll["available"] is True
    assert result.weekly_boll["available"] is True


import pytest
